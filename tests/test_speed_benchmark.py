import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark_inference_modes as bench


def fixture(root):
    rows = []
    for case, split, label in (("C1", "train", "중복"), ("C2", "val", "무관"),
                                ("C3", "test", "중복"), ("C4", "test", "유사·관련"),
                                ("C5", "test", "무관")):
        rows.append(dict(case_id=case, req_a_id=case + "A", req_a_text="본문 " + case + " A",
                         req_b_id=case + "B", req_b_text="본문 " + case + " B", label=label,
                         reason="정답 근거", diff_conditions="정답 차이", split=split))
    source = root / "pairs.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "training_info.json").write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return source, adapter, rows


def predicted(row, mode):
    labels = {"C1": "중복", "C2": "무관", "C3": "중복", "C4": "유사·관련", "C5": "무관"}
    output = {**row, "pred_label": labels[row["case_id"]], "status": "ok", "error": "",
              "pred_reason": "모델 근거", "pred_diff_conditions": "없음", "output_tokens": 100}
    if mode == "fast":
        output.update(pred_reason="", pred_diff_conditions="", output_tokens=8,
                      response_format="label_prefix", explanation_status="not_requested")
    return output


class FakeClassifier:
    metadata = {"model_id": "test-double", "generation": {"do_sample": True}}

    def __init__(self, config):
        self.config = config
        self.calls = []

    def predict_pairs(self, rows, batch_size):
        self.calls.append(("full", rows, batch_size))
        return [predicted(row, "full") for row in rows]

    def predict_relations(self, rows, batch_size):
        self.calls.append(("fast", rows, batch_size))
        return [predicted(row, "fast") for row in rows]


class SpeedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.sync = self.stack.enter_context(patch.object(bench, "cuda_synchronize"))
        self.reset = self.stack.enter_context(patch.object(bench, "cuda_reset_peak"))
        self.stack.enter_context(patch.object(bench, "cuda_peak_gib", return_value=1.25))
        self.stack.enter_context(patch.object(bench, "release_gpu_cache"))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def test_one_model_same_inputs_train_only_warmup_and_repeated_timings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, adapter, _ = fixture(root)
            instances = []

            def factory(config):
                result = FakeClassifier(config)
                instances.append(result)
                return result

            with patch.object(bench.time, "perf_counter", side_effect=[0, 2, 10, 11, 20, 21.5, 30, 31.2]):
                result = bench.run_benchmark(source, adapter, root / "speed", split="test", batch_size=2,
                                             classifier_factory=factory)
            self.assertEqual(len(instances), 1)
            model = instances[0]
            self.assertEqual(model.config.max_new_tokens, 768)
            self.assertTrue(model.config.local_files_only)
            self.assertEqual([(mode, rows[0]["case_id"]) for mode, rows, _ in model.calls[:2]],
                             [("full", "C1"), ("fast", "C1")])
            for mode, rows, size in model.calls:
                self.assertEqual(size, 2)
                self.assertTrue(all({"label", "reason", "diff_conditions"}.isdisjoint(row) for row in rows))
            measured = model.calls[2:]
            self.assertEqual([mode for mode, _, _ in measured], ["full"] * 2 + ["fast"] * 6)
            self.assertEqual([row for _, rows, _ in measured[:2] for row in rows],
                             [row for _, rows, _ in measured[2:4] for row in rows])
            self.assertEqual(self.sync.call_count, 9)
            self.assertEqual(self.reset.call_count, 4)
            self.assertEqual(result["full"]["median_seconds"], 2)
            self.assertEqual(result["fast"]["median_seconds"], 1.2)
            self.assertAlmostEqual(result["speedup"], 2 / 1.2)
            self.assertTrue(result["labels_identical"])
            self.assertTrue(result["accuracy_preserved"])
            self.assertTrue(result["fast"]["predictions_identical_across_runs"])
            self.assertEqual(result["full"]["metrics"]["accuracy"], 1)
            self.assertEqual(result["fast"]["runs"][0]["output_tokens_total"], 24)
            self.assertEqual(result["fast"]["runs"][0]["generated_explanations"], 0)
            self.assertEqual(result["fast"]["runs"][0]["omitted_explanations"], 3)
            self.assertEqual(result["settings"]["relation_max_new_tokens"], 32)
            self.assertEqual((root / "speed/source_snapshot.csv").read_bytes(), source.read_bytes())
            saved = json.loads((root / "speed/fast_03_pred.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["run"]["generation"]["do_sample"])
            self.assertEqual(saved["predictions"][0]["explanation_status"], "not_requested")
            self.assertTrue((root / "speed/performance.json").exists())

    def test_empty_explanations_require_prefix_contract_and_fallback_is_explicit(self):
        row = {"case_id": "C3", "req_a_id": "A", "req_a_text": "본문 A", "req_b_id": "B", "req_b_text": "본문 B"}
        fast = predicted(row, "fast")
        self.assertEqual(bench.checked_predictions([row], [fast], "fast")[0]["status"], "ok")
        self.assertEqual(bench.checked_predictions([row], [fast], "full")[0]["status"], "error")
        for modified in ({**fast, "explanation_status": "generated"},
                         {**fast, "response_format": "full_json"},
                         {**fast, "pred_reason": "설명을 지어냄"}):
            self.assertEqual(bench.checked_predictions([row], [modified], "fast")[0]["status"], "error")
        fallback = {**predicted(row, "full"), "relation_fallback": True,
                    "response_format": "full_json", "explanation_status": "generated"}
        self.assertEqual(bench.checked_predictions([row], [fallback], "fast")[0]["status"], "ok")

    def test_missing_invalid_or_changed_fast_outputs_do_not_claim_accuracy_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, adapter, _ = fixture(root)

            class BrokenFast(FakeClassifier):
                def predict_relations(self, rows, batch_size):
                    if rows[0]["split"] == "train":
                        return super().predict_relations(rows, batch_size)
                    return [predicted(rows[0], "fast"), {**predicted(rows[1], "fast"), "pred_label": ["무관"]}]

            report = bench.run_benchmark(source, adapter, root / "speed", split="test", fast_repeats=1,
                                          classifier_factory=BrokenFast)
            self.assertEqual(report["fast"]["metrics"]["total"], 3)
            self.assertEqual(report["fast"]["metrics"]["accuracy"], 1 / 3)
            self.assertEqual(report["fast"]["metrics"]["errors"], ["C4", "C5"])
            self.assertFalse(report["accuracy_preserved"])
            self.assertFalse(report["labels_identical"])
            self.assertTrue((root / "speed/performance.json").exists())

    def test_unknown_duplicate_and_modified_source_predictions_are_rejected(self):
        row = {"case_id": "C3", "req_a_id": "A", "req_a_text": "본문 A", "req_b_id": "B", "req_b_text": "본문 B"}
        good = predicted(row, "fast")
        for rows in ([good, good], [{**good, "case_id": "other"}], [{**good, "req_b_text": "changed"}]):
            with self.assertRaises(ValueError):
                bench.checked_predictions([row], rows, "fast")

    def test_dataset_binding_and_existing_output_are_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, adapter, _ = fixture(root)
            (adapter / "training_info.json").write_text(json.dumps({"dataset_sha256": "0" * 64}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                bench.run_benchmark(source, adapter, root / "speed", classifier_factory=lambda c: self.fail("loaded"))
            self.assertFalse((root / "speed").exists())
            (root / "speed").mkdir()
            (root / "speed/keep").write_text("original")
            with self.assertRaises(FileExistsError):
                bench.run_benchmark(source, adapter, root / "speed")
            self.assertEqual((root / "speed/keep").read_text(), "original")

    def test_source_changes_invalidate_final_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, adapter, _ = fixture(root)

            class Mutating(FakeClassifier):
                def predict_pairs(self, rows, batch_size):
                    if rows[0]["split"] == "val":
                        source.write_bytes(source.read_bytes() + b"\n")
                    return super().predict_pairs(rows, batch_size)

            with self.assertRaisesRegex(ValueError, "changed during benchmark"):
                bench.run_benchmark(source, adapter, root / "speed", classifier_factory=Mutating)
            self.assertFalse((root / "speed/performance.json").exists())

    def test_implementation_changes_invalidate_final_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, adapter, _ = fixture(root)
            changed = [False]
            original_hash = bench.sha256

            class CodeChanging(FakeClassifier):
                def predict_pairs(self, rows, batch_size):
                    if rows[0]["split"] == "val":
                        changed[0] = True
                    return super().predict_pairs(rows, batch_size)

            def file_hash(path):
                if Path(path).name == "inference.py" and changed[0]:
                    return "changed-code"
                return original_hash(path)

            with patch.object(bench, "sha256", side_effect=file_hash), \
                    self.assertRaisesRegex(ValueError, "implementation changed"):
                bench.run_benchmark(source, adapter, root / "speed", fast_repeats=1,
                                    classifier_factory=CodeChanging)
            self.assertFalse((root / "speed/performance.json").exists())


if __name__ == "__main__":
    unittest.main()
