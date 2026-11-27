import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.compare_model_versions import normalize_predictions, run_comparison


def pair(case, split, label):
    return dict(case_id=case, req_a_id=case + "A", req_a_text="본문 " + case + " A",
                req_b_id=case + "B", req_b_text="본문 " + case + " B", label=label,
                reason="사람 정답 근거", diff_conditions="사람 정답 차이", split=split)


def fixture(root):
    rows = [pair("C1", "train", "중복"), pair("C2", "val", "무관"),
            pair("C3", "test", "중복"), pair("C4", "test", "유사·관련"), pair("C5", "test", "무관")]
    source = root / "gold.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for name, dataset_hash in (("previous", "0" * 64), ("current", digest)):
        directory = root / name
        directory.mkdir()
        (directory / "training_info.json").write_text(json.dumps({"dataset_sha256": dataset_hash}), encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(name.encode())
    return source, root / "previous", root / "current", rows


def prediction(source, label="무관"):
    return {**source, "pred_label": label, "pred_reason": "예측 근거", "pred_diff_conditions": "없음",
            "status": "ok", "error": ""}


class ModelComparisonTests(unittest.TestCase):
    def test_same_inputs_greedy_settings_no_gold_leakage_and_sequential_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, previous, current, gold = fixture(root)
            gold_by_id = {row["case_id"]: row["label"] for row in gold}
            configs, received, live, events = [], {}, [], []
            outer = self

            class Fake:
                def __init__(self, config):
                    outer.assertEqual(live, [])
                    self.name = config.adapter.name if config.adapter else "base"
                    live.append(self.name)
                    events.append(("load", self.name))
                    configs.append(config)
                    self.metadata = {"model_id": "fake", "generation": {"do_sample": True}}
                    received[self.name] = []

                def predict_pairs(self, rows, batch_size):
                    outer.assertEqual(batch_size, 2)
                    for row in rows:
                        outer.assertTrue({"label", "reason", "diff_conditions"}.isdisjoint(row))
                    received[self.name].extend(rows)
                    return [prediction(row, gold_by_id[row["case_id"]] if self.name == "current" else "무관") for row in rows]

                def __del__(self):
                    events.append(("release", self.name))
                    live.remove(self.name)

            with redirect_stdout(io.StringIO()):
                result = run_comparison(source, previous, current, root / "evaluation", include_base=True,
                                        batch_size=2, classifier_factory=Fake)
            self.assertEqual(events, [(action, name) for name in ("base", "previous", "current") for action in ("load", "release")])
            self.assertEqual(received["base"], received["previous"])
            self.assertEqual(received["previous"], received["current"])
            self.assertTrue(all(config.local_files_only for config in configs))
            self.assertEqual(len({(c.max_input_tokens, c.max_new_tokens, c.seed) for c in configs}), 1)
            self.assertEqual(result["models"]["current"]["metrics"]["accuracy"], 1)
            self.assertEqual(result["models"]["previous"]["metrics"]["accuracy"], 1 / 3)
            self.assertEqual(result["dataset"]["test_support"], {"중복": 1, "유사·관련": 1, "무관": 1})
            self.assertTrue(result["all_predictions_complete"])
            saved = json.loads((root / "evaluation/current_test_pred.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["run"]["generation"]["do_sample"])
            self.assertEqual(saved["run"]["generation"]["batch_size"], 2)
            self.assertEqual((root / "evaluation/gold_snapshot.csv").read_bytes(), source.read_bytes())
            self.assertEqual(len(result["case_comparison"]), 3)

    def test_missing_and_invalid_predictions_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, previous, current, _ = fixture(root)

            class Fake:
                metadata = {}
                def __init__(self, config):
                    pass
                def predict_pairs(self, rows, batch_size):
                    return [prediction(rows[0], "중복"), prediction(rows[1], ["유사·관련"])]

            with redirect_stdout(io.StringIO()):
                result = run_comparison(source, previous, current, root / "eval", classifier_factory=Fake)
            for name in ("previous", "current"):
                metrics = result["models"][name]["metrics"]
                self.assertEqual(metrics["accuracy"], 1 / 3)
                self.assertEqual(metrics["total"], 3)
                self.assertEqual(metrics["errors"], ["C4", "C5"])
                self.assertEqual(metrics["confusion_matrix"]["유사·관련"]["ERROR"], 1)
                saved = json.loads((root / f"eval/{name}_test_pred.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["failed"], 2)
                self.assertEqual(saved["total"], 3)
            self.assertFalse(result["all_predictions_complete"])

    def test_current_dataset_hash_mismatch_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, previous, current, _ = fixture(root)
            (current / "training_info.json").write_text(json.dumps({"dataset_sha256": "1" * 64}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not trained"):
                run_comparison(source, previous, current, root / "eval", classifier_factory=lambda config: self.fail("model loaded"))
            self.assertFalse((root / "eval").exists())

    def test_changed_dataset_during_inference_invalidates_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, previous, current, _ = fixture(root)

            class Fake:
                metadata = {}
                def __init__(self, config):
                    pass
                def predict_pairs(self, rows, batch_size):
                    source.write_bytes(source.read_bytes() + b"\r\n")
                    return [prediction(row) for row in rows]

            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "changed during evaluation"):
                run_comparison(source, previous, current, root / "eval", classifier_factory=Fake)
            self.assertFalse((root / "eval/comparison.json").exists())

    def test_original_evaluation_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, previous, current, _ = fixture(root)
            output = root / "eval"
            output.mkdir()
            (output / "keep.txt").write_text("original")
            with self.assertRaises(FileExistsError):
                run_comparison(source, previous, current, output)
            self.assertEqual((output / "keep.txt").read_text(), "original")

    def test_unknown_duplicate_and_modified_source_predictions_rejected(self):
        source = pair("C", "test", "무관")
        for predictions in ([prediction(source), prediction(source)],
                            [prediction({**source, "case_id": "other"})],
                            [prediction({**source, "req_a_text": "changed"})]):
            with self.assertRaises(ValueError):
                normalize_predictions([source], predictions)


if __name__ == "__main__":
    unittest.main()
