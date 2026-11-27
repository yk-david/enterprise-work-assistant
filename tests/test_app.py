"""UI integration uses a fake classifier and the real document analysis pipeline."""

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from src import app


def create_adapter(directory, dataset_hash, completed=True):
    from src.inference import DEFAULT_MODEL, DEFAULT_REVISION, SYSTEM_PROMPT

    adapter = Path(directory) / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    info = {"dataset_sha256": dataset_hash, "base_model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "selected_epoch": 2, "split_counts": {"train": 88, "val": 9, "test": 18}}
    (adapter / "training_info.json").write_text(json.dumps(info), encoding="utf-8")
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"test-weights-placeholder")
    (adapter.parent / "training_result.json").write_text(json.dumps({**info, "completed": completed}), encoding="utf-8")
    return adapter


def comparison_report(models):
    entries = {}
    for report_key, model_key, accuracy in (("previous", "previous", .9), ("current", "latest", .8)):
        model = models[model_key]
        entries[report_key] = {"adapter_path": str(model["adapter"].resolve()),
                              "training_dataset_sha256": model["training"]["dataset_sha256"],
                              "inference_seconds": 58.127 if report_key == "previous" else 126.096,
                              "metrics": {"total": 18, "accuracy": accuracy, "macro_f1": accuracy, "errors": []}}
    return {"dataset": {"sha256": models["latest"]["training"]["dataset_sha256"], "test_count": 18},
            "generation": {"max_new_tokens": 384},
            "models": entries, "all_predictions_complete": True}


def speed_report(models):
    model = models["latest"]
    metrics = {"total": 18, "accuracy": 17 / 18, "macro_f1": .9218, "errors": []}
    return {"dataset_sha256": model["training"]["dataset_sha256"],
            "adapter_path": str(model["adapter"].resolve()),
            "adapter_weights_sha256": hashlib.sha256((model["adapter"] / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "split": "test", "test_count": 18,
            "settings": {"max_new_tokens": 768, "batch_size": 4, "relation_max_new_tokens": 32},
            "full": {"metrics": dict(metrics), "median_seconds": 120.0, "runs": [{}]},
            "fast": {"metrics": dict(metrics), "median_seconds": 12.0, "runs": [{}, {}, {}]},
            "labels_identical": True, "accuracy_preserved": True, "speedup": 10.0}


def write_speed_report(models, report):
    path = models["latest"]["adapter"].parent / "speed_optimization" / "test" / "performance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def sample_requirements():
    return [
        {"req_id": "R1", "source_req_id": "SER-01", "req_text": "● 접속 기록을 1년 보관한다.",
         "raw_req_text": "• 접속 기록을 1년 보관한다.", "doc_name": "sample.hwpx",
         "doc_type": "HWPX", "source_location": "표 1", "warnings": ["원문 대조"]},
        {"req_id": "R2", "source_req_id": "SER-02", "req_text": "● 접속 기록을 3년 보관한다.",
         "raw_req_text": "● 접속 기록을 3년 보관한다.", "doc_name": "sample.hwpx",
         "doc_type": "HWPX", "source_location": "표 2", "warnings": []},
    ]


class FakeClassifier:
    def __init__(self):
        self.metadata = {"test_double": True}
        self.calls = 0

    def predict_pairs(self, pairs, batch_size=4):
        self.calls += 1
        return [{**pair, "status": "ok", "pred_label": "유사·관련", "pred_reason": "보관 대상이 같다.",
                 "pred_diff_conditions": "1년과 3년", "error": ""} for pair in pairs]


class AppHelperTests(unittest.TestCase):
    def test_uploaded_hwpx_extension_name_and_temp_cleanup(self):
        observed = []

        def parse(path):
            observed.append(path)
            self.assertEqual(path.suffix, ".hwpx")
            self.assertEqual(path.read_bytes(), b"content")
            return sample_requirements()

        with patch.object(app, "parse_rfp", side_effect=parse):
            rows = app.parse_uploaded_file("uploaded.HWPX", b"content")
        self.assertTrue(all(row["doc_name"] == "uploaded.HWPX" for row in rows))
        self.assertFalse(observed[0].exists())

    def test_invalid_upload_never_reaches_parser(self):
        with patch.object(app, "parse_rfp") as parser:
            for name, content in (("bad.txt", b"x"), ("empty.pdf", b"")):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    app.parse_uploaded_file(name, content)
            parser.assert_not_called()

    def test_changed_content_or_removal_clears_both_results(self):
        state = {}
        with patch.object(app.st, "session_state", state):
            app.sync_uploaded_file("same.pdf", b"one")
            state.update(requirements=[1], analysis_result={"old": True}, requirement_id="R1")
            app.sync_uploaded_file("same.pdf", b"one")
            self.assertIn("analysis_result", state)
            app.sync_uploaded_file("same.pdf", b"two")
            self.assertNotIn("analysis_result", state)
            self.assertNotIn("requirements", state)
            state.update(requirements=[2], analysis_result={"old": True})
            app.sync_uploaded_file(None, None)
            self.assertNotIn("analysis_result", state)
            self.assertNotIn("requirements", state)

    def test_csv_and_json_keep_korean_conditions_and_protect_formula_cells(self):
        group = {"group_id": "G1", "group_name": "=1+1", "summary": "● 기록, 보관\n\"1년\"과 3년",
                 "req_ids": ["R1", "R2"], "diff_conditions": "- 1년 / 3년",
                 "source_locations": ["표 1", "표 2"]}
        data = app.group_csv(app.group_rows([group]))
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        row = next(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        self.assertEqual(row["req_ids"], "R1|R2")
        self.assertEqual(row["summary"], group["summary"])
        self.assertEqual(row["group_name"], "'=1+1")
        self.assertEqual(row["diff_conditions"], "'- 1년 / 3년")
        self.assertEqual(json.loads(app.analysis_json({"groups": [group]}))["groups"][0], group)

    def test_single_requirement_needs_no_model_and_keeps_source(self):
        with patch.object(app, "load_classifier") as loader:
            result = app.run_analysis(sample_requirements()[:1], model_key="base")
        loader.assert_not_called()
        self.assertEqual(result["stats"]["pair_count"], 0)
        self.assertEqual(result["groups"][0]["req_ids"], ["R1"])
        self.assertIn("1년", result["groups"][0]["summary"])

    def test_model_switch_releases_previous_model_before_loading_and_can_retry_failure(self):
        from src import inference

        slot = app._ModelSlot()
        calls = []
        previous = None

        def construct(config):
            nonlocal previous
            if previous is not None:
                self.assertIsNone(previous(), "old classifier must be released before replacement")
            calls.append(config)
            self.assertTrue(config.local_files_only)
            if len(calls) == 2:
                raise RuntimeError("load failed")
            classifier = FakeClassifier()
            previous = weakref.ref(classifier)
            return classifier

        catalog = {"base": {"adapter": None}, "previous": {"adapter": app.PREVIOUS_ADAPTER},
                   "latest": {"adapter": app.LATEST_ADAPTER}}
        with patch.object(inference, "RequirementClassifier", side_effect=construct), \
                patch.object(app, "available_models", return_value=catalog):
            slot.load("base")
            slot.load("base")
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                slot.load("previous")
            self.assertIsNone(slot.classifier)
            slot.load("previous")
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[-1].adapter, app.PREVIOUS_ADAPTER)
            slot.load("latest")
            self.assertEqual(calls[-1].adapter, app.LATEST_ADAPTER)
            previous_count = len(calls)
            slot.load("latest", max_new_tokens=768)
            self.assertEqual(len(calls), previous_count + 1)
            self.assertEqual(calls[-1].max_new_tokens, 768)
            slot.load("latest", max_new_tokens=768)
            self.assertEqual(len(calls), previous_count + 1)
            with self.assertRaisesRegex(ValueError, "1 이상의 정수"):
                slot.load("latest", max_new_tokens=0)

    def test_latest_is_first_only_when_completed_and_metadata_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = create_adapter(Path(temp) / "previous", "b" * 64)
            latest = create_adapter(Path(temp) / "latest", "a" * 64, completed=False)
            with patch.object(app, "LATEST_ADAPTER", latest), patch.object(app, "PREVIOUS_ADAPTER", previous):
                self.assertEqual(list(app.available_models()), ["previous", "base"])
                create_adapter(latest.parent, "a" * 64)
                self.assertEqual(list(app.available_models()), ["latest", "previous", "base"])
                result_path = latest.parent / "training_result.json"
                result = json.loads(result_path.read_text())
                result["dataset_sha256"] = "c" * 64
                result_path.write_text(json.dumps(result))
                self.assertEqual(list(app.available_models()), ["previous", "base"])
                (previous / "adapter_model.safetensors").unlink()
                self.assertEqual(list(app.available_models()), ["base"])
                with self.assertRaisesRegex(ValueError, "완성되지 않았습니다"):
                    app.selected_model("latest")

    def test_comparison_must_match_current_model_and_data_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = create_adapter(Path(temp) / "previous", "b" * 64)
            latest = create_adapter(Path(temp) / "latest", "a" * 64)
            comparison = Path(temp) / "comparison.json"
            with patch.object(app, "LATEST_ADAPTER", latest), patch.object(app, "PREVIOUS_ADAPTER", previous), \
                    patch.object(app, "COMPARISON_PATH", comparison):
                models = app.available_models()
                report = comparison_report(models)
                comparison.write_text(json.dumps(report))
                self.assertEqual(app.read_model_comparison(models), report)
                for field, value in (("dataset", {"sha256": "c" * 64, "test_count": 18}),
                                     ("models", {"current": report["models"]["current"]})):
                    comparison.write_text(json.dumps({**report, field: value}))
                    self.assertIsNone(app.read_model_comparison(models))
                report["models"]["current"]["metrics"]["accuracy"] = float("nan")
                comparison.write_text(json.dumps(report))
                self.assertIsNone(app.read_model_comparison(models))

    def test_speed_comparison_requires_same_latest_weights_data_and_test_split(self):
        with tempfile.TemporaryDirectory() as temp:
            latest = create_adapter(Path(temp) / "latest", "a" * 64)
            with patch.object(app, "LATEST_ADAPTER", latest):
                models = app.available_models()
                self.assertIsNone(app.read_speed_comparison(models))
                report = speed_report(models)
                write_speed_report(models, report)
                self.assertEqual(app.read_speed_comparison(models), report)
                for key, value in (("dataset_sha256", "b" * 64), ("adapter_path", str(latest.parent)),
                                   ("adapter_weights_sha256", "f" * 64), ("split", "val"),
                                   ("labels_identical", "true"), ("speedup", float("nan"))):
                    write_speed_report(models, {**report, key: value})
                    self.assertIsNone(app.read_speed_comparison(models), key)
                write_speed_report(models, report)
                (latest / "adapter_model.safetensors").write_bytes(b"different-weights")
                self.assertIsNone(app.read_speed_comparison(models))


class AppRuntimeTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest

        self.temp = tempfile.TemporaryDirectory()
        previous = create_adapter(Path(self.temp.name) / "previous", "b" * 64)
        latest = create_adapter(Path(self.temp.name) / "latest", "a" * 64)
        self.patches = [patch.object(app, "PREVIOUS_ADAPTER", previous), patch.object(app, "LATEST_ADAPTER", latest),
                        patch.object(app, "COMPARISON_PATH", Path(self.temp.name) / "comparison.json")]
        for item in self.patches:
            item.start()
        self.at = AppTest.from_string("from src.app import main\nmain()", default_timeout=15)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def assert_page_ok(self):
        self.assertFalse(self.at.exception, [item.message for item in self.at.exception])

    def upload_and_extract(self):
        self.at.run()
        self.at.file_uploader[0].set_value(("sample.hwpx", b"content", "application/octet-stream")).run()
        self.at.button[0].click().run()
        self.assert_page_ok()

    def test_real_pipeline_result_download_and_rerun_without_reanalysis(self):
        classifier = FakeClassifier()
        with patch.object(app, "parse_rfp", return_value=sample_requirements()) as parser, \
                patch.object(app, "load_classifier", return_value=classifier) as loader:
            self.upload_and_extract()
            self.at.button[1].click().run()
            self.assert_page_ok()
            self.assertFalse(self.at.error)
            result = self.at.session_state["analysis_result"]
            self.assertEqual(result["groups"][0]["req_ids"], ["R1", "R2"])
            self.assertEqual(result["stats"]["successful_pairs"], 1)
            self.assertEqual(len(self.at.download_button), 2)
            self.assertTrue(any("1년" in item.value and "3년" in item.value for item in self.at.text))
            self.at.selectbox(key="requirement_id").set_value("R2").run()
            self.at.download_button[0].click().run()
            self.assert_page_ok()
            self.assertEqual(parser.call_count, 1)
            self.assertEqual(loader.call_count, 1)
            loader.assert_called_once_with("latest", max_new_tokens=384)
            self.assertEqual(self.at.selectbox(key="analysis_model_choice").value, "latest")
            self.assertEqual(classifier.calls, 1)
            self.assertEqual(self.at.session_state["analysis_result"], result)

    def test_fast_default_and_detail_toggle_keep_group_summary_and_reset_results(self):
        classifier = FakeClassifier()
        calls = []

        def fast(pairs, batch_size=4):
            calls.append("fast")
            return [{**pair, "status": "ok", "pred_label": "유사·관련", "pred_reason": "",
                     "pred_diff_conditions": "", "explanation_status": "not_requested",
                     "response_format": "label_prefix", "error": ""} for pair in pairs]

        classifier.predict_relations = fast
        with patch.object(app, "parse_rfp", return_value=sample_requirements()), \
                patch.object(app, "load_classifier", return_value=classifier):
            self.upload_and_extract()
            self.assertFalse(self.at.checkbox(key="analysis_detailed_pairs").value)
            self.at.button[1].click().run()
            self.assert_page_ok()
            result = self.at.session_state["analysis_result"]
            self.assertEqual(calls, ["fast"])
            self.assertEqual(classifier.calls, 0)
            self.assertEqual(result["run"]["pair_explanations"], "not_requested")
            self.assertTrue(any("1년" in item.value and "3년" in item.value for item in self.at.text))
            self.assertEqual(len(self.at.download_button), 2)
            groups = result["groups"]
            self.at.checkbox(key="analysis_detailed_pairs").check().run()
            self.assertNotIn("analysis_result", self.at.session_state)
            self.assertIn("requirements", self.at.session_state)
            self.at.button[1].click().run()
            self.assert_page_ok()
            self.assertEqual(calls, ["fast"])
            self.assertEqual(classifier.calls, 1)
            result = self.at.session_state["analysis_result"]
            self.assertEqual(result["run"]["pair_explanations"], "generated")
            self.assertEqual(result["groups"], groups)

    def test_model_change_file_change_and_analysis_failure_clear_old_results(self):
        with patch.object(app, "parse_rfp", side_effect=lambda _: copy.deepcopy(sample_requirements())), \
                patch.object(app, "load_classifier", return_value=FakeClassifier()):
            self.upload_and_extract()
            self.at.button[1].click().run()
            self.at.selectbox(key="analysis_model_choice").set_value("previous").run()
            self.assertNotIn("analysis_result", self.at.session_state)
            self.at.button[1].click().run()
            self.at.selectbox(key="analysis_model_choice").set_value("base").run()
            self.assertNotIn("analysis_result", self.at.session_state)
            self.assertIn("requirements", self.at.session_state)
            self.at.button[1].click().run()
            self.at.selectbox(key="analysis_max_new_tokens").set_value(768).run()
            self.assertNotIn("analysis_result", self.at.session_state)
            self.assertIn("requirements", self.at.session_state)
            self.at.button[1].click().run()
            with patch.object(app, "run_analysis", side_effect=RuntimeError("test model failure")):
                self.at.button[1].click().run()
            self.assert_page_ok()
            self.assertNotIn("analysis_result", self.at.session_state)
            self.assertTrue(any("test model failure" in error.value for error in self.at.error))
            self.at.button[1].click().run()
            self.at.file_uploader[0].set_value(("sample.hwpx", b"changed", "application/octet-stream")).run()
            self.assertNotIn("requirements", self.at.session_state)
            self.assertNotIn("analysis_result", self.at.session_state)
            self.assertTrue(self.at.button[1].disabled)
            self.at.file_uploader[0].set_value(None).run()
            self.assertTrue(self.at.button[0].disabled)
            self.assert_page_ok()

    def test_partial_pair_error_visible_and_every_requirement_preserved(self):
        classifier = FakeClassifier()

        def failed(pairs, batch_size=4):
            return [{**pair, "status": "error", "error": "입력 토큰 초과"} for pair in pairs]

        classifier.predict_pairs = failed
        with patch.object(app, "parse_rfp", return_value=sample_requirements()), \
                patch.object(app, "load_classifier", return_value=classifier):
            self.upload_and_extract()
            self.at.button[1].click().run()
            self.assert_page_ok()
            result = self.at.session_state["analysis_result"]
            self.assertEqual(result["stats"]["failed_pairs"], 1)
            self.assertFalse(self.at.success)
            self.assertTrue(any("일부 비교를 완료하지 못했습니다" in item.value for item in self.at.warning))
            self.assertEqual({req_id for group in result["groups"] for req_id in group["req_ids"]}, {"R1", "R2"})
            self.assertTrue(any("비교하지 못한 쌍 1건" == item.label for item in self.at.expander))
            self.assertEqual(len(self.at.download_button), 2)

    def test_regression_comparison_is_visible_without_claiming_improvement(self):
        models = app.available_models()
        app.COMPARISON_PATH.write_text(json.dumps(comparison_report(models)))
        self.at.run()
        self.assert_page_ok()
        self.assertEqual(self.at.selectbox(key="analysis_model_choice").value, "latest")
        self.assertTrue(any("기존 모델보다 낮습니다" in item.value for item in self.at.warning))
        self.assertTrue(any("같은 평가 데이터의 모델 성능 비교" == item.label for item in self.at.expander))
        comparison_table = self.at.dataframe[0].value
        self.assertIn("추론 시간(초)", comparison_table.columns)
        self.assertEqual(comparison_table["추론 시간(초)"].tolist(), [58.1, 126.1])
        self.assertTrue(any("모델 로딩을 제외" in item.value for item in self.at.caption))
        self.assertTrue(any("추론 시간이 더 길었습니다" in item.value for item in self.at.caption))
        self.at.selectbox(key="analysis_max_new_tokens").set_value(768).run()
        self.assert_page_ok()
        self.assertTrue(any("응답 최대 384토큰" in item.value and "768토큰" in item.value for item in self.at.caption))
        self.assertTrue(any("다시 계산되지 않습니다" in item.value for item in self.at.caption))

    def test_speed_comparison_shows_agreement_sample_size_and_distinct_explanations(self):
        models = app.available_models()
        report = speed_report(models)
        write_speed_report(models, report)
        self.at.run()
        self.assert_page_ok()
        self.assertTrue(any(item.label == "최신 모델의 빠른 분석 성능 검증" for item in self.at.expander))
        self.assertTrue(any("18쌍" in item.value and "모든 판정 일치" in item.value for item in self.at.caption))
        self.assertTrue(any("상세 설명은 1회" in item.value and "3회" in item.value for item in self.at.caption))
        self.assertTrue(any("쌍별 추론만" in item.value and "그룹 요약 시간을 포함하지 않습니다" in item.value for item in self.at.caption))
        table = self.at.dataframe[0].value
        self.assertEqual(table["추론 시간(초)"].tolist(), [120.0, 12.0])
        self.assertEqual(table["정확도"].tolist(), ["94.4%", "94.4%"])
        write_speed_report(models, {**report, "labels_identical": False, "accuracy_preserved": False})
        self.at.run()
        self.assert_page_ok()
        self.assertTrue(any("보존 조건이 일치하지 않습니다" in item.value for item in self.at.warning))


if __name__ == "__main__":
    unittest.main()
