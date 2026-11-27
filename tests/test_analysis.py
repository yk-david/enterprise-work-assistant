import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.analysis import (analyze_requirements, group_csv_bytes, main, make_pairs,
                          save_analysis, validate_requirements)


def requirements():
    return [dict(req_id=rid, req_text=text, raw_req_text=text,
                 doc_name="제안요청서.pdf", doc_type="PDF", source_location=f"p.{i}",
                 source_req_id=f"SFR-00{i}", warnings=[])
            for i, (rid, text) in enumerate([
                ("A", "● 장애는 접수 후 2시간 이내 처리한다."),
                ("B", "● 장애는 접수 후 4시간 이내 처리한다."),
                ("C", "● 보안 교육을 월 1회 실시한다.")], start=1)]


class Classifier:
    metadata = {"model_id": "test-double"}

    def predict_pair(self, pair):
        label = "유사·관련" if pair["req_a_id"] == "A" and pair["req_b_id"] == "B" else "무관"
        return {**pair, "pred_label": label, "pred_reason": "원문 비교",
                "pred_diff_conditions": "시간 조건 차이", "status": "ok", "error": ""}


class DocumentAnalysisTests(unittest.TestCase):
    def test_fast_relation_mode_preserves_groups_conditions_and_explicit_explanation_state(self):
        calls = []

        def predict_relations(pairs, batch_size):
            calls.append("fast")
            return [{**Classifier().predict_pair(pair), "pred_reason": "", "pred_diff_conditions": "",
                     "explanation_status": "not_requested", "response_format": "label_prefix"} for pair in pairs]

        def predict_pairs(pairs, batch_size):
            calls.append("detailed")
            return [Classifier().predict_pair(pair) for pair in pairs]

        classifier = SimpleNamespace(predict_relations=predict_relations, predict_pairs=predict_pairs,
                                     config=SimpleNamespace(max_new_tokens=16))
        fast = analyze_requirements(requirements(), classifier)
        self.assertEqual(calls, ["fast"])
        self.assertEqual(fast["run"]["pair_inference_method"], "predict_relations")
        self.assertEqual(fast["run"]["pair_explanations"], "not_requested")
        self.assertEqual(fast["run"]["document_generation"]["relation_max_new_tokens"], 16)
        self.assertEqual(fast["run"]["pair_explanation_counts"], {"not_requested": 3, "generated": 0})
        self.assertTrue(all(row["pred_reason"] == "" for row in fast["pair_results"]))
        self.assertEqual(fast["stats"]["failed_pairs"], 0)
        for field in ("pair_inference_seconds", "grouping_seconds", "summary_seconds"):
            self.assertGreaterEqual(fast["stats"][field], 0)
        detailed = analyze_requirements(requirements(), classifier, detailed_pairs=True)
        self.assertEqual(calls, ["fast", "detailed"])
        self.assertEqual(detailed["run"]["pair_explanations"], "generated")
        self.assertIsNone(detailed["run"]["document_generation"]["relation_max_new_tokens"])
        self.assertEqual(fast["groups"], detailed["groups"])
        shown = "\n".join(g["summary"] + g["diff_conditions"] for g in fast["groups"])
        self.assertTrue(all(phrase in shown for phrase in ("2시간", "4시간", "월 1회")))

    def test_empty_explanations_require_exact_fast_protocol_and_still_validate_sources(self):
        base = {"pred_reason": "", "pred_diff_conditions": "", "explanation_status": "not_requested",
                "response_format": "label_prefix"}
        for override in ({"explanation_status": "generated"}, {"response_format": "json"},
                         {"pred_reason": None}, {"req_a_text": "changed"}, {"pred_label": ["중복"]}):
            def predict(pairs, batch_size):
                return [{**Classifier().predict_pair(pair), **base, **override} for pair in pairs]
            report = analyze_requirements(requirements(), SimpleNamespace(predict_relations=predict))
            self.assertEqual(report["stats"]["failed_pairs"], 3)
        forged = SimpleNamespace(predict_pairs=lambda pairs, batch_size: [
            {**Classifier().predict_pair(pair), **base} for pair in pairs])
        report = analyze_requirements(requirements(), forged, detailed_pairs=True)
        self.assertEqual(report["stats"]["failed_pairs"], 3)

    def test_fast_method_can_fall_back_to_real_detailed_responses(self):
        classifier = SimpleNamespace(predict_relations=lambda pairs, batch_size: [Classifier().predict_pair(pair) for pair in pairs])
        report = analyze_requirements(requirements(), classifier)
        self.assertEqual(report["stats"]["failed_pairs"], 0)
        self.assertEqual(report["run"]["pair_explanation_counts"], {"not_requested": 0, "generated": 3})

    def test_full_flow_preserves_all_sources_and_conditions(self):
        rows = requirements()
        before = copy.deepcopy(rows)
        progress = []
        report = analyze_requirements(rows, Classifier(), lambda done, total: progress.append((done, total)), batch_size=2)
        self.assertEqual(rows, before)
        self.assertEqual(report["stats"]["pair_count"], 3)
        self.assertEqual(report["stats"]["failed_pairs"], 0)
        self.assertEqual(progress, [(2, 3), (3, 3)])
        self.assertEqual({frozenset(g["req_ids"]) for g in report["groups"]}, {frozenset("AB"), frozenset("C")})
        text = "\n".join(g["summary"] + "\n" + g["diff_conditions"] for g in report["groups"])
        for phrase in ("2시간", "4시간", "월 1회"):
            self.assertIn(phrase, text)
        self.assertEqual(report["pair_results"][0]["req_a_source_location"], "p.1")
        self.assertEqual(report["requirements"][0]["raw_req_text"], rows[0]["raw_req_text"])

    def test_empty_and_single_input_need_no_model(self):
        for rows, count in (([], 0), (requirements()[:1], 1)):
            report = analyze_requirements(rows, None)
            self.assertEqual(report["stats"]["pair_count"], 0)
            self.assertEqual(len(report["groups"]), count)

    def test_invalid_input_rejected_before_inference(self):
        rows = requirements()
        for invalid in (rows + [rows[0]], [{**rows[0], "req_text": "#NAME?"}],
                        [{**rows[0], "doc_type": None}], {"requirements": rows}):
            with self.assertRaises(ValueError):
                validate_requirements(invalid)
        self.assertEqual(len(make_pairs(rows)), 3)

    def test_error_and_mismatched_result_never_create_edges(self):
        rows = requirements()
        for bad in (lambda pair: {**Classifier().predict_pair(pair), "req_a_text": "changed"},
                    lambda pair: {**Classifier().predict_pair(pair), "pred_label": ["중복"]},
                    lambda pair: {**pair, "status": "error", "pred_label": "중복", "error": "bad JSON"}):
            report = analyze_requirements(rows, SimpleNamespace(predict_pair=bad))
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["stats"]["failed_pairs"], 3)
            self.assertTrue(all(len(g["req_ids"]) == 1 for g in report["groups"]))
            self.assertTrue(all(r["pred_label"] == "" for r in report["pair_results"]))

    def test_batch_preserves_order_and_bad_batch_remains_visible(self):
        classifier = SimpleNamespace(predict_pairs=lambda pairs, batch_size: [Classifier().predict_pair(p) for p in pairs])
        report = analyze_requirements(requirements(), classifier, batch_size=2)
        self.assertEqual([r["case_id"] for r in report["pair_results"]], ["DOC-000001", "DOC-000002", "DOC-000003"])
        classifier.predict_pairs = lambda pairs, batch_size: []
        report = analyze_requirements(requirements(), classifier)
        self.assertEqual(report["stats"]["failed_pairs"], 3)
        self.assertEqual(len(report["groups"]), 3)

    def test_csv_roundtrip_and_json_preservation(self):
        report = analyze_requirements(requirements(), Classifier())
        report["groups"][0]["group_name"] = '=HYPERLINK("test")'
        data = group_csv_bytes(report["groups"])
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        loaded = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        self.assertEqual(loaded[0]["group_name"], "'" + report["groups"][0]["group_name"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "analysis.json"
            save_analysis(report, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(output.with_suffix(".csv").read_bytes(), data)
            with self.assertRaises(FileExistsError):
                save_analysis(report, output)
            with self.assertRaisesRegex(ValueError, "입력 파일"):
                save_analysis(report, output, output, overwrite=True)

    def test_cli_json_uses_actual_analysis_and_saves_both_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.json", Path(temp) / "analysis.json"
            source.write_text(json.dumps(requirements(), ensure_ascii=False), encoding="utf-8")
            with patch("src.analysis.RequirementClassifier", return_value=Classifier()):
                code = main(["--input", str(source), "--out", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["stats"]["pair_count"], 3)
            self.assertTrue(output.with_suffix(".csv").exists())

    def test_cli_detailed_flag_dispatches_full_predictions(self):
        calls = []
        fake = SimpleNamespace(
            predict_relations=lambda *args, **kwargs: self.fail("Detailed CLI must not use the label-prefix path"),
            predict_pairs=lambda pairs, batch_size: calls.append(batch_size) or [Classifier().predict_pair(pair) for pair in pairs])
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.json", Path(temp) / "result.json"
            source.write_text(json.dumps(requirements()), encoding="utf-8")
            with patch("src.analysis.RequirementClassifier", return_value=fake):
                self.assertEqual(main(["--input", str(source), "--out", str(output), "--detailed-pairs"]), 0)
            self.assertEqual(calls, [4])
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["run"]["detailed_pairs_requested"])


if __name__ == "__main__":
    unittest.main()
