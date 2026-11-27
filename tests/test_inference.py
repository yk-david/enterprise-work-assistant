import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import inference


def pair(case_id="C1", split="test"):
    return {"case_id": case_id, "req_a_id": "R1", "req_a_text": "● 접속 이력\n1년 보관",
            "req_b_id": "R2", "req_b_text": "접속 이력 3년 보관", "split": split}


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class InferenceTests(unittest.TestCase):
    def test_valid_json_and_fenced_json(self):
        answer = '{"label":"유사·관련","reason":"보관 대상이 같다.","diff_conditions":"1년과 3년"}'
        for text in (answer, f"```json\n{answer}\n```"):
            result = inference.parse_prediction(text)
            self.assertEqual(result["pred_label"], "유사·관련")
            self.assertEqual(result["pred_diff_conditions"], "1년과 3년")

    def test_invalid_model_outputs_are_not_relabeled(self):
        invalid = [
            '설명 {"label":"중복","reason":"r","diff_conditions":"없음"}',
            '{"label":"유사","reason":"r","diff_conditions":"없음"}',
            '{"label":"중복","reason":"","diff_conditions":"없음"}',
            '{"label":"중복","reason":"r"}',
            '{"label":"중복","label":"무관","reason":"r","diff_conditions":"없음"}',
            '{"label":"중복","reason":null,"diff_conditions":"없음"}',
            '[]', '{"label":',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                inference.parse_prediction(text)

    def test_prompt_excludes_ground_truth_and_preserves_document_text(self):
        row = {**pair(), "label": "SECRET_LABEL", "reason": "SECRET_REASON", "diff_conditions": "SECRET_DIFF"}
        messages = inference.build_messages(row)
        joined = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("SECRET", joined)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["requirement_a"]["text"], row["req_a_text"])
        self.assertNotIn("id", payload["requirement_a"])

    def test_csv_split_and_multiline_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pairs.csv"
            write_csv(path, [pair("TRAIN", "train"), pair("TEST", "test")])
            rows = inference.load_pairs_csv(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "TEST")
        self.assertEqual(rows[0]["req_a_text"], pair()["req_a_text"])

    def test_csv_rejects_duplicate_ids_and_empty_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pairs.csv"
            write_csv(path, [pair(), pair()])
            with self.assertRaisesRegex(ValueError, "중복 case_id"):
                inference.load_pairs_csv(path)
            write_csv(path, [pair(split="train")])
            with self.assertRaisesRegex(ValueError, "해당하는"):
                inference.load_pairs_csv(path)

    def test_truncated_csv_is_rejected_without_using_partial_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "truncated.csv"
            path.write_text('case_id,req_a_id,req_a_text,req_b_id,req_b_text,split\nC1,R1,본문,R2,"잘린 본문', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "잘린 파일"):
                inference.load_pairs_csv(path)

    def test_missing_columns_or_self_pair_fail_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "pairs.csv"
            for rows in ([{"case_id": "C1"}], [{**pair(), "req_b_id": "R1"}]):
                write_csv(source, rows)
                with patch.object(inference, "RequirementClassifier") as model, redirect_stderr(io.StringIO()):
                    self.assertEqual(inference.main(["--pairs", str(source), "--out", str(root / "pred.csv")]), 1)
                    model.assert_not_called()

    def test_parser_json_keeps_source_location_and_accepts_warning_lists(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "requirements.json"
            rows = [
                {"req_id": "R1", "req_text": "● 본문", "doc_name": "문서.pdf", "doc_type": "PDF",
                 "source_req_id": "PMR-01", "source_location": "p.30", "warnings": ["미확인 기호"]},
                {"req_id": "R2", "req_text": "다른 본문", "doc_name": "문서.pdf", "source_location": "p.32–33"},
            ]
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            result = inference.pair_from_requirements(path, "R1", "R2")
            self.assertEqual(result["req_a_source_req_id"], "PMR-01")
            self.assertEqual(result["req_b_source_location"], "p.32–33")
            self.assertEqual(result["req_a_text"], "● 본문")
            path.write_text(json.dumps(rows + [rows[0]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "중복 req_id"):
                inference.pair_from_requirements(path, "R1", "R2")

    def test_existing_results_and_input_cannot_be_overwritten_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "original.json"
            source.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "원본 입력"):
                inference.check_output_paths(root / "original.csv", source, overwrite=True)
            output = root / "pred.csv"
            output.write_text("previous", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                inference.check_output_paths(output, source)
            self.assertEqual(output.read_text(), "previous")

    def test_cli_saves_failed_rows_and_returns_failure(self):
        def predict(row):
            result = {**row, "pred_label": "무관", "pred_reason": "다른 기능", "pred_diff_conditions": "없음",
                      "status": "ok", "error": "", "raw_response": "raw"}
            if row["case_id"] == "C2":
                result.update(pred_label="", pred_reason="", pred_diff_conditions="", status="error", error="invalid JSON")
            return result

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "pairs.csv", root / "pred.csv"
            write_csv(source, [pair("C1"), pair("C2")])
            fake = SimpleNamespace(metadata={"model_id": "test-double"}, predict_pair=predict)
            with patch.object(inference, "RequirementClassifier", return_value=fake), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = inference.main(["--pairs", str(source), "--out", str(output)])
            self.assertEqual(status, 1)
            report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual((report["total"], report["succeeded"], report["failed"]), (2, 1, 1))
            self.assertFalse(report["ready_for_evaluation"])
            with output.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([r["case_id"] for r in records], ["C1", "C2"])
            self.assertEqual(records[1]["pred_label"], "")

    def test_long_input_fails_without_truncating_or_calling_generate(self):
        classifier = inference.RequirementClassifier.__new__(inference.RequirementClassifier)
        classifier.config = inference.InferenceConfig(max_input_tokens=100)
        classifier.tokenizer = Mock(return_value={"input_ids": SimpleNamespace(shape=(1, 101))})
        classifier.tokenizer.apply_chat_template.return_value = "full prompt"
        classifier.model = Mock()
        classifier.model.config.max_position_embeddings = 32768
        fake_modules = {"torch": SimpleNamespace(), "transformers": SimpleNamespace(set_seed=Mock())}
        with patch.dict(sys.modules, fake_modules):
            result = classifier.predict_pair(pair())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["pred_label"], "")
        self.assertEqual(result["input_tokens"], 101)
        self.assertIn("너무 깁니다", result["error"])
        classifier.model.generate.assert_not_called()
        self.assertFalse(classifier.tokenizer.call_args.kwargs["truncation"])


if __name__ == "__main__":
    unittest.main()
