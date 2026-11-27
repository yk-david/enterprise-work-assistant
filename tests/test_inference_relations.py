"""Relation-only decoding contracts without GPU, model files or optional packages."""

import csv
import json
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.inference import (InferenceConfig, RequirementClassifier, SYSTEM_PROMPT,
                           parse_relation_prefix, save_predictions)


class Vector:
    device = "cpu"

    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def tolist(self):
        return self.values

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        return Vector(value) if isinstance(index, slice) else value

    def __or__(self, other):
        return Vector(a or b for a, b in zip(self.values, other.values, strict=True))


class Matrix:
    device = "cpu"

    def __init__(self, values):
        self.values = [list(row) for row in values]
        self.shape = (len(values), len(values[0]) if values else 0)

    def __iter__(self):
        return iter(Vector(row) for row in self.values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return Vector(self.values[key])
        row, col = key
        if isinstance(row, slice):
            return Matrix([value[col] for value in self.values[row]])
        return Vector(self.values[row][col])

    def tolist(self):
        return self.values


class Inputs(dict):
    def to(self, _):
        return self


class CriteriaList(list):
    def __call__(self, ids, scores, **kwargs):
        done = Vector([False] * ids.shape[0])
        for criteria in self:
            done = done | criteria(ids, scores, **kwargs)
        return done


class Tokenizer:
    eos_token_id = 99
    eos_token = "<eos>"

    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.calls, self.messages = [], []
        self.fragments = {
            11: '{"label":"중', 12: '복"', 13: ',',
            21: '{"label":"유사', 22: '·관련', 23: '"',
            31: '{"label":"무관",',
            41: '{"reason":"먼저 설명', 42: '', 43: '"}',
            51: '{"label":"중복",', 88: 'UNWANTED',
        }

    @property
    def pad_token_id(self):
        return self.eos_token_id if self.pad_token is not None else None

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return messages[-1]["content"]

    def encode(self, prompt, **kwargs):
        text = json.loads(prompt)["requirement_a"]["text"]
        return [5] * {"short": 4, "medium": 7, "long": 101}[text]

    def __call__(self, prompts, **kwargs):
        self.calls.append({**kwargs, "padding_side": self.padding_side})
        if isinstance(prompts, str):
            prompts = [prompts]
        values = [self.encode(prompt) for prompt in prompts]
        width = max(map(len, values))
        return Inputs(
            input_ids=Matrix([[self.pad_token_id] * (width - len(row)) + row for row in values]),
            attention_mask=Matrix([[0] * (width - len(row)) + [1] * len(row) for row in values]),
        )

    def decode(self, tokens, **kwargs):
        return "".join(self.fragments.get(token, "") for token in tokens)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]


class Model:
    """Generate incrementally and invoke supplied per-row stopping criteria."""
    device = "cpu"

    def __init__(self, completions):
        self.completions = completions
        self.config = SimpleNamespace(max_position_embeddings=512)
        self.generation_config = SimpleNamespace(eos_token_id=[99, 100])
        self.calls = []
        self.error = None

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        outputs = [list(row) for row in kwargs["input_ids"].values]
        if len(outputs) != len(self.completions):
            raise AssertionError("Invalid inputs reached the decoder")
        finished = [False] * len(outputs)
        for step in range(kwargs["max_new_tokens"]):
            for index, output in enumerate(outputs):
                tokens = self.completions[index]
                token = 99 if finished[index] or step >= len(tokens) else tokens[step]
                output.append(token)
                finished[index] |= token in {99, 100}
            criteria = kwargs.get("stopping_criteria", CriteriaList())
            flags = criteria(Matrix(outputs), None).tolist()
            finished = [a or b for a, b in zip(finished, flags, strict=True)]
            if all(finished):
                break
        self.generated = outputs
        return Matrix(outputs)


def pair(case_id, text="short"):
    return {"case_id": case_id, "req_a_id": case_id + "A", "req_a_text": text,
            "req_b_id": case_id + "B", "req_b_text": "● 이력은 3년 보존",
            "req_a_source_location": "p.1", "req_b_source_location": "p.2"}


def classifier(completions):
    value = RequirementClassifier.__new__(RequirementClassifier)
    value.config = InferenceConfig(device="cpu", max_input_tokens=10, max_new_tokens=64)
    value._generation_lock = threading.RLock()
    value.tokenizer = Tokenizer()
    value.model = Model(completions)
    return value


def full_prediction(row, label="무관", status="ok"):
    return {**row, "pred_label": label if status == "ok" else "",
            "pred_reason": "전체 응답의 근거" if status == "ok" else "",
            "pred_diff_conditions": "없음" if status == "ok" else "",
            "status": status, "error": "" if status == "ok" else "invalid JSON"}


class RelationInferenceTests(unittest.TestCase):
    def setUp(self):
        fake_torch = SimpleNamespace(
            inference_mode=nullcontext, bool=bool,
            tensor=lambda values, **kwargs: Vector(values),
            full=lambda size, value, **kwargs: Vector([value] * size[0]),
            zeros=lambda size, **kwargs: Vector([False] * (size if isinstance(size, int) else size[0])),
        )
        self.modules = patch.dict(sys.modules, {
            "torch": fake_torch,
            "transformers": SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=CriteriaList),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_prefix_requires_first_complete_known_label_and_comma(self):
        cases = [('{"label":"중복",', "중복"),
                 (' \n{ "label" : "유사·관련" , "reason":', "유사·관련"),
                 ('```json\n{"label":"무관",', "무관"),
                 ('```\n{"label":"중복",', "중복"),
                 (r'{"label":"\uc911\ubcf5",', "중복")]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_relation_prefix(text), expected)
        invalid = ['{"label":"중복"', '{"label":"중복"}', '{"label":"중복성",',
                   '중복', 'text {"label":"중복",', '{"reason":"중복","label":"중복",',
                   '{"label":123,', '{"label":null,', '{"label":"중복\\",',
                   '{"label":"중복" garbage,', '{"label":"\\uZZZZ",']
        for text in invalid:
            with self.subTest(text=text):
                self.assertIsNone(parse_relation_prefix(text))

    def test_different_completion_lengths_trim_padding_and_preserve_sources(self):
        model = classifier([[11, 12, 13, 88], [21, 22, 23, 13, 88]])
        rows = [pair("C1"), pair("C2", "medium")]
        result = model.predict_relations(rows, batch_size=2)
        self.assertEqual([r["case_id"] for r in result], ["C1", "C2"])
        self.assertEqual([r["pred_label"] for r in result], ["중복", "유사·관련"])
        self.assertEqual([r["output_tokens"] for r in result], [3, 4])
        self.assertTrue(all(r["status"] == "ok" for r in result))
        self.assertTrue(all(r["explanation_status"] == "not_requested" for r in result))
        self.assertTrue(all(r["response_format"] == "label_prefix" for r in result))
        self.assertTrue(all(r["pred_reason"] == r["pred_diff_conditions"] == "" for r in result))
        self.assertNotIn("UNWANTED", str(result))
        self.assertEqual(result[1]["req_b_source_location"], "p.2")
        call = model.model.calls[0]
        self.assertFalse(call["do_sample"])
        self.assertEqual(call["attention_mask"].values, [[0, 0, 0, 1, 1, 1, 1], [1] * 7])
        self.assertEqual(model.tokenizer.messages[0][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(model.tokenizer.padding_side, "right")
        self.assertIsNone(model.tokenizer.pad_token)
        self.assertFalse(model.tokenizer.calls[0]["truncation"])

    def test_prefix_completed_at_budget_is_success_without_eos(self):
        model = classifier([[42] * 31 + [51]])
        model._predict_batch = Mock(side_effect=AssertionError("Unexpected fallback"))
        result = model.predict_relations([pair("C1")])
        self.assertEqual(result[0]["status"], "ok")
        self.assertEqual(result[0]["pred_label"], "중복")
        self.assertEqual(result[0]["output_tokens"], 32)
        model._predict_batch.assert_not_called()

    def test_saved_predictions_explain_why_pair_explanation_fields_are_empty(self):
        result = classifier([[11, 12, 13]]).predict_relations([pair("C1")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "relations.csv"
            save_predictions(result, {"mode": "relations"}, output)
            with output.open(encoding="utf-8-sig", newline="") as stream:
                row = list(csv.DictReader(stream))[0]
            self.assertEqual(row["explanation_status"], "not_requested")
            self.assertEqual(row["response_format"], "label_prefix")
            self.assertEqual(row["pred_reason"], "")
            self.assertEqual(row["pred_diff_conditions"], "")
            report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(report["predictions"][0]["raw_response"], '{"label":"중복",')

    def test_fallback_keeps_original_valid_batch_but_replaces_only_failed_prefix(self):
        model = classifier([[11, 12, 13], [41, 43, 99]])
        rows = [pair("C1"), pair("C2", "long"), pair("C3", "medium")]
        model._predict_batch = Mock(side_effect=lambda batch: [full_prediction(r) for r in batch])
        result = model.predict_relations(rows, batch_size=3)
        self.assertEqual([r["case_id"] for r in model._predict_batch.call_args.args[0]], ["C1", "C3"])
        self.assertEqual(result[0]["pred_label"], "중복")
        self.assertEqual(result[0]["explanation_status"], "not_requested")
        self.assertEqual(result[1]["status"], "error")
        self.assertIn("너무 깁니다", result[1]["error"])
        self.assertEqual(result[2]["pred_label"], "무관")
        self.assertTrue(result[2]["relation_fallback"])
        self.assertEqual(result[2]["explanation_status"], "generated")
        self.assertEqual(result[2]["response_format"], "full_json")

    def test_invalid_full_fallback_remains_error(self):
        model = classifier([[41, 43, 99]])
        model._predict_batch = Mock(side_effect=lambda batch: [full_prediction(r, status="error") for r in batch])
        result = model.predict_relations([pair("C1")])
        self.assertEqual(result[0]["status"], "error")
        self.assertEqual(result[0]["pred_label"], "")
        self.assertEqual(result[0]["explanation_status"], "unavailable")
        self.assertTrue(result[0]["relation_fallback"])

    def test_runtime_error_does_not_guess_or_retry(self):
        model = classifier([])
        model.model.error = RuntimeError("allocation failure")
        model._predict_batch = Mock(side_effect=AssertionError("Runtime error must not retry"))
        result = model.predict_relations([pair("C1"), pair("C2", "long")])
        self.assertTrue(all(r["status"] == "error" and r["pred_label"] == "" for r in result))
        self.assertIn("allocation failure", result[0]["error"])
        self.assertIn("너무 깁니다", result[1]["error"])
        self.assertEqual(model.tokenizer.padding_side, "right")
        self.assertIsNone(model.tokenizer.pad_token)
        model._predict_batch.assert_not_called()

    def test_empty_long_and_context_overflow_do_not_generate(self):
        model = classifier([])
        self.assertEqual(model.predict_relations([]), [])
        result = model.predict_relations([pair("C1", "long")])
        self.assertEqual(result[0]["status"], "error")
        model.model.config.max_position_embeddings = 50  # Full output reservation is 64.
        result = model.predict_relations([pair("C2")])
        self.assertEqual(result[0]["status"], "error")
        self.assertFalse(model.model.calls)
        self.assertFalse(model.tokenizer.calls)


if __name__ == "__main__":
    unittest.main()
