"""Batch inference contracts without torch, GPU, model weights, or network access."""

import json
import sys
import threading
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.inference import InferenceConfig, RequirementClassifier
from src.summarization import summarize_group


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Matrix:
    """Only the indexing used to separate prompt and completion tokens."""

    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]) if values else 0)

    def __getitem__(self, key):
        row, column = key
        if isinstance(row, slice):
            return Matrix([value[column] for value in self.values[row]])
        return Vector(self.values[row][column])

    def tolist(self):
        return self.values


class BatchInputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 99
    eos_token = "<eos>"

    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.calls = []
        self.summary_length = 4
        self.responses = {
            11: json.dumps({"label": "중복", "reason": "같은 의무", "diff_conditions": "없음"}, ensure_ascii=False),
            12: "invalid JSON",
            13: json.dumps({"label": "유사·관련", "reason": "보관 기간이 다름", "diff_conditions": "1년 / 3년"}, ensure_ascii=False),
            14: '{"title_unit_id":"U001","summary_unit_ids":["U001"]}',
            42: "",  # Additional generated tokens with no displayed text.
            88: "UNEXPECTED TRAILING TEXT",
        }

    @property
    def pad_token_id(self):
        return None if self.pad_token is None else self.eos_token_id

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def encode(self, prompt, **kwargs):
        payload = json.loads(prompt)
        if "requirement_a" in payload:
            text = payload["requirement_a"]["text"]
            count = {"short": 4, "medium": 7, "long": 101}[text]
        else:
            count = self.summary_length
        return [5] * count

    def __call__(self, prompt, **kwargs):
        self.calls.append({**kwargs, "padding_side": self.padding_side, "prompt": prompt})
        prompts = prompt if isinstance(prompt, list) else [prompt]
        values = [self.encode(item) for item in prompts]
        width = max(map(len, values))
        padded = [[self.pad_token_id] * (width - len(row)) + row for row in values]
        mask = [[0] * (width - len(row)) + [1] * len(row) for row in values]
        return BatchInputs(input_ids=Matrix(padded), attention_mask=Matrix(mask))

    def decode(self, tokens, **kwargs):
        return "".join(self.responses.get(token, "") for token in tokens)


class Model:
    def __init__(self, completions=None):
        self.config = SimpleNamespace(max_position_embeddings=512)
        self.generation_config = SimpleNamespace(eos_token_id=[99, 100])
        self.device = "cpu"
        self.completions = completions or []
        self.calls = []
        self.error = None
        self.adapter_disabled = False

    def generate(self, **kwargs):
        self.calls.append({**kwargs, "adapter_disabled": self.adapter_disabled})
        if self.error:
            raise self.error
        prefixes = kwargs["input_ids"].values
        if len(prefixes) != len(self.completions):
            raise AssertionError("Wrong number of valid inputs reached the model")
        return Matrix([prefix + completion for prefix, completion in zip(prefixes, self.completions, strict=True)])

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False


def pair(case_id, text):
    return {"case_id": case_id, "req_a_id": case_id + "A", "req_a_text": text,
            "req_b_id": case_id + "B", "req_b_text": "● 보관 기간 3년",
            "req_a_source_location": "p.1", "req_b_source_location": "p.2"}


def classifier(completions=None):
    result = RequirementClassifier.__new__(RequirementClassifier)
    result.config = InferenceConfig(device="cpu", max_input_tokens=10, max_new_tokens=4)
    result._generation_lock = threading.RLock()
    result.tokenizer = Tokenizer()
    result.model = Model(completions)
    return result


class BatchInferenceTests(unittest.TestCase):
    def setUp(self):
        # Tests must also run when the optional torch dependency is not installed.
        self.torch_patch = patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def test_mixed_long_inputs_keep_row_alignment_and_left_padding(self):
        model = classifier([[11, 99, 88, 88], [13, 100, 99, 99]])
        rows = [pair("C1", "short"), pair("C2", "long"), pair("C3", "medium")]
        result = model.predict_pairs(rows, batch_size=3)
        self.assertEqual([row["case_id"] for row in result], ["C1", "C2", "C3"])
        self.assertEqual([row["status"] for row in result], ["ok", "error", "ok"])
        self.assertEqual([row["input_tokens"] for row in result], [4, 101, 7])
        self.assertEqual(result[0]["pred_label"], "중복")
        self.assertEqual(result[2]["pred_label"], "유사·관련")
        self.assertEqual(result[0]["output_tokens"], 2)
        self.assertEqual(result[2]["output_tokens"], 2)
        self.assertNotIn("TRAILING", result[0]["raw_response"])
        self.assertIn("너무 깁니다", result[1]["error"])
        for expected, actual in zip(rows, result, strict=True):
            self.assertEqual(actual["req_a_text"], expected["req_a_text"])
            self.assertEqual(actual["req_b_source_location"], "p.2")
        call = model.model.calls[0]
        self.assertEqual(call["attention_mask"].values, [[0, 0, 0, 1, 1, 1, 1], [1] * 7])
        self.assertFalse(call["do_sample"])
        self.assertEqual(model.tokenizer.calls[0]["padding_side"], "left")
        self.assertFalse(model.tokenizer.calls[0]["truncation"])
        self.assertEqual(model.tokenizer.padding_side, "right")
        self.assertIsNone(model.tokenizer.pad_token)

    def test_eos_at_limit_succeeds_but_missing_eos_and_invalid_json_fail(self):
        model = classifier([[13, 42, 42, 99], [11, 42, 42, 42], [12, 99, 99, 99]])
        result = model.predict_pairs([pair("C1", "short"), pair("C2", "short"), pair("C3", "short")], batch_size=3)
        self.assertEqual([row["status"] for row in result], ["ok", "error", "error"])
        self.assertEqual(result[0]["output_tokens"], 4)
        self.assertIn("토큰 상한", result[1]["error"])
        self.assertEqual(result[1]["pred_label"], "")
        self.assertIn("JSON", result[2]["error"])
        self.assertEqual(result[2]["pred_label"], "")

    def test_batch_runtime_error_does_not_relabel_or_replace_long_input_error(self):
        model = classifier()
        model.model.error = RuntimeError("test allocation failure")
        result = model.predict_pairs([pair("C1", "short"), pair("C2", "long"), pair("C3", "medium")], batch_size=3)
        self.assertTrue(all(row["status"] == "error" and row["pred_label"] == "" for row in result))
        self.assertIn("test allocation failure", result[0]["error"])
        self.assertIn("너무 깁니다", result[1]["error"])
        self.assertNotIn("allocation", result[1]["error"])
        self.assertIn("test allocation failure", result[2]["error"])
        self.assertEqual(model.tokenizer.padding_side, "right")
        self.assertIsNone(model.tokenizer.pad_token)

    def test_all_long_inputs_and_context_overflow_never_call_generation(self):
        model = classifier()
        result = model.predict_pairs([pair("C1", "long"), pair("C2", "long")])
        self.assertTrue(all(row["status"] == "error" for row in result))
        self.assertFalse(model.model.calls)
        self.assertFalse(model.tokenizer.calls)
        model.model.config.max_position_embeddings = 7
        result = model.predict_pairs([pair("C3", "short")])
        self.assertEqual(result[0]["status"], "error")
        self.assertFalse(model.model.calls)

    def test_summary_overflow_falls_back_without_truncating_source(self):
        model = classifier()
        model.tokenizer.summary_length = 101
        requirements = [{"req_id": "R1", "req_text": "● 접속 이력은 1년 보관", "source_location": "p.1"},
                        {"req_id": "R2", "req_text": "● 접속 이력은 3년 보관", "source_location": "p.2"}]
        result = summarize_group(requirements, "G1", generate=model.generate_group_summary)
        self.assertEqual(result["summary_status"], "fallback")
        self.assertTrue(any("너무 깁니다" in warning for warning in result["warnings"]))
        shown = result["summary"] + result["diff_conditions"]
        self.assertIn(requirements[0]["req_text"], shown)
        self.assertIn(requirements[1]["req_text"], shown)
        self.assertEqual(result["source_unit_count"], result["retained_source_unit_count"])
        self.assertFalse(model.tokenizer.calls[0]["truncation"])
        self.assertFalse(model.model.calls)

    def test_summary_disables_adapter_and_restores_it_after_failure(self):
        model = classifier()
        model.config = InferenceConfig(device="cpu", adapter=Path("test-adapter"), max_input_tokens=10, max_new_tokens=4)
        model.model.error = RuntimeError("summary failure")
        requirements = [{"req_id": "R1", "req_text": "● 매월 보고서 제출", "source_location": "p.1"}]
        result = summarize_group(requirements, "G1", generate=model.generate_group_summary)
        self.assertEqual(result["summary_status"], "fallback")
        self.assertTrue(model.model.calls[0]["adapter_disabled"])
        self.assertFalse(model.model.adapter_disabled)
        self.assertFalse(model.model.calls[0]["do_sample"])


if __name__ == "__main__":
    unittest.main()
