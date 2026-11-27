import copy
import itertools
import unittest

from src.grouping import build_groups, validate_requirements


def req(req_id, text=None):
    return {"req_id": req_id, "req_text": text or f"● {req_id}의 요구사항", "source_location": "p.3",
            "raw_req_text": "보존할 원문", "warnings": ["원문 확인"]}


def pair(a, b, label="유사·관련", status="ok"):
    return {"req_a_id": a, "req_b_id": b, "pred_label": label, "status": status}


def ids(groups):
    return [[row["req_id"] for row in group] for group in groups]


class GroupingTests(unittest.TestCase):
    def test_empty_singleton_and_all_unrelated_survive(self):
        self.assertEqual(build_groups([], []), [])
        self.assertEqual(ids(build_groups([req("A")], [])), [["A"]])
        sources = [req(key) for key in "ABC"]
        unrelated = [pair(a, b, "무관") for a, b in itertools.combinations("ABC", 2)]
        self.assertEqual(ids(build_groups(sources, unrelated)), [["A"], ["B"], ["C"]])

    def test_transitive_chain_does_not_collapse(self):
        sources = [req(key) for key in "ABCD"]
        predictions = [pair("A", "B"), pair("B", "C"), pair("A", "C", "무관")]
        self.assertEqual(ids(build_groups(sources, predictions)), [["A", "B"], ["C"], ["D"]])

    def test_duplicate_priority_and_order_independence(self):
        sources = [req(key) for key in "ABC"]
        predictions = [pair("A", "B"), pair("B", "C", "중복"), pair("A", "C", "무관")]
        for ordering in itertools.permutations(predictions):
            with self.subTest(ordering=ordering):
                self.assertEqual(ids(build_groups(sources[::-1], list(ordering))), [["A"], ["B", "C"]])

    def test_fully_related_group_is_allowed(self):
        sources = [req(key) for key in "ABCD"]
        predictions = [pair(a, b) for a, b in itertools.combinations("ABC", 2)]
        self.assertEqual(ids(build_groups(sources, predictions)), [["A", "B", "C"], ["D"]])

    def test_missing_or_failed_comparison_blocks_merge(self):
        sources = [req(key) for key in "ABC"]
        for incomplete in ([], [pair("A", "C", "中복", "error")]):
            predictions = [pair("A", "B"), pair("B", "C"), *incomplete]
            self.assertEqual(ids(build_groups(sources, predictions)), [["A", "B"], ["C"]])
        self.assertEqual(ids(build_groups(sources, [pair("A", "B", "중복", "error")])),
                         [["A"], ["B"], ["C"]])

    def test_invalid_or_duplicate_predictions_rejected(self):
        sources = [req("A"), req("B")]
        cases = [[pair("A", "X")], [pair("A", "A")],
                 [pair("A", "B"), pair("B", "A", "무관")],
                 [pair("A", "B", "유사")], [pair("A", "B", [], "ok")],
                 [pair("A", "B", status="unknown")],
                 [{**pair("A", "B"), "req_a_text": "다른 문서의 본문"}]]
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                build_groups(sources, predictions)

    def test_source_metadata_and_input_objects_preserved(self):
        sources = [req("A"), req("B")]
        original = copy.deepcopy(sources)
        result = build_groups(sources, [pair("A", "B")])
        self.assertEqual(sources, original)
        self.assertEqual(result[0], original)

    def test_invalid_sources_rejected(self):
        for sources in ([req("A"), req(" A ")], [req("A", "#NAME?")],
                        [{"req_id": "A", "req_text": "  "}],
                        [{**req("A"), "source_location": [3]}], [None]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                validate_requirements(sources)


if __name__ == "__main__":
    unittest.main()
