import copy
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "document_parser", Path(__file__).resolve().parents[1] / "src" / "document_parser.py"
)
parser = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parser
SPEC.loader.exec_module(parser)


def requirement(source_id="SFR-001", text="FIRST", footer=False):
    rows = [["요구사항분류", "기능"], ["요구사항번호", source_id], ["세부내용", text]]
    return rows + ([["산출정보", "report"]] if footer else [])


class FakePage:
    def __init__(self, tables):
        self.tables = tables

    def extract_tables(self):
        return self.tables

    def close(self):
        pass


class FakePDF:
    def __init__(self, pages):
        self.pages = [FakePage(tables) for tables in copy.deepcopy(pages)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def parse_pages(pages):
    module = types.SimpleNamespace(open=lambda path: FakePDF(pages))
    with patch.dict(sys.modules, {"pdfplumber": module}):
        return parser.parse_rfp("sample.pdf")


class ParserTests(unittest.TestCase):
    def test_two_pages_with_footer(self):
        items = parse_pages([[requirement()], [[[None, "SECOND"], ["산출정보", "report"]]]])
        self.assertEqual(items[0]["req_text"], "FIRST\nSECOND")
        self.assertEqual(items[0]["source_location"], "p.1–2")

    def test_missing_footer_does_not_drop_second_page(self):
        items = parse_pages([[requirement()], [[[None, "SECOND"]]]])
        self.assertEqual(items[0]["req_text"], "FIRST\nSECOND")
        self.assertTrue(items[0]["warnings"])

    def test_three_pages(self):
        items = parse_pages([
            [requirement()], [[[None, "SECOND"]]],
            [[[None, "THIRD"], ["관련요구사항", "SFR-002"]]],
        ])
        self.assertEqual(items[0]["req_text"], "FIRST\nSECOND\nTHIRD")
        self.assertEqual(items[0]["source_location"], "p.1–3")
        self.assertFalse(items[0]["warnings"])

    def test_continuation_before_new_requirement_in_same_table(self):
        next_table = [[None, "SECOND"], ["산출정보", "report"]] + requirement("SFR-002", "NEW", True)
        items = parse_pages([[requirement()], [next_table]])
        self.assertEqual([r["req_text"] for r in items], ["FIRST\nSECOND", "NEW"])
        self.assertEqual([r["source_req_id"] for r in items], ["SFR-001", "SFR-002"])

    def test_new_requirement_without_classification_header(self):
        rows = requirement(footer=True) + requirement("SFR-002", "NEW", True)[1:]
        self.assertEqual(len(parse_pages([[rows]])), 2)

    def test_continuation_before_new_id_without_classification(self):
        rows = [[None, "SECOND"], ["산출정보", "report"]] + requirement("SFR-002", "NEW", True)[1:]
        items = parse_pages([[requirement()], [rows]])
        self.assertEqual([r["req_text"] for r in items], ["FIRST\nSECOND", "NEW"])

    def test_repeated_headers_across_three_pages(self):
        items = parse_pages([[requirement()], [requirement(text="SECOND")], [requirement(text="THIRD", footer=True)]])
        self.assertEqual(items[0]["req_text"], "FIRST\nSECOND\nTHIRD")
        self.assertEqual(items[0]["source_location"], "p.1–3")

    def test_labeled_continuation_and_footer_only_page(self):
        items = parse_pages([[requirement()], [[["상세내용", "SECOND"]]], [[["산출정보", "report"]]]])
        self.assertEqual(items[0]["req_text"], "FIRST\nSECOND")
        self.assertEqual(items[0]["source_location"], "p.1–3")

    def test_ambiguous_table_is_not_silently_merged(self):
        with self.assertRaisesRegex(ValueError, "판단하기 어려운"):
            parse_pages([[requirement()], [[["unrelated", "value"]]]])

    def test_missing_page_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "추출하지 못했습니다"):
            parse_pages([[requirement()], [], [[[None, "THIRD"]]]])

    def test_new_id_closes_previous_requirement_without_footer(self):
        items = parse_pages([[requirement()], [requirement("SFR-002", "NEW", True)]])
        self.assertEqual([r["req_text"] for r in items], ["FIRST", "NEW"])

    def test_duplicate_complete_requirements_raise(self):
        with self.assertRaisesRegex(ValueError, "반복됩니다"):
            parse_pages([[requirement(footer=True)], [requirement(footer=True)]])

    def test_text_cleanup_preserves_conditions_and_raw_text(self):
        raw = "● 조건 ≥ 95%, ≤ 100%, ① 필수\r\n• 가\u00a0나\n\uf06c 항목\n\ufffd 항목\x00끝"
        item = parse_pages([[requirement(text=raw, footer=True)]])[0]
        self.assertEqual(item["raw_req_text"], raw)
        self.assertIn("● 조건 ≥ 95%, ≤ 100%, ① 필수\n● 가 나", item["req_text"])
        self.assertIn("\uf06c", item["req_text"])
        self.assertIn("\ufffd", item["req_text"])
        self.assertNotIn("\x00", item["req_text"])
        self.assertEqual(len(item["warnings"]), 2)
        self.assertEqual(json.loads(json.dumps(item, ensure_ascii=False).encode("utf-8")), item)

    def test_only_confirmed_extra_bullets_are_changed(self):
        with patch.object(parser, "EXTRA_BULLET_CHARS", {"\uf06c"}):
            text, notes = parser.clean_text("\uf06c 항목\n문장 중간 \uf06c 문자")
        self.assertEqual(text, "● 항목\n문장 중간 \uf06c 문자")
        self.assertTrue(notes)

    def test_cid_is_flagged_and_not_deleted(self):
        text, notes = parser.clean_text("(cid:42) 항목")
        self.assertEqual(text, "(cid:42) 항목")
        self.assertTrue(notes)

    def test_hangul_document_bullets_keep_indentation_and_conditions(self):
        raw = "◦ 첫 항목\n  ❍ 둘째 항목\n\t○ 셋째 항목\n- 하위 조건 ≤ 5%\n① 순서\nㅇ 선택 항목\n본문의 ◦ 기호\n○붙어있는표시"
        text, notes = parser.clean_text(raw)
        self.assertEqual(text, "● 첫 항목\n  ● 둘째 항목\n\t● 셋째 항목\n- 하위 조건 ≤ 5%\n① 순서\nㅇ 선택 항목\n본문의 ◦ 기호\n○붙어있는표시")
        self.assertFalse(notes)

    def test_document_bullet_override_only_changes_line_prefix(self):
        raw = "ㅇ 글머리\n\t\uf0a7 확인한 기호\n본문의 ㅇ 및 \uf0a7 유지\nㅇ붙어있는글자"
        text, notes = parser.clean_text(raw, bullet_chars="ㅇ\uf0a7")
        self.assertEqual(text, "● 글머리\n\t● 확인한 기호\n본문의 ㅇ 및 \uf0a7 유지\nㅇ붙어있는글자")
        self.assertIn("U+F0A7", notes[0])
        # 문서별 설정이 다음 호출에 남지 않아야 합니다.
        self.assertEqual(parser.clean_text(raw)[0], raw)

    def test_invalid_bullet_overrides_are_rejected(self):
        for chars in ["ㅇ ", "\n", "\x00", "\ufffd", "\ud800", None]:
            with self.subTest(chars=chars), self.assertRaises(ValueError):
                parser.clean_text("본문", bullet_chars=chars)

    def test_verified_document_map_distinguishes_bullet_and_middle_dot(self):
        raw = "\uf09f 자료 송\uf09e수신 및 진단\uf09e평가"
        text, notes = parser.clean_text(raw, symbol_map={"\uf09f": "•", "\uf09e": "·"})
        self.assertEqual(text, "● 자료 송·수신 및 진단·평가")
        self.assertFalse(notes)
        self.assertEqual(parser.clean_text(raw)[0], raw)

    def test_symbol_map_preserves_extracted_raw_text(self):
        raw = "\uf09f 조건 ≥ 95% 및 송\uf09e수신"
        block = parser.TableBlock("p.30", requirement(text=raw, footer=True))
        with patch.object(parser, "read_pdf", return_value=[block]):
            item = parser.parse_rfp("sample.pdf", symbol_map={"\uf09f": "•", "\uf09e": "·"})[0]
        self.assertEqual(item["raw_req_text"], raw)
        self.assertEqual(item["req_text"], "● 조건 ≥ 95% 및 송·수신")

    def test_invalid_symbol_maps_are_rejected(self):
        for mapping in [[], {"ab": "x"}, {"x": ""}, {"x": 1}]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                parser.validate_symbol_map(mapping)

    def test_hwp_raw_text_is_also_preserved(self):
        block = parser.TableBlock("표 1", requirement(text="● 항목", footer=True))
        with patch.object(parser, "read_hwp", return_value=[block]):
            item = parser.parse_rfp("sample.hwp")[0]
        self.assertEqual(item["req_text"], "● 항목")
        self.assertEqual(item["raw_req_text"], "● 항목")

    def test_hwp_and_hwpx_share_cleanup_without_altering_raw_text(self):
        raw = "◦ 항목\nㅇ 확인한 글머리\n- 송\uf09e수신 ≥ 95%\n\uf06c 미확인 기호"
        block = parser.TableBlock("표 1", requirement(text=raw, footer=True))
        for suffix in ["hwp", "hwpx"]:
            with self.subTest(suffix=suffix), patch.object(parser, "read_hwp", return_value=[block]):
                item = parser.parse_rfp(f"sample.{suffix}", symbol_map={"\uf09e": "·"}, bullet_chars="ㅇ")[0]
            self.assertEqual(item["doc_type"], suffix.upper())
            self.assertEqual(item["raw_req_text"], raw)
            self.assertEqual(item["req_text"], "● 항목\n● 확인한 글머리\n- 송·수신 ≥ 95%\n\uf06c 미확인 기호")
            self.assertIn("U+F06C", item["warnings"][0])

    def test_cli_symbol_map_only_processes_selected_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "data" / "input"
            input_dir.mkdir(parents=True)
            selected = input_dir / "sample.pdf"
            selected.write_bytes(b"selected fixture")
            (input_dir / "unrelated.hwp").write_bytes(b"unrelated fixture")
            mapping = {"\uf09f": "●", "\uf09e": "·"}
            map_path = root / "symbols.json"
            map_path.write_text(json.dumps(mapping), encoding="utf-8")
            raw = "\uf09f 송\uf09e수신"
            block = parser.TableBlock("p.1", requirement(text=raw, footer=True))
            argv = ["parser", "--file", str(selected), "--symbol-map", str(map_path)]
            with patch.object(parser, "__file__", str(root / "document_parser.py")), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                with patch.object(parser, "read_pdf", return_value=[block]) as read_pdf, patch.object(parser, "read_hwp") as read_hwp:
                    self.assertEqual(parser.main(), 0)
                read_pdf.assert_called_once_with(selected)
                read_hwp.assert_not_called()
            outputs = list((root / "data" / "output").glob("*.json"))
            self.assertEqual(len(outputs), 1)
            item = json.loads(outputs[0].read_text(encoding="utf-8"))[0]
            self.assertEqual(item["req_text"], "● 송·수신")
            self.assertEqual(item["raw_req_text"], raw)

    def test_cli_rejects_map_without_selected_file(self):
        with patch.object(sys, "argv", ["parser", "--symbol-map", "symbols.json"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.main()
        self.assertEqual(error.exception.code, 2)

    def test_cli_rejects_bullet_override_without_selected_file(self):
        with patch.object(sys, "argv", ["parser", "--bullet-chars", "ㅇ"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.main()
        self.assertEqual(error.exception.code, 2)

    def test_cli_rejects_missing_or_unsupported_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            unsupported = root / "document.txt"
            unsupported.write_text("fixture", encoding="utf-8")
            for source in [root / "missing.pdf", unsupported]:
                with self.subTest(source=source), patch.object(sys, "argv", ["parser", "--file", str(source)]), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parser.main()
                    self.assertEqual(error.exception.code, 2)

    def test_cli_overwrite_and_failed_rebuild(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "data" / "input"
            input_dir.mkdir(parents=True)
            (input_dir / "sample.pdf").write_bytes(b"test fixture")
            output = root / "data" / "output" / "sample.pdf.requirements.json"
            item = {"source_req_id": "SFR-001", "req_text": "FIRST", "warnings": []}
            with patch.object(parser, "__file__", str(root / "document_parser.py")), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch.object(sys, "argv", ["parser"]), patch.object(parser, "parse_rfp", return_value=[item]) as mocked:
                    self.assertEqual(parser.main(), 0)
                    self.assertEqual(parser.main(), 0)
                    self.assertEqual(mocked.call_count, 1)
                item["req_text"] = "UPDATED"
                with patch.object(sys, "argv", ["parser", "--overwrite"]), patch.object(parser, "parse_rfp", return_value=[item]):
                    self.assertEqual(parser.main(), 0)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["req_text"], "UPDATED")
                with patch.object(sys, "argv", ["parser", "--overwrite"]), patch.object(parser, "parse_rfp", side_effect=ValueError("bad input")):
                    self.assertEqual(parser.main(), 1)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["req_text"], "UPDATED")


if __name__ == "__main__":
    unittest.main()
