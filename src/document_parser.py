"""PDF/HWP/HWPX 요구사항 표 추출. 원문을 보존하고 정리본과 검토 경고를 제공합니다."""

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


ID_LABELS = {"요구사항번호", "요구사항고유번호"}
DETAIL_LABELS = {"요구사항상세내용", "상세내용", "요구사항세부내용", "세부내용"}
FOOTER_LABELS = {"산출정보", "관련요구사항", "요구사항출처"}
CLASS_LABEL = "요구사항분류"

# PDF/HWP/HWPX에서 공통으로 쓰이는 명시적인 글머리 기호입니다.
# 'ㅇ', 'o', '-'는 글자/하위 항목/수식일 수 있어 기본 치환 대상에서 제외합니다.
STANDARD_BULLET_CHARS = "•●▪‣⁃◦○❍"

# 원문과 대조해 글머리표임을 확인한 문자만 추가하세요.
# 예: {"\uf06c"}. 사설 영역 문자 전체나 대체 문자(�)를 일괄 삭제하지 않습니다.
EXTRA_BULLET_CHARS: set[str] = set()


@dataclass
class TableBlock:
    location: str
    rows: list[list[str | None]]
    hwp_table: Any = None
    first_page: int = 0
    last_page: int = 0
    warnings: list[str] = field(default_factory=list)


def compact(text: str | None) -> str:
    return "".join((text or "").split())


def row_label(row: list[str | None]) -> str:
    return next((compact(cell) for cell in row if cell and cell.strip()), "")


def validate_symbol_map(symbol_map: Any) -> dict[str, str]:
    if not isinstance(symbol_map, dict) or any(
        not isinstance(key, str) or len(key) != 1
        or not isinstance(value, str) or not value
        for key, value in symbol_map.items()
    ):
        raise ValueError("기호 치환표는 한 글자 키와 비어 있지 않은 문자열 값의 JSON 객체여야 합니다.")
    return symbol_map


def validate_bullet_chars(bullet_chars: str) -> str:
    if not isinstance(bullet_chars, str) or any(
        c.isspace() or c == "\ufffd" or unicodedata.category(c) in {"Cc", "Cs", "Cf"}
        for c in bullet_chars
    ):
        raise ValueError("글머리 문자에는 공백, 제어문자, 대체 문자(�)를 지정할 수 없습니다.")
    return bullet_chars


def clean_text(
    raw: str, *, symbol_map: dict[str, str] | None = None, bullet_chars: str = ""
) -> tuple[str, list[str]]:
    """확실한 서식만 정리합니다. 의미 불명의 문자는 남기고 경고합니다."""
    # 사설 문자 의미는 문서/폰트에 따라 다르므로 확인한 치환표를 지정했을 때만 적용합니다.
    mapped = raw.translate(str.maketrans(validate_symbol_map(symbol_map))) if symbol_map is not None else raw
    text = unicodedata.normalize("NFC", mapped).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    notes = []
    controls = {c for c in text if unicodedata.category(c) == "Cc" and c not in "\n\t"}
    if controls:
        codes = ", ".join(f"U+{ord(c):04X}" for c in sorted(controls))
        notes.append(f"제어문자를 공백으로 바꿈: {codes}. raw_req_text와 원문을 확인하세요.")
        text = "".join(" " if c in controls else c for c in text)

    # 줄 시작 + 뒤 공백이 있는 글머리표만 ●로 통일합니다. %, ≥, ≤, ① 등은 유지합니다.
    bullets = STANDARD_BULLET_CHARS + "".join(sorted(EXTRA_BULLET_CHARS)) + validate_bullet_chars(bullet_chars)
    text = re.sub(r"^([ \t]*)[" + re.escape(bullets) + r"](?=[ \t]|$)", r"\1●", text, flags=re.M)
    unknown = {c for c in text if c == "\ufffd" or unicodedata.category(c) in {"Co", "Cs", "Cf"}}
    if unknown:
        codes = ", ".join(f"U+{ord(c):04X}" for c in sorted(unknown))
        notes.append(f"원문/폰트 확인이 필요한 문자를 보존함: {codes}")
    if re.search(r"\(cid:\d+\)", text):
        notes.append("PDF 문자 매핑 실패 가능성: (cid:숫자) 표기를 원문과 대조하세요.")
    return text, notes


def source_ids(rows: list[list[str | None]]) -> set[str]:
    found = set()
    for row in rows:
        for col, cell in enumerate(row[:-1]):
            if compact(cell) in ID_LABELS:
                value = next((compact(v) for v in row[col + 1:]
                              if v and v.strip() and compact(v) not in ID_LABELS), "")
                # 요구사항 목록표의 열 제목은 실제 요구사항 번호가 아닙니다.
                if value.startswith("요구사항명"):
                    continue
                if not re.fullmatch(r"[A-Z]+-[0-9]+", value):
                    raise ValueError(f"요구사항 원본 번호 형식을 확인하세요: {value!r}")
                found.add(value)
    return found


def split_sections(rows: list[list[str | None]]):
    """앞쪽의 이어지는 본문을 보존하고, 붙어 있는 새 요구사항 표를 나눕니다."""
    section = []
    has_id = False
    for row in rows:
        contains_id = any(compact(cell) in ID_LABELS for cell in row[:-1])
        # 분류 행 없이 새 번호가 시작해도 앞의 연속 본문을 별도 구간으로 남깁니다.
        new_id = contains_id and (has_id or not any(row_label(r) == CLASS_LABEL for r in section))
        if section and (row_label(row) == CLASS_LABEL or new_id):
            yield section
            section = []
            has_id = False
        section.append(row)
        has_id = has_id or contains_id
    if section:
        yield section


def continuation_rows(rows: list[list[str | None]]) -> list[list[str | None]]:
    # 종료 항목이 없으면 전체를 본문 후보로 사용합니다. rows[:0]으로 버리지 않습니다.
    end = next((i for i, row in enumerate(rows) if row_label(row) in FOOTER_LABELS), len(rows))
    body = []
    for row in rows[:end]:
        cells = [(i, c) for i, c in enumerate(row) if c and c.strip()]
        if not cells:
            continue
        if compact(cells[0][1]) in DETAIL_LABELS:
            values = [c for _, c in cells[1:] if compact(c) not in DETAIL_LABELS]
            if values:
                body.append(["세부내용", "\t".join(values)])
        elif len(cells) == 1 and cells[0][0] == len(row) - 1:
            body.append(["세부내용", cells[0][1]])
        else:
            raise ValueError("이전 요구사항의 연속인지 판단하기 어려운 표입니다. 원문을 확인하세요.")
    return body


def read_pdf(path: Path) -> list[TableBlock]:
    import pdfplumber

    blocks = []
    pending = None
    with pdfplumber.open(path) as doc:
        for number, page in enumerate(doc.pages, start=1):
            try:
                page_tables = page.extract_tables()
            finally:
                page.close()
            for rows in page_tables:
                for section in split_sections(rows):
                    if not any(cell and cell.strip() for row in section for cell in row):
                        continue
                    ids = source_ids(section)
                    has_footer = any(row_label(row) in FOOTER_LABELS for row in section)
                    if ids:
                        # 다음 페이지에 같은 요구사항 번호가 반복된 경우도 연결합니다.
                        repeated_header = (
                            pending is not None
                            and number == pending.last_page + 1
                            and ids == source_ids(pending.rows)
                        )
                        if repeated_header:
                            pending.rows.extend(section)
                        else:
                            pending = TableBlock(f"p.{number}", section, first_page=number, last_page=number)
                            blocks.append(pending)
                    elif pending is not None:
                        if number not in {pending.last_page, pending.last_page + 1}:
                            raise ValueError(f"p.{number}: 이어지는 요구사항 사이에 페이지가 빠졌습니다.")
                        try:
                            body = continuation_rows(section)
                        except ValueError as error:
                            raise ValueError(f"p.{number}: {error}") from error
                        if not body and not has_footer:
                            raise ValueError(f"p.{number}: 이어지는 본문을 확인할 수 없습니다.")
                        pending.rows.extend(body)
                    else:
                        # 일반 표는 뒤에서 제외하고, 번호 없는 상세내용 표는 오류로 알립니다.
                        blocks.append(TableBlock(f"p.{number}", section))
                        continue

                    pending.last_page = number
                    pending.location = (
                        f"p.{pending.first_page}" if pending.first_page == number
                        else f"p.{pending.first_page}–{number}"
                    )
                    if has_footer:
                        pending = None

            if pending is not None and pending.last_page < number:
                raise ValueError(f"p.{number}: 이전 요구사항이 끝나지 않았지만 이어지는 표를 추출하지 못했습니다.")
    if pending is not None:
        pending.warnings.append("문서 끝에서 요구사항 종료 항목을 찾지 못했습니다. 본문 완전성을 확인하세요.")
    return blocks


def read_hwp(path: Path) -> list[TableBlock]:
    from hwpx import HwpxDocument

    doc = HwpxDocument.open(path)
    tables = doc.tables.with_nested
    blocks = []
    for entry in doc.tables.map()["tables"]:
        rows = [[None] * entry["cols"] for _ in range(entry["rows"])]
        for cell in entry["cells"]:
            rows[cell["row"]][cell["col"]] = cell["text"]
        index = entry["table_index"]
        blocks.append(TableBlock(f"표 {index + 1}", rows, tables[index]))
    return blocks


def parse_rfp(
    file_path: str | Path, *, symbol_map: dict[str, str] | None = None, bullet_chars: str = ""
) -> list[dict[str, Any]]:
    validate_bullet_chars(bullet_chars)
    path = Path(file_path)
    doc_type = path.suffix[1:].upper()
    if doc_type == "PDF":
        blocks = read_pdf(path)
    elif doc_type in {"HWP", "HWPX"}:
        blocks = read_hwp(path)
    else:
        raise ValueError("지원 형식은 .hwp, .hwpx와 .pdf입니다.")

    result = []
    seen = set()
    for block in blocks:
        ids = source_ids(block.rows)
        details = []
        for row_number, row in enumerate(block.rows):
            # 중첩된 상세내용 라벨이 있으면 오른쪽의 마지막 라벨을 사용합니다.
            label_cols = [i for i, cell in enumerate(row[:-1]) if compact(cell) in DETAIL_LABELS]
            if not label_cols:
                continue
            col = label_cols[-1]
            # 병합 셀 표의 '정의' 행을 세부 요구사항 본문으로 오인하지 않습니다.
            if col + 1 < len(row) and compact(row[col + 1]) == "정의":
                continue
            for value_col in range(col + 1, len(row)):
                value = row[value_col]
                if value and value.strip():
                    break
            else:
                continue
            if block.hwp_table is not None:
                cell = block.hwp_table.cell(row_number, value_col)
                if cell.tables:
                    parts = []
                    for paragraph in cell.paragraphs:
                        parts.append(paragraph.text or "")
                        for nested in paragraph.tables:
                            parts.extend("\t".join(c.text or "" for c in r.cells) for r in nested.rows)
                    value = "\n".join(parts)
            details.append(value)

        if not ids and not details:
            continue
        if len(ids) != 1 or not details:
            raise ValueError(f"{block.location}: 요구사항 번호와 상세내용을 짝지을 수 없습니다.")
        source_id = next(iter(ids))
        if source_id in seen:
            raise ValueError(f"{block.location}: 요구사항 {source_id}가 반복됩니다.")
        seen.add(source_id)
        raw = "\n".join(details)
        cleaned, notes = clean_text(raw, symbol_map=symbol_map, bullet_chars=bullet_chars)
        result.append({
            "req_id": f"REQ-{len(result) + 1:03d}",
            "source_req_id": source_id,
            "doc_name": path.name,
            "doc_type": doc_type,
            "source_location": block.location,
            "req_text": cleaned,
            "raw_req_text": raw,
            "warnings": block.warnings + notes,
        })
    if not result:
        raise ValueError("문서에서 요구사항 번호와 상세내용이 있는 표를 찾지 못했습니다.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="기존 JSON도 다시 생성")
    parser.add_argument("--symbol-map", type=Path, help="원문과 대조한 문서용 기호 치환표 JSON")
    parser.add_argument("--bullet-chars", default="", help="이 문서에서 확인한 추가 글머리 문자(예: ㅇ). 줄 시작에서만 ●로 치환")
    parser.add_argument("--file", type=Path, help="input 폴더 전체 대신 지정한 PDF/HWP/HWPX 한 파일만 처리")
    args = parser.parse_args()
    if args.bullet_chars and args.file is None:
        parser.error("문서별 글머리 문자는 --file로 지정한 한 파일에만 적용하세요.")
    try:
        validate_bullet_chars(args.bullet_chars)
    except ValueError as error:
        parser.error(str(error))
    symbol_map = None
    if args.symbol_map is not None:
        if args.file is None:
            parser.error("문서별 기호 치환표는 --file로 지정한 한 파일에만 적용하세요.")
        try:
            symbol_map = validate_symbol_map(json.loads(args.symbol_map.read_text(encoding="utf-8-sig")))
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(f"기호 치환표를 읽을 수 없습니다: {error}")
    data_dir = Path(__file__).resolve().parent / "data"
    input_dir, output_dir = data_dir / "input", data_dir / "output"
    if args.file is not None:
        if not args.file.is_file() or args.file.suffix.lower() not in {".hwp", ".hwpx", ".pdf"}:
            parser.error("--file에는 존재하는 PDF/HWP/HWPX 파일을 지정하세요.")
        sources = [args.file]
    elif not input_dir.is_dir():
        parser.exit(1, f"입력 폴더를 만들고 PDF/HWP/HWPX를 넣어주세요: {input_dir}\n")
    else:
        sources = sorted(input_dir.iterdir())
    output_dir.mkdir(parents=True, exist_ok=True)
    created = skipped = failed = 0
    for source in sources:
        suffix = source.suffix.lower()
        if not source.is_file() or suffix not in {".hwp", ".hwpx", ".pdf"}:
            continue
        name = source.stem if suffix == ".hwp" else source.name
        output_path = output_dir / f"{name}.requirements.json"
        if output_path.is_file() and not args.overwrite:
            print(f"건너뜀 {source.name} (--overwrite로 다시 생성 가능)")
            skipped += 1
            continue
        try:
            result = parse_rfp(source, symbol_map=symbol_map, bullet_chars=args.bullet_chars)
            with TemporaryDirectory(dir=output_dir) as temp_dir:
                temp_path = Path(temp_dir) / output_path.name
                with temp_path.open("w", encoding="utf-8") as output_file:
                    json.dump(result, output_file, ensure_ascii=False, indent=2)
                    output_file.write("\n")
                temp_path.replace(output_path)
        except Exception as error:
            print(f"실패 {source.name} → {error}", file=sys.stderr)
            failed += 1
            continue
        created += 1
        print(f"저장 {source.name} → {output_path.name} ({len(result)}건)")
        for item in result:
            for note in item["warnings"]:
                print(f"검토 {source.name} / {item['source_req_id']}: {note}", file=sys.stderr)
    print(f"파일 기준 · 생성 {created}개 · 건너뜀 {skipped}개 · 실패 {failed}개")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
