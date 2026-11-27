"""문서 전체 요구사항 비교 → 보수적인 그룹화 → 원문에 근거한 요약.

python -m src.analysis --input requirements.json --out analysis.json --adapter artifacts/run/adapter
"""

import argparse
import copy
import csv
import io
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from src.inference import (InferenceConfig, LABELS, PAIR_FIELDS, RELATION_MAX_NEW_TOKENS, SOURCE_FIELDS,
                           SPREADSHEET_ERRORS, RequirementClassifier, validate_pair)
from src.grouping import build_groups
from src.summarization import summarize_group

GROUP_FIELDS = ("group_id", "group_name", "summary", "req_ids", "diff_conditions", "source_locations")


def validate_requirements(requirements: list[dict]) -> list[dict]:
    if not isinstance(requirements, list):
        raise ValueError("요구사항은 JSON 배열이어야 합니다.")
    rows, seen = [], set()
    for index, original in enumerate(requirements, start=1):
        if not isinstance(original, dict):
            raise ValueError(f"요구사항 {index}는 객체여야 합니다.")
        row = copy.deepcopy(original)
        for field in ("req_id", "req_text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"요구사항 {index}: {field}가 비어 있거나 문자열이 아닙니다.")
        row["req_id"] = row["req_id"].strip()
        if row["req_id"] in seen:
            raise ValueError(f"중복 요구사항 ID: {row['req_id']}")
        seen.add(row["req_id"])
        if row["req_text"].strip().upper() in SPREADSHEET_ERRORS:
            raise ValueError(f"{row['req_id']}: 본문에 스프레드시트 오류가 있습니다. 원문을 복구하세요.")
        for field in SOURCE_FIELDS:
            if not isinstance(row.get(field, ""), str):
                raise ValueError(f"{row['req_id']}: {field}는 문자열이어야 합니다.")
            row.setdefault(field, "")
        rows.append(row)
    documents = {row["doc_name"] for row in rows if row["doc_name"]}
    if len(documents) > 1:
        raise ValueError("한 번에 하나의 문서를 분석하세요. 여러 RFP 간 비교는 지원하지 않습니다.")
    return rows


def make_pairs(requirements: list[dict]) -> list[dict]:
    rows = validate_requirements(requirements)
    result = []
    for index, (a, b) in enumerate(combinations(rows, 2), start=1):
        pair = {"case_id": f"DOC-{index:06d}"}
        for side, row in (("a", a), ("b", b)):
            pair[f"req_{side}_id"] = row["req_id"]
            pair[f"req_{side}_text"] = row["req_text"]
            for key in SOURCE_FIELDS:
                pair[f"req_{side}_{key}"] = row[key]
        result.append(validate_pair(pair))
    return result


def _error_record(pair, error):
    return {**pair, "pred_label": "", "pred_reason": "", "pred_diff_conditions": "",
            "status": "error", "error": str(error)}


def _checked_prediction(pair, prediction, *, allow_label_prefix=False):
    if not isinstance(prediction, dict):
        return _error_record(pair, "분류기가 객체를 반환하지 않았습니다.")
    if any(prediction.get(field) != pair[field] for field in PAIR_FIELDS):
        return _error_record(pair, "분류 결과의 사례 ID 또는 요구사항 본문이 입력과 다릅니다.")
    if prediction.get("status") == "ok":
        explanation_fields = ("pred_reason", "pred_diff_conditions")
        label_only = (allow_label_prefix and prediction.get("explanation_status") == "not_requested"
                      and prediction.get("response_format") == "label_prefix"
                      and all(prediction.get(field) == "" for field in explanation_fields))
        if (not isinstance(prediction.get("pred_label"), str) or prediction["pred_label"] not in LABELS
                or (not label_only and any(not isinstance(prediction.get(field), str) or not prediction[field].strip()
                                           for field in explanation_fields))):
            return _error_record(pair, "분류 결과의 라벨·근거·차이점이 올바르지 않습니다.")
        return {**prediction, **pair}
    return {**prediction, **_error_record(pair, prediction.get("error") or "분류에 실패했습니다.")}


def analyze_requirements(requirements: list[dict], classifier, progress_callback=None,
                         batch_size: int = 4, detailed_pairs: bool = False) -> dict:
    """진행 콜백은 (완료 쌍 수, 전체 쌍 수). 누락/오류는 무관으로 간주하지 않습니다."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size는 1 이상의 정수여야 합니다.")
    if type(detailed_pairs) is not bool:
        raise ValueError("detailed_pairs는 참 또는 거짓이어야 합니다.")
    started = time.perf_counter()
    rows = validate_requirements(requirements)
    pairs = make_pairs(rows)
    records, warnings = [], []
    use_relations = not detailed_pairs and callable(getattr(classifier, "predict_relations", None))
    batch_predictor = (classifier.predict_relations if use_relations else getattr(classifier, "predict_pairs", None))
    method = "predict_relations" if use_relations else "predict_pairs" if callable(batch_predictor) else "predict_pair"
    configured_output_limit = getattr(getattr(classifier, "config", None), "max_new_tokens", None)
    relation_output_limit = (min(RELATION_MAX_NEW_TOKENS, configured_output_limit)
                             if use_relations and type(configured_output_limit) is int else None)
    pair_started = time.perf_counter()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        if callable(batch_predictor):
            try:
                predictions = batch_predictor(batch, batch_size=batch_size)
                if not isinstance(predictions, list) or len(predictions) != len(batch):
                    raise ValueError("배치 분류 결과의 개수가 입력과 다릅니다.")
            except (RuntimeError, ValueError) as error:
                predictions = [_error_record(pair, error) for pair in batch]
        else:
            predictions = []
            for pair in batch:
                try:
                    predictions.append(classifier.predict_pair(pair))
                except (RuntimeError, ValueError) as error:
                    predictions.append(_error_record(pair, error))
        records.extend(_checked_prediction(pair, pred, allow_label_prefix=use_relations)
                       for pair, pred in zip(batch, predictions, strict=True))
        if progress_callback:
            progress_callback(len(records), len(pairs))
    pair_seconds = time.perf_counter() - pair_started
    failed = sum(row["status"] != "ok" for row in records)
    if failed:
        warnings.append(f"{failed}개 쌍의 관계를 판정하지 못했습니다. 해당 관계를 무관으로 처리하거나 그룹 연결에 사용하지 않았습니다.")
    grouping_started = time.perf_counter()
    clusters = build_groups(rows, records)
    grouping_seconds = time.perf_counter() - grouping_started
    summary_generator = getattr(classifier, "generate_group_summary", None)
    groups = []
    summary_started = time.perf_counter()
    for index, members in enumerate(clusters, start=1):
        # 단독 요구사항은 원문 추출 요약으로 충분하며 추가 모델 추론이 필요하지 않습니다.
        group = summarize_group(members, f"GROUP-{index:03d}",
                                generate=summary_generator if len(members) > 1 else None)
        groups.append(group)
        warnings.extend(f"{group['group_id']}: {message}" for message in group.get("warnings", []))
    summary_seconds = time.perf_counter() - summary_started
    actual_ids = [req_id for group in groups for req_id in group["req_ids"]]
    if len(actual_ids) != len(rows) or set(actual_ids) != {row["req_id"] for row in rows}:
        raise ValueError("그룹 결과에 요구사항 누락 또는 중복이 있습니다.")
    return {
        "schema_version": "rfp-document-v1", "requirements": rows,
        "groups": groups, "pair_results": records,
        "status": "partial" if failed else "ok",
        "warnings": warnings,
        "stats": {"requirement_count": len(rows), "pair_count": len(pairs),
                  "successful_pairs": len(records) - failed, "failed_pairs": failed,
                  "group_count": len(groups),
                  "pair_inference_seconds": round(pair_seconds, 3),
                  "grouping_seconds": round(grouping_seconds, 3),
                  "summary_seconds": round(summary_seconds, 3),
                  "elapsed_seconds": round(time.perf_counter() - started, 3)},
        "run": {"model": copy.deepcopy(getattr(classifier, "metadata", {})), "batch_size": batch_size,
                "pair_generation": "all_unordered_pairs",
                "pair_inference_method": method if pairs else "none",
                "detailed_pairs_requested": detailed_pairs,
                "pair_explanations": "not_requested" if use_relations else "generated",
                "pair_explanation_counts": {
                    "not_requested": sum(row.get("explanation_status") == "not_requested" and row["status"] == "ok" for row in records),
                    "generated": sum(row.get("explanation_status") != "not_requested" and row["status"] == "ok" for row in records),
                },
                "document_decoding": "greedy" if callable(batch_predictor) else "classifier_default",
                "document_generation": {
                    "do_sample": False if callable(batch_predictor) else None,
                    "max_input_tokens": getattr(getattr(classifier, "config", None), "max_input_tokens", None),
                    "max_new_tokens": getattr(getattr(classifier, "config", None), "max_new_tokens", None),
                    "relation_max_new_tokens": relation_output_limit,
                    "summary_adapter_enabled": False,
                },
                "grouping": "all_member_relations_required",
                "summary": "source_unit_selection_with_unselected_text_preserved"},
    }


def group_csv_bytes(groups: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=GROUP_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    for group in groups:
        row = {}
        for key in GROUP_FIELDS:
            value = group[key]
            separator = "|" if key == "req_ids" else " | "
            text = separator.join(value) if isinstance(value, list) else str(value)
            # CSV를 Excel로 열 때 원문을 수식으로 실행/변환하지 않도록 표시용 이스케이프.
            # JSON에는 따옴표를 붙이지 않은 원문을 그대로 저장합니다.
            row[key] = "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")


def save_analysis(report: dict, output: Path, input_path: Path | None = None, overwrite=False):
    if output.suffix.lower() != ".json":
        raise ValueError("문서 분석 결과는 .json 경로를 지정하세요. 같은 이름의 CSV도 저장됩니다.")
    targets = (output, output.with_suffix(".csv"))
    for target in targets:
        if input_path and target.resolve() == input_path.resolve():
            raise ValueError("분석 결과로 입력 파일을 덮어쓸 수 없습니다.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"이미 존재하는 결과입니다: {target}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp)
        (staged / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staged / "result.csv").write_bytes(group_csv_bytes(report["groups"]))
        (staged / "result.json").replace(targets[0])
        (staged / "result.csv").replace(targets[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="요구사항 JSON 또는 PDF/HWP/HWPX")
    parser.add_argument("--out", type=Path, required=True, help="결과 JSON (그룹 CSV도 함께 저장)")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--detailed-pairs", action="store_true", help="관계 판정 외에 각 쌍의 상세 근거·차이점도 생성")
    parser.add_argument("--allow-download", action="store_true", help="기본값은 로컬 캐시만 사용")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("batch-size는 1 이상이어야 합니다.")
        # 모델 로딩 전에 출력 충돌을 확인합니다.
        if args.out.suffix.lower() != ".json":
            raise ValueError("--out은 .json 파일이어야 합니다.")
        for path in (args.out, args.out.with_suffix(".csv")):
            if path.resolve() == args.input.resolve() or (path.exists() and not args.overwrite):
                raise ValueError(f"입력 또는 기존 결과를 덮어쓸 수 없습니다: {path}")
        if args.input.suffix.lower() == ".json":
            requirements = json.loads(args.input.read_text(encoding="utf-8-sig"))
        else:
            from src.document_parser import parse_rfp
            requirements = parse_rfp(str(args.input))
        requirements = validate_requirements(requirements)
        classifier = None
        if len(requirements) > 1:
            print("로컬 모델 로딩 중...", file=sys.stderr, flush=True)
            classifier = RequirementClassifier(InferenceConfig(
                adapter=args.adapter, device=args.device, local_files_only=not args.allow_download,
                max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens))
        report = analyze_requirements(requirements, classifier, batch_size=args.batch_size,
                                      detailed_pairs=args.detailed_pairs,
                                      progress_callback=lambda done, total: print(f"비교 {done}/{total}", file=sys.stderr, flush=True))
        report["run"]["input_path"] = str(args.input.resolve())
        save_analysis(report, args.out, args.input, args.overwrite)
        print(json.dumps(report["stats"], ensure_ascii=False))
        print(f"저장: {args.out}, {args.out.with_suffix('.csv')}")
        return 1 if report["stats"]["failed_pairs"] else 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"문서 분석 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
