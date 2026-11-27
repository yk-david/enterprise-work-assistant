"""Conservative groups: every member pair must have a successful related prediction."""

from itertools import combinations

RELATED_LABELS = {"중복", "유사·관련"}
VALID_LABELS = RELATED_LABELS | {"무관"}


def validate_requirements(requirements: list[dict]) -> list[dict]:
    """Validate source records without dropping parser metadata or changing text."""
    if not isinstance(requirements, list):
        raise ValueError("요구사항은 배열이어야 합니다.")
    result, seen = [], set()
    for row in requirements:
        if not isinstance(row, dict):
            raise ValueError("요구사항은 객체여야 합니다.")
        req_id, text = row.get("req_id"), row.get("req_text")
        if not isinstance(req_id, str) or not req_id.strip():
            raise ValueError("비어 있지 않은 req_id가 필요합니다.")
        req_id = req_id.strip()
        if req_id in seen:
            raise ValueError(f"중복 req_id: {req_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"비어 있지 않은 req_text가 필요합니다: {req_id}")
        if text.strip() in {"#NAME?", "#REF!", "#VALUE!", "#N/A", "#DIV/0!", "#NUM!", "#NULL!"}:
            raise ValueError(f"스프레드시트 오류가 본문에 저장되어 있습니다: {req_id}")
        for field in ("doc_name", "doc_type", "source_location", "source_req_id"):
            if field in row and not isinstance(row[field], str):
                raise ValueError(f"{field}는 문자열이어야 합니다: {req_id}")
        seen.add(req_id)
        result.append({**row, "req_id": req_id})
    return result


def build_groups(requirements: list[dict], pair_results: list[dict]) -> list[list[dict]]:
    """Merge fully related groups, preferring duplicate edges with stable ID ties.

This deterministic clique partition is conservative, not a maximum-clique solver.
Missing/error results never create links. Input order does not affect membership.
"""
    sources = validate_requirements(requirements)
    by_id = {row["req_id"]: row for row in sources}
    if not isinstance(pair_results, list):
        raise ValueError("쌍 판정 결과는 배열이어야 합니다.")
    relationships, seen = {}, set()
    for row in pair_results:
        if not isinstance(row, dict):
            raise ValueError("쌍 판정 결과는 객체여야 합니다.")
        a, b = row.get("req_a_id"), row.get("req_b_id")
        if not isinstance(a, str) or not isinstance(b, str) or a not in by_id or b not in by_id:
            raise ValueError(f"원문에 없는 쌍 ID: {a!r}, {b!r}")
        if a == b:
            raise ValueError(f"자기 자신과 비교한 쌍: {a}")
        key = tuple(sorted((a, b)))
        if key in seen:
            raise ValueError(f"중복된 쌍 판정: {key}")
        seen.add(key)
        for side, req_id in (("a", a), ("b", b)):
            field = f"req_{side}_text"
            if field in row and row[field] != by_id[req_id]["req_text"]:
                raise ValueError(f"쌍 본문과 원문이 일치하지 않습니다: {req_id}")
        if row.get("status") not in {"ok", "error"}:
            raise ValueError("쌍 판정 status는 ok 또는 error여야 합니다.")
        if row["status"] == "ok":
            label = row.get("pred_label")
            if not isinstance(label, str) or label not in VALID_LABELS:
                raise ValueError(f"허용되지 않은 분류값: {label!r}")
            relationships[key] = label

    membership = {req_id: {req_id} for req_id in by_id}
    edges = sorted((0 if label == "중복" else 1, *key)
                   for key, label in relationships.items() if label in RELATED_LABELS)
    for _, a, b in edges:
        left, right = membership[a], membership[b]
        if left is right:
            continue
        if all(relationships.get(tuple(sorted((x, y)))) in RELATED_LABELS
               for x in left for y in right):
            merged = left | right
            for req_id in merged:
                membership[req_id] = merged

    grouped_ids = sorted({tuple(sorted(members)) for members in membership.values()})
    flat = [req_id for members in grouped_ids for req_id in members]
    if len(flat) != len(by_id) or set(flat) != set(by_id):
        raise RuntimeError("그룹 결과의 요구사항이 누락되거나 중복되었습니다.")
    for members in grouped_ids:
        if any(relationships.get(pair) not in RELATED_LABELS for pair in combinations(members, 2)):
            raise RuntimeError("그룹 전체의 관련성이 확인되지 않았습니다.")
    return [[by_id[req_id] for req_id in members] for members in grouped_ids]
