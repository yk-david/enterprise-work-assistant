"""문서 추출과 팀원3의 실제 분류·그룹화·요약을 연결하는 Streamlit 화면."""

import gc
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TypedDict

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_parser import parse_rfp

LATEST_ADAPTER = PROJECT_ROOT / "artifacts" / "rfp3-qlora-relabelled-20261127" / "adapter"
PREVIOUS_ADAPTER = PROJECT_ROOT / "artifacts" / "rfp3-qlora-20261127" / "adapter"
COMPARISON_PATH = LATEST_ADAPTER.parent / "evaluation" / "comparison.json"
BASE_MODEL_LABEL = "기본 모델"
LATEST_MODEL_LABEL = "최신 라벨 학습 모델"
PREVIOUS_MODEL_LABEL = "기존 라벨 학습 모델"


def adapter_metadata(adapter: Path) -> dict | None:
    """완료된 학습과 저장된 Adapter의 데이터 버전이 일치하는 경우만 노출합니다."""
    try:
        info = json.loads((adapter / "training_info.json").read_text(encoding="utf-8"))
        result = json.loads((adapter.parent / "training_result.json").read_text(encoding="utf-8"))
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        weights = adapter / "adapter_model.safetensors"
        if not all(isinstance(item, dict) for item in (info, result, config)):
            return None
        from src.inference import DEFAULT_MODEL, DEFAULT_REVISION, SYSTEM_PROMPT

        if (info.get("base_model") != DEFAULT_MODEL or info.get("revision") != DEFAULT_REVISION
                or info.get("system_prompt_sha256") != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
                or config.get("peft_type") != "LORA"):
            return None
        dataset_hash = info.get("dataset_sha256")
        if (result.get("completed") is not True or not weights.is_file() or weights.stat().st_size < 8
                or not isinstance(dataset_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", dataset_hash)):
            return None
        if any(info.get(key) != result.get(key) for key in (
                "dataset_sha256", "base_model", "revision", "system_prompt_sha256", "selected_epoch", "split_counts")):
            return None
        if not isinstance(info.get("split_counts"), dict) or any(
                type(info["split_counts"].get(split)) is not int or info["split_counts"][split] < 1
                for split in ("train", "val", "test")):
            return None
        return info
    except (OSError, ValueError, TypeError):
        return None


def available_models() -> dict[str, dict]:
    models = {}
    for key, label, adapter in (("latest", LATEST_MODEL_LABEL, LATEST_ADAPTER),
                                ("previous", PREVIOUS_MODEL_LABEL, PREVIOUS_ADAPTER)):
        info = adapter_metadata(adapter)
        if info is not None:
            models[key] = {"label": label, "adapter": adapter, "training": info}
    models["base"] = {"label": BASE_MODEL_LABEL, "adapter": None, "training": None}
    return models


def selected_model(model_key: str) -> dict:
    models = available_models()
    if not isinstance(model_key, str) or model_key not in models:
        raise ValueError("선택한 모델이 없거나 학습 파일이 완성되지 않았습니다. 모델 목록을 확인하세요.")
    return models[model_key]


def read_model_comparison(models: dict[str, dict]) -> dict | None:
    """다른 데이터·모델의 평가를 현재 화면의 성능으로 표시하지 않습니다."""
    if not {"latest", "previous"}.issubset(models):
        return None
    try:
        report = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
        current_hash = models["latest"]["training"]["dataset_sha256"]
        if report["dataset"]["sha256"] != current_hash:
            return None
        for entry_key, model_key in (("current", "latest"), ("previous", "previous")):
            entry = report["models"][entry_key]
            model = models[model_key]
            if (entry["training_dataset_sha256"] != model["training"]["dataset_sha256"]
                    or Path(entry["adapter_path"]).resolve() != model["adapter"].resolve()):
                return None
        if not isinstance(report["dataset"]["test_count"], int) or report["dataset"]["test_count"] < 1:
            return None
        output_limit = report.get("generation", {}).get("max_new_tokens")
        if type(output_limit) is not int or output_limit < 1:
            return None
        for entry in report["models"].values():
            metrics = entry["metrics"]
            if metrics["total"] != report["dataset"]["test_count"] or not isinstance(metrics["errors"], list):
                return None
            if any(type(metrics[key]) not in (int, float) or not math.isfinite(metrics[key])
                   or not 0 <= metrics[key] <= 1 for key in ("accuracy", "macro_f1")):
                return None
            seconds = entry.get("inference_seconds")
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                return None
        return report
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def render_model_comparison(models: dict[str, dict], configured_max_new_tokens: int = 384) -> None:
    report = read_model_comparison(models)
    if report is None:
        return
    with st.expander("같은 평가 데이터의 모델 성능 비교"):
        st.caption("아래 기록은 각 쌍의 상세 설명까지 생성한 기존 방식의 비교 결과입니다. 빠른 그룹 분석의 처리 시간과 구분해 주세요.")
        labels = {"base": BASE_MODEL_LABEL, "previous": PREVIOUS_MODEL_LABEL, "current": LATEST_MODEL_LABEL}
        rows = []
        for key in ("base", "previous", "current"):
            if key in report["models"]:
                metrics = report["models"][key]["metrics"]
                rows.append({"모델": labels[key], "정확도": f"{metrics['accuracy']:.1%}",
                             "Macro F1": f"{metrics['macro_f1']:.3f}", "실행 오류": len(metrics["errors"]),
                             "추론 시간(초)": round(report["models"][key]["inference_seconds"], 1)})
        st.dataframe(rows, hide_index=True)
        st.caption(f"같은 테스트 {report['dataset']['test_count']}쌍과 같은 생성 설정으로 비교한 결과입니다. 소규모 평가이며 전체 문서의 정확도를 보장하지 않습니다.")
        evaluated_limit = report["generation"]["max_new_tokens"]
        st.caption(f"이 비교는 응답 최대 {evaluated_limit}토큰 설정으로 측정했습니다. 현재 문서 분석 설정은 {configured_max_new_tokens}토큰입니다.")
        if configured_max_new_tokens != evaluated_limit:
            st.caption("응답 길이 설정을 바꿔도 위 평가 점수는 다시 계산되지 않습니다.")
        st.caption("추론 시간은 모델 로딩을 제외한 한 번의 측정값입니다. 반복 측정한 속도 기준은 아닙니다.")
        previous_seconds = report["models"]["previous"]["inference_seconds"]
        current_seconds = report["models"]["current"]["inference_seconds"]
        if current_seconds > previous_seconds:
            st.caption(f"이번 측정에서 최신 모델은 {current_seconds:.1f}초, 기존 모델은 {previous_seconds:.1f}초로 최신 모델의 추론 시간이 더 길었습니다.")
        if not report.get("all_predictions_complete", False):
            st.warning("일부 모델의 예측에 실행 오류가 있습니다. 누락·오류 사례도 점수 계산에 포함했습니다.")
        old = report["models"]["previous"]["metrics"]
        current = report["models"]["current"]["metrics"]
        if current["macro_f1"] < old["macro_f1"] or current["accuracy"] < old["accuracy"]:
            st.warning("최신 라벨 학습 모델의 평가 지표 중 일부가 기존 모델보다 낮습니다. 두 모델의 판정과 원문을 비교해 주세요.")


def read_speed_comparison(models: dict[str, dict]) -> dict | None:
    """현재 최신 Adapter와 같은 데이터·가중치로 측정한 test 결과만 읽습니다."""
    if "latest" not in models:
        return None
    try:
        latest = models["latest"]
        adapter = latest["adapter"]
        path = adapter.parent / "speed_optimization" / "test" / "performance.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report["split"] != "test" or report["dataset_sha256"] != latest["training"]["dataset_sha256"]
                or Path(report["adapter_path"]).resolve() != adapter.resolve()
                or type(report["test_count"]) is not int or report["test_count"] < 1):
            return None
        with (adapter / "adapter_model.safetensors").open("rb") as stream:
            if report["adapter_weights_sha256"] != hashlib.file_digest(stream, "sha256").hexdigest():
                return None
        if any(type(report[key]) is not bool for key in ("labels_identical", "accuracy_preserved")):
            return None
        for key in ("max_new_tokens", "batch_size", "relation_max_new_tokens"):
            if type(report["settings"][key]) is not int or report["settings"][key] < 1:
                return None
        for mode in ("full", "fast"):
            entry = report[mode]
            metrics = entry["metrics"]
            if (metrics["total"] != report["test_count"] or not isinstance(metrics["errors"], list)
                    or not isinstance(entry["runs"], list) or not entry["runs"]):
                return None
            if any(type(metrics[key]) not in (int, float) or not math.isfinite(metrics[key])
                   or not 0 <= metrics[key] <= 1 for key in ("accuracy", "macro_f1")):
                return None
            if (type(entry["median_seconds"]) not in (int, float)
                    or not math.isfinite(entry["median_seconds"]) or entry["median_seconds"] <= 0):
                return None
        if type(report["speedup"]) not in (int, float) or not math.isfinite(report["speedup"]) or report["speedup"] <= 0:
            return None
        return report
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def render_speed_comparison(models: dict[str, dict]) -> None:
    report = read_speed_comparison(models)
    if report is None:
        return
    with st.expander("최신 모델의 빠른 분석 성능 검증"):
        rows = []
        for key, label in (("full", "쌍별 상세 설명"), ("fast", "빠른 관계 판정")):
            entry = report[key]
            metrics = entry["metrics"]
            rows.append({"처리 방식": label, "정확도": f"{metrics['accuracy']:.1%}",
                         "Macro F1": f"{metrics['macro_f1']:.3f}", "실행 오류": len(metrics["errors"]),
                         "추론 시간(초)": round(entry["median_seconds"], 3)})
        st.dataframe(rows, hide_index=True)
        agreement = "모든 판정 일치" if report["labels_identical"] else "일부 판정 다름"
        preserved = "분류 정확도 유지" if report["accuracy_preserved"] else "분류 정확도 유지 조건 미충족"
        st.caption(f"같은 최신 모델·평가 {report['test_count']}쌍: {agreement} · {preserved} · 측정 속도 {report['speedup']:.2f}배")
        st.caption(f"상세 설명은 {len(report['full']['runs'])}회 측정, 빠른 판정은 {len(report['fast']['runs'])}회 측정의 중앙값입니다. 모델 로딩은 제외했습니다.")
        settings = report["settings"]
        st.caption(f"측정 설정: 배치 {settings['batch_size']}, 상세 응답 최대 {settings['max_new_tokens']}토큰, 빠른 관계 판정 최대 {settings['relation_max_new_tokens']}토큰. 현재 화면 설정을 바꾸어도 이 기록은 다시 계산되지 않습니다.")
        st.caption("빠른 방식은 각 쌍의 긴 설명을 기본 생성하지 않습니다. 그룹 요약·원문·조건 보존은 계속 제공하지만, 위 시간은 쌍별 추론만 측정했으며 그룹 요약 시간을 포함하지 않습니다.")
        st.caption("소규모 평가에서의 결과이며 새로운 문서의 정확도나 전체 처리 시간을 보장하지 않습니다.")
        if not report["accuracy_preserved"] or not report["labels_identical"]:
            st.warning("두 방식의 판정 또는 정확도 보존 조건이 일치하지 않습니다. 상세 결과와 원문을 확인해 주세요.")


class GroupResult(TypedDict):
    group_id: str
    group_name: str
    summary: str
    req_ids: list[str]
    diff_conditions: str
    source_locations: list[str]


GROUP_COLUMNS = {
    "group_id": "그룹 ID",
    "group_name": "그룹 이름",
    "summary": "요약",
    "req_ids": "포함 요구사항 ID",
    "diff_conditions": "조건·차이 검토",
    "source_locations": "원문 위치",
}


def parse_uploaded_file(filename: str, content: bytes) -> list[dict]:
    """문서 형식을 확인하고 임시 파일을 통해 기존 파서를 호출합니다."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".hwp", ".hwpx"}:
        raise ValueError("PDF, HWP 또는 HWPX 파일을 선택해 주세요.")
    if not content:
        raise ValueError("파일이 비어 있습니다.")
    with TemporaryDirectory(prefix="reqcheck-") as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(content)
        requirements = parse_rfp(path)
    for requirement in requirements:
        requirement["doc_name"] = filename
    return requirements


def clear_analysis() -> None:
    """다른 파일·모델의 결과가 남아 있지 않도록 분석 상태를 지웁니다."""
    for key in ("analysis_result", "analysis_model"):
        st.session_state.pop(key, None)


def clear_extraction() -> None:
    for key in ("requirements", "requirement_id"):
        st.session_state.pop(key, None)
    clear_analysis()


def sync_uploaded_file(filename: str | None, content: bytes | None) -> None:
    """같은 이름의 새 파일도 구별하고, 파일 제거 시 모든 결과를 비웁니다."""
    fingerprint = None
    if filename is not None and content is not None:
        fingerprint = (filename, hashlib.sha256(content).hexdigest())
    if st.session_state.get("upload_fingerprint") != fingerprint:
        clear_extraction()
        st.session_state["upload_fingerprint"] = fingerprint


class _ModelSlot:
    """세션들이 GPU 모델 하나를 공유하며, 모델 교체는 진행 중 분석이 끝난 뒤 수행합니다."""

    def __init__(self):
        self.lock = RLock()
        self.classifier = None
        self.adapter = None
        self.max_new_tokens = None

    def load(self, model_key: str, max_new_tokens: int = 384):
        from src.inference import InferenceConfig, RequirementClassifier

        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("응답 최대 토큰은 1 이상의 정수여야 합니다.")
        adapter = selected_model(model_key)["adapter"]
        with self.lock:
            if (self.classifier is not None and self.adapter == adapter
                    and self.max_new_tokens == max_new_tokens):
                return self.classifier
            # 새 모델을 먼저 로딩하면 두 모델이 동시에 GPU 메모리를 점유하게 됩니다.
            self.classifier = None
            self.adapter = None
            self.max_new_tokens = None
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.classifier = RequirementClassifier(InferenceConfig(
                adapter=adapter, local_files_only=True, max_new_tokens=max_new_tokens))
            self.adapter = adapter
            self.max_new_tokens = max_new_tokens
            return self.classifier


@st.cache_resource(show_spinner=False)
def model_slot():
    return _ModelSlot()


def load_classifier(model_key: str, max_new_tokens: int = 384):
    return model_slot().load(model_key, max_new_tokens=max_new_tokens)


def run_analysis(requirements: list[dict], model_key: str, progress_callback=None, batch_size: int = 4,
                 max_new_tokens: int = 384, detailed_pairs: bool = False) -> dict:
    """UI 외부에서도 교체·검증할 수 있는 실제 분석 호출 경계입니다."""
    from src.analysis import analyze_requirements

    # 모델 교체와 추론을 같은 잠금 범위에 두어 여러 세션의 분석을 안전하게 순서대로 실행합니다.
    with model_slot().lock:
        classifier = load_classifier(model_key, max_new_tokens=max_new_tokens) if len(requirements) > 1 else None
        return analyze_requirements(requirements, classifier, progress_callback=progress_callback,
                                    batch_size=batch_size, detailed_pairs=detailed_pairs)


def render_requirements(requirements: list[dict]) -> None:
    st.subheader(f"추출한 요구사항 {len(requirements)}건")
    first = requirements[0]
    st.caption(f"문서: {first.get('doc_name', '')} · 형식: {first.get('doc_type', '')}")
    st.dataframe(
        [{"추출 ID": item["req_id"], "원문 번호": item.get("source_req_id", ""),
          "원문 위치": item.get("source_location", "")} for item in requirements],
        hide_index=True,
    )
    st.caption("PDF 위치는 파일의 페이지 번호입니다. HWP·HWPX는 추출 위치와 원문 번호를 함께 확인하세요.")
    by_id = {item["req_id"]: item for item in requirements}
    selected_id = st.selectbox(
        "본문을 볼 요구사항", options=list(by_id), key="requirement_id",
        format_func=lambda req_id: (
            f"{req_id} · {by_id[req_id].get('source_req_id', '')} · "
            f"{by_id[req_id].get('source_location', '')}"
        ),
    )
    item = by_id[selected_id]
    st.text(item["req_text"])
    for warning in item.get("warnings", []):
        st.warning(warning)
    if item.get("raw_req_text", item["req_text"]) != item["req_text"]:
        with st.expander("정리 전 추출 본문"):
            st.text(item["raw_req_text"])


def group_rows(groups: list[GroupResult]) -> list[dict[str, str]]:
    return [{
        "group_id": group["group_id"], "group_name": group["group_name"],
        "summary": group["summary"], "req_ids": "|".join(group["req_ids"]),
        "diff_conditions": group["diff_conditions"],
        "source_locations": " | ".join(group["source_locations"]),
    } for group in groups]


def group_csv(rows: list[dict[str, str]]) -> bytes:
    from src.analysis import group_csv_bytes

    return group_csv_bytes(rows)


def analysis_json(result: dict) -> bytes:
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def render_analysis(result: dict, requirements: list[dict]) -> None:
    stats = result["stats"]
    result_title = f"요구사항 {stats['requirement_count']}건 → {stats['group_count']}개 그룹"
    if stats["failed_pairs"]:
        st.warning(f"일부 비교를 완료하지 못했습니다. {result_title}")
    else:
        st.success(result_title)
    st.caption(
        f"비교 {stats['pair_count']}쌍 · 성공 {stats['successful_pairs']}쌍 · "
        f"실패 {stats['failed_pairs']}쌍 · 분석 {stats['elapsed_seconds']:.1f}초"
    )
    if result.get("run", {}).get("pair_inference_method") == "predict_relations":
        counts = result["run"]["pair_explanation_counts"]
        st.caption(f"관계만 판정 {counts['not_requested']}쌍 · 상세 설명으로 대체 {counts['generated']}쌍. 그룹 요약과 원문별 조건은 별도로 제공합니다.")
    for warning in result.get("warnings", []):
        st.warning(warning)
    rows = group_rows(result["groups"])
    st.dataframe(rows, column_config=GROUP_COLUMNS, hide_index=True)
    by_id = {item["req_id"]: item for item in requirements}
    for group in result["groups"]:
        with st.expander(f"{group['group_id']} · {group['group_name']}"):
            st.text(group["summary"])
            st.write("조건·차이 검토")
            st.text(group["diff_conditions"] or "없음")
            for warning in group.get("warnings", []):
                st.warning(warning)
            st.write("원문 확인")
            for req_id in group["req_ids"]:
                item = by_id[req_id]
                st.caption(f"{req_id} · {item.get('source_req_id', '')} · {item.get('source_location', '')}")
                st.text(item["req_text"])
    failed = [pair for pair in result["pair_results"] if pair.get("status") != "ok"]
    if failed:
        with st.expander(f"비교하지 못한 쌍 {len(failed)}건"):
            st.caption("실패한 비교는 무관 판정과 구분됩니다. 원문과 오류를 확인하고 다시 분석할 수 있습니다.")
            st.caption("응답 길이 제한 오류이면 처리 설정의 응답 최대 토큰을 768로 높여 다시 분석해 주세요.")
            st.dataframe([
                {"요구사항 A": pair["req_a_id"], "요구사항 B": pair["req_b_id"],
                 "오류": pair.get("error", "")} for pair in failed
            ], hide_index=True)
    st.download_button("그룹 결과 CSV 다운로드", data=group_csv(rows),
                       file_name="reqcheck-groups.csv", mime="text/csv", on_click="ignore")
    st.download_button("전체 분석 JSON 다운로드", data=analysis_json(result),
                       file_name="reqcheck-analysis.json", mime="application/json", on_click="ignore")


def main() -> None:
    st.set_page_config(page_title="ReqCheck", layout="wide")
    st.title("ReqCheck")
    extraction_tab, analysis_tab = st.tabs(["요구사항 추출", "문서 분석 결과"])
    with extraction_tab:
        uploaded_file = st.file_uploader("제안요청서 업로드", type=["pdf", "hwp", "hwpx"],
                                         on_change=clear_extraction)
        content = uploaded_file.getvalue() if uploaded_file is not None else None
        sync_uploaded_file(uploaded_file.name if uploaded_file is not None else None, content)
        if st.button("요구사항 추출", disabled=uploaded_file is None, type="primary"):
            clear_extraction()
            try:
                with st.spinner("요구사항을 추출하고 있어요."):
                    st.session_state["requirements"] = parse_uploaded_file(uploaded_file.name, content)
                if not st.session_state["requirements"]:
                    st.warning("추출한 요구사항이 없습니다. 문서의 요구사항 표와 형식을 확인해 주세요.")
            except Exception as error:
                st.error(f"요구사항을 추출하지 못했습니다. {error}")
        requirements = st.session_state.get("requirements", [])
        if requirements:
            render_requirements(requirements)

    with analysis_tab:
        models = available_models()
        choices = list(models)
        if st.session_state.get("analysis_model_choice") not in models:
            st.session_state["analysis_model_choice"] = choices[0]
            clear_analysis()
        model_key = st.selectbox("분석 모델", choices, key="analysis_model_choice",
                                 format_func=lambda key: models[key]["label"], on_change=clear_analysis)
        model = models[model_key]
        model_label = model["label"]
        if model["training"] is not None:
            counts = model["training"]["split_counts"]
            st.caption(f"{model_label} · 학습 {counts['train']}쌍 / 검증 {counts['val']}쌍 / 평가 {counts['test']}쌍")
            if model_key == "previous":
                st.caption("이 모델은 수정 전 라벨로 학습한 비교용 버전입니다.")
        with st.expander("처리 설정"):
            detailed_pairs = st.checkbox("쌍별 상세 설명 생성", value=False, key="analysis_detailed_pairs",
                                         on_change=clear_analysis,
                                         help="기본 분석은 관계 판정 후 그룹 요약과 원문별 조건을 제공합니다. 이 옵션을 켜면 모든 쌍의 상세 근거·차이점도 생성해 JSON에 저장하므로 더 오래 걸립니다.")
            batch_size = st.selectbox("한 번에 비교할 쌍 수", [4, 2, 1], key="analysis_batch_size",
                                      on_change=clear_analysis,
                                      help="GPU 메모리 부족 오류가 나면 2 또는 1로 줄여 다시 분석하세요.")
            max_new_tokens = st.selectbox("응답 최대 토큰", [384, 768, 1024], key="analysis_max_new_tokens",
                                          on_change=clear_analysis,
                                          help="응답 길이 제한 오류가 나면 768 또는 1024로 늘려 다시 분석하세요. 처리 시간이 늘어날 수 있습니다.")
        render_model_comparison(models, configured_max_new_tokens=max_new_tokens)
        render_speed_comparison(models)
        if not detailed_pairs:
            st.caption("빠른 그룹 분석: 각 쌍은 관계를 판정하고, 그룹 요약·원문·조건은 그대로 제공합니다. 쌍별 상세 설명은 기본적으로 생성하지 않으며, 관계를 확정하지 못한 쌍은 상세 방식으로 다시 처리합니다.")
        else:
            st.caption("상세 분석: 모든 쌍의 관계와 상세 근거·차이점까지 생성해 JSON에 저장합니다.")
        if not requirements:
            st.info("먼저 제안요청서를 업로드하고 요구사항을 추출해 주세요.")
        else:
            count = len(requirements)
            st.caption(f"요구사항 {count}건의 {count * (count - 1) // 2}쌍을 비교하고 그룹별 요약을 만듭니다. 문서 길이에 따라 시간이 걸릴 수 있습니다.")
        if st.button("문서 전체 분석", disabled=not requirements, type="primary"):
            clear_analysis()
            progress = st.progress(0.0, text="모델을 준비하고 있어요.")

            def update_progress(completed, total):
                progress.progress(completed / total if total else 1.0,
                                  text=f"요구사항 비교 {completed}/{total}쌍 · 비교가 끝나면 그룹 요약을 생성합니다.")

            try:
                with st.spinner("문서 전체를 비교하고 그룹 요약을 만들고 있어요."):
                    result = run_analysis(requirements, model_key, update_progress, batch_size=batch_size,
                                          max_new_tokens=max_new_tokens, detailed_pairs=detailed_pairs)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_model"] = model_label
            except Exception as error:
                st.error(f"문서를 분석하지 못했습니다. {error}")
                st.info("학습·추론 패키지와 로컬 모델 파일을 먼저 준비해 주세요. 실행 방법은 guide_app.md를 확인하세요.")
            finally:
                progress.empty()
        result = st.session_state.get("analysis_result")
        if result is not None:
            if result["stats"]["pair_count"]:
                st.caption(f"사용 모델: {st.session_state.get('analysis_model', model_label)}")
            else:
                st.caption("단독 요구사항은 모델 비교 없이 원문에서 요약했습니다.")
            render_analysis(result, requirements)


if __name__ == "__main__":
    main()
