# 팀원3 기본 모델 추론 실행 가이드

현재 구현 범위는 환경 확인, Qwen3 기본 모델 및 학습된 Adapter 로딩, 요구사항 쌍 분류, 문서 전체 그룹화·원문 기반 요약, CSV/JSON 저장 및 화면 연결입니다. QLoRA 학습과 Adapter 추론 명령은 README의 로컬 학습 안내를 참고하세요. Adapter를 사용할 때는 `uv sync --locked --extra training --python 3.13`으로 PEFT도 설치한 뒤 `--adapter <학습실행폴더/adapter>`를 추가합니다. 화면 실행은 [화면 가이드](guide_app.md)를 참고하세요.

## 1. 환경 설치

프로젝트 최상위 폴더에서 실행합니다. 검증 환경은 Windows, Python 3.13.18, RTX 4070 Ti 12GB, PyTorch 2.9.1+cu128, Transformers 4.57.6입니다. 기본 모델은 `Qwen/Qwen3-1.7B`, 리비전은 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`로 고정했습니다.

`uv`가 없는 PC에서는 먼저 `python -m pip install uv`로 설치하고, 아래 명령을 실행하세요.

```powershell
uv sync --locked --extra inference --python 3.13
uv run --locked --extra inference python -X utf8 main.py --check-env
```

설치 시 `.venv`를 만들고 공식 CUDA 12.8용 PyTorch를 선택합니다. 현재 프로젝트의 지원 Python 범위는 3.13~3.14이며 실제 검증은 3.13에서 했습니다. 3.15는 일부 잠금 패키지의 wheel 미지원으로 제외했습니다. `--check-env` 결과의 `ready: true`, `cuda_available: true`, GPU 이름을 확인하세요. GPU를 사용할 수 없으면 기본 실행은 오류로 중단합니다. CPU 실행은 `--device cpu`로 명시할 수 있지만 속도와 메모리 사용량은 별도로 확인해야 합니다.

첫 실행에는 모델 다운로드가 필요합니다. 모델은 프로젝트의 `.cache/huggingface/`에 저장되며 Git에서 제외됩니다. 요구사항 본문은 로컬 모델에서 처리합니다. CUDA Toolkit을 별도로 설치하지 않고 PyTorch wheel과 NVIDIA 드라이버로 검증했습니다.

`uv sync`나 `uv run`을 쓸 때는 `--extra inference`를 포함하세요. 생략하면 추론용 선택 패키지가 환경에서 제거될 수 있습니다.

## 2. 예제 CSV 실행

```powershell
uv run --locked --extra inference python -X utf8 main.py --pairs examples/inference_pairs.csv --out src/data/output/demo_base_pred.csv
```

이미 환경과 모델을 설치한 현재 PC에서는 아래처럼 바로 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --pairs examples/inference_pairs.csv --out src/data/output/demo_base_pred.csv --local-files-only --overwrite
```

`--local-files-only`는 다운로드 없이 기존 캐시만 사용합니다. 모델을 로컬 snapshot 경로로 로딩하므로 토크나이저의 추가 Hub 조회도 하지 않습니다. 기존 결과는 기본적으로 덮어쓰지 않으며, 의도적으로 다시 생성할 때만 `--overwrite`를 붙입니다.

예제는 기능 점검용으로 직접 만든 3쌍이며 학습·평가 데이터가 아닙니다. 정답 예상을 세우면 DEMO-001은 중복, DEMO-002는 유사·관련, DEMO-003은 무관입니다. 현재 기본 모델 실행에서는 각각 유사·관련, 유사·관련, 무관으로 출력했습니다. **실행 성공과 분류 정확도는 별개이며, 첫 사례를 오분류했습니다.** 라벨링 데이터로 별도 평가하고 학습해야 합니다.

## 3. 문서 파서 JSON에서 두 요구사항 비교

먼저 [파서 가이드](guide_parser.md)에 따라 JSON을 생성합니다. 다음 예제는 샘플 PDF의 REQ-020과 REQ-021을 비교합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --requirements "src/data/output/제안요청서.pdf.requirements.json" --a REQ-020 --b REQ-021 --case-id PDF-020-021 --out src/data/output/pdf_base_pred.csv --local-files-only
```

입력 JSON의 `req_text`를 사용하고, `req_id`, `source_req_id`, `doc_name`, `doc_type`, `source_location`을 결과에 연결합니다. 원본 JSON과 `raw_req_text`는 수정하지 않습니다. JSON 입력은 한 문서 기준입니다. 여러 문서를 합칠 때는 먼저 문서별로 겹치지 않는 요구사항 ID를 부여해야 합니다.

## 4. 팀원1의 요구사항 쌍 CSV 연결

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --pairs src/data/output/pairs_gt.csv --split test --out src/data/output/base_pred.csv --local-files-only
```

- 필수 열: `case_id`, `req_a_id`, `req_a_text`, `req_b_id`, `req_b_text`. `--split train/val/test` 사용 시 `split`도 필요합니다.
- 기본 선택은 `test`입니다. `--split all`을 지정하면 전부 처리합니다. 이미 정해진 분할을 다시 나누지 않습니다.
- `label`, `reason`, `diff_conditions`가 있는 정답 CSV도 읽지만, 이 열들은 프롬프트에 넣지 않습니다. 빈 초안도 추론할 수 있습니다. 모델 출력으로 정답 열을 자동으로 채우지 않습니다.
- ID와 문서명은 모델 판정 근거에 들어가지 않습니다. 추적 정보로 결과에만 보존합니다.
- `req_a_doc_name`, `req_a_doc_type`, `req_a_source_location`, `req_a_source_req_id` 및 B의 같은 열이 있으면 결과에 보존합니다. 없으면 빈 문자열입니다.
- 잘린 CSV, 필수 항목 누락, 같은 ID끼리의 비교, 중복 `case_id`는 모델 로딩 전에 거절합니다. 따옴표 안의 본문 줄바꿈은 유지합니다.

## 5. 출력과 오류 처리

`--out base_pred.csv`를 지정하면 같은 폴더에 `base_pred.csv`와 `base_pred.json`을 저장합니다.

| 결과 | 내용 |
| --- | --- |
| CSV | 입력 사례·본문·원문 위치, `pred_label`, `pred_reason`, `pred_diff_conditions`, `status`, `error` |
| JSON | 같은 예측과 모델 원응답, 토큰 수, 소요 시간, 모델 리비전·프롬프트 버전·생성 설정·실행 환경 |

분류값은 `중복`, `유사·관련`, `무관`만 허용합니다. 형식이 잘못되거나 토큰 상한에 도달하면 `status=error`로 해당 사례를 남기며, 임의로 `무관` 등으로 채우지 않습니다. 오류가 한 건이라도 있으면 프로세스 종료 코드는 1, JSON의 `ready_for_evaluation`은 false입니다. 이 값은 출력 형식의 완전성을 뜻하며 정답률을 보장하지 않습니다.

입력이 기본 4096토큰보다 길면 본문을 자르지 않고 오류로 남깁니다. 출력은 최대 384토큰입니다. 필요하면 `--max-input-tokens`, `--max-new-tokens`를 조정하되 GPU 메모리를 확인하세요. 판단 근거는 짧게 출력하며 `enable_thinking=False`를 사용합니다. 생성 설정은 temperature 0.7, top_p 0.8, top_k 20, 사례별 seed 42입니다. 동일 환경의 재실행 기준을 고정하지만 서로 다른 하드웨어·라이브러리에서 완전히 같은 출력은 보장하지 않습니다.

PR #2의 `evaluate.py`는 예측 누락·잘못된 라벨을 제외하고 점수를 계산하는 문제가 확인된 임시 버전입니다. 평가기를 완성한 뒤 **오류 0건이고 모든 평가 case_id가 일치하는 예측 파일**로 성능을 비교하세요. 모델의 판정과 이유·차이점은 사람이 검토해야 합니다.

## 6. 화면에서 호출할 함수

```python
from src.inference import InferenceConfig, RequirementClassifier, pair_from_requirements
from pathlib import Path

classifier = RequirementClassifier(InferenceConfig(local_files_only=True))
pair = pair_from_requirements(Path("src/data/output/제안요청서.pdf.requirements.json"), "REQ-020", "REQ-021")
result = classifier.predict_pair(pair)
```

화면에서는 모델 객체를 한 번 생성해 재사용하세요. `result["status"]`가 `ok`일 때 예측을 표시하고, `error`이면 오류 내용을 보여주세요. 위 함수는 한 쌍용이며 문서 전체는 다음 API를 사용합니다.

```python
from src.analysis import analyze_requirements
from src.document_parser import parse_rfp

requirements = parse_rfp("제안요청서.pdf")
report = analyze_requirements(requirements, classifier, batch_size=4)
groups = report["groups"]
```

`report`에는 모든 원문, 쌍별 결과, 그룹, 오류·경고, 처리 통계가 포함됩니다. 기본 문서 분석은 `predict_relations()`를 사용해 기존 프롬프트의 첫 라벨이 완성되면 생성을 멈춥니다. 쌍별 설명은 `explanation_status=not_requested`, `response_format=label_prefix`와 빈 값으로 명시합니다. 상세 설명까지 생성하려면 `analyze_requirements(..., detailed_pairs=True)` 또는 CLI `--detailed-pairs`를 사용합니다. 기존 `predict_pairs()`와 `main.py`는 상세 응답을 계속 지원합니다.

문서 배치는 greedy decoding을 사용하며 실제 판정 방식·설명 생성 건수·토큰 설정은 `run`에 기록됩니다. 단독 항목과 분류 실패도 보존합니다. 그룹 요약은 두 모드 모두 기본 모델이 원문 ID를 선택하고 코드가 해당 원문을 표시하는 방식이며, 실패하면 원문 추출 요약으로 대체합니다. 학습 Adapter는 관계 분류에만 적용합니다. 요약에 선택되지 않은 조건도 원문별 보존 항목에 남기므로 그룹의 `diff_conditions`를 자동 확정된 의미 차이로 해석하지 마세요.

```powershell
.\.venv\Scripts\python.exe -m src.analysis --input "src/data/output/제안요청서.pdf.requirements.json" --out src/data/output/document_analysis.json --adapter artifacts/rfp3-qlora-relabelled-20261127/adapter --batch-size 4
```

기본적으로 다운로드하지 않습니다. GPU 메모리가 부족하면 `--batch-size 2` 또는 `1`을 사용하세요. 입력·출력 토큰 제한을 넘긴 경우 원문을 자르거나 정상 결과로 처리하지 않습니다. 분석에 실패한 쌍이 있으면 결과를 저장하고 종료 코드 1을 반환합니다.

같은 최신 Adapter로 빠른 판정과 상세 설명 생성을 비교하는 도구는 `scripts/benchmark_inference_modes.py`입니다. 검증 9쌍에서 정책을 고정한 뒤 test 18쌍을 측정한 결과, 모든 라벨이 같았고 정확도 94.4%·Macro F1 0.9218·오류 0건을 유지했습니다. 준비 실행 이후 쌍별 추론은 상세 51.429초(1회), 빠른 3.376초(3회 중앙값)였습니다. 모델 로딩과 그룹 요약 시간은 포함하지 않습니다. 상세 측정 조건과 제한은 README의 빠른 문서 분석 검증을 참고하세요.

## 7. 검증

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v
```

모델 다운로드 없이 입력 검증, 정답 열 제외, 응답 JSON 검사, 원문 위치 보존, 오류 사례 보존, 긴 본문 처리, 기존 파일 보호를 검사합니다. 실제 GPU 실행으로 예제 3쌍과 PDF 요구사항 1쌍의 생성·저장을 별도 확인했습니다. 이는 동작 점검이며 정식 성능 평가가 아닙니다.

설치·생성 설정 참고: [Qwen 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-1.7B), [PyTorch 공식 이전 버전 설치 안내](https://pytorch.org/get-started/previous-versions/), [uv의 PyTorch 설정](https://docs.astral.sh/uv/guides/integration/pytorch/).
