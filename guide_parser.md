# RFP 파서 실행 가이드

공유받은 프로젝트에서 PDF·HWP·HWPX 제안요청서의 요구사항을 추출해 내 컴퓨터에 JSON으로 저장하는 방법입니다. 세 형식 모두 같은 `document_parser.py`를 사용합니다. `src/data/input/`에 여러 파일이 있어도 한 번의 실행으로 처리할 수 있습니다.

## 1. 프로젝트 폴더 열기

공유받은 브랜치를 체크아웃하거나 ZIP 파일을 압축 해제한 뒤, VS Code에서 `enterprise-work-assistant` 폴더를 여세요. **Terminal → New Terminal**을 선택하고, 아래 경로가 보이는 프로젝트 최상위 폴더에서 명령을 실행하세요.

| 경로 | 역할 |
| --- | --- |
| `src/document_parser.py` | 실행할 파이썬 스크립트 |
| `src/data/input/` | 처리할 `.pdf`, `.hwp`, `.hwpx` 파일을 넣는 폴더 |
| `src/data/output/` | 결과 JSON이 저장되는 폴더. 실행할 때 자동으로 생성됩니다. |

샘플 HWP와 PDF가 `src/data/input/`에 포함되어 있습니다. 두 파일은 제목만 같고 내용이 다른 문서입니다.

## 2. Python 환경 만들고 파서 실행하기

프로젝트의 `.python-version`은 **Python 3.13**입니다. Python 3.15에서는 현재 잠금 파일의 일부 의존성에 호환 wheel이 없어 설치가 실패하므로, 검증한 3.13 환경을 사용하세요.

### uv 사용 (권장)

`uv`가 설치되어 있다면 다음 명령으로 `uv.lock`에 기록된 PDF·HWP·HWPX 패키지를 함께 설치하고 실행할 수 있습니다. 필요한 Python 3.13도 자동으로 준비됩니다.

```text
uv sync --locked --python 3.13
uv run --locked python -X utf8 src/document_parser.py
```

학습용 패키지도 사용하는 환경에서는 `uv sync`와 `uv run`에 `--extra training`을 함께 지정하세요. 이미 준비된 환경에서는 `.venv`의 Python을 직접 실행해도 됩니다.

### pip 사용

아래 방법은 파서에 필요한 패키지만 설치합니다. 프로젝트 전체의 잠금 버전으로 설치하려면 위의 `uv sync --locked`를 사용하세요.

### Windows PowerShell

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "python-hwpx>=7.0.1,<8" "pdfplumber==0.11.9"
.\.venv\Scripts\python.exe -X utf8 src/document_parser.py
```

`py` 명령을 사용할 수 없다면 설치된 Python 3.13 실행 파일의 경로를 지정하거나 위의 uv 방법을 사용하세요.

### macOS, Linux 또는 WSL

```bash
python3.13 --version
python3.13 -m venv .venv
.venv/bin/python -m pip install 'python-hwpx>=7.0.1,<8' 'pdfplumber==0.11.9'
.venv/bin/python -X utf8 src/document_parser.py
```

설치와 실행에는 `.venv` 안의 같은 Python을 사용합니다. 설치를 마친 뒤 다시 실행할 때는 위 블록의 마지막 줄만 실행하면 됩니다.

## 3. JSON 결과 확인하기

처리가 끝나면 터미널에 파일별 처리 상태와 생성·건너뜀·실패 건수가 표시됩니다. `저장`으로 표시된 파일의 결과는 VS Code에서 `src/data/output/`을 열어 확인하세요.

| 샘플 입력 | 결과 파일 이름 | 샘플 추출 건수 |
| --- | --- | ---: |
| `제안요청서.hwp` | `제안요청서.requirements.json` | 33건 |
| `제안요청서.pdf` | `제안요청서.pdf.requirements.json` | 44건 |
| `복권_제안요청서.hwpx` (별도 확보한 파일) | `복권_제안요청서.hwpx.requirements.json` | 54건 |

PDF와 HWPX 결과 이름에는 각각 `.pdf`, `.hwpx`가 포함됩니다. 원본의 이름이 같아도 형식별 결과는 별도 파일로 저장됩니다. 추출 건수는 입력 문서에 따라 달라집니다.

JSON 배열의 각 항목은 요구사항 한 건입니다.

| 필드 | 의미 |
| --- | --- |
| `req_id` | 파서가 순서대로 붙인 ID |
| `source_req_id` | 원문에 적힌 요구사항 번호 |
| `doc_name`, `doc_type` | 원본 파일 이름과 형식 |
| `source_location` | PDF 페이지 범위(예: `p.32–33`) 또는 HWP/HWPX 표 번호 |
| `req_text` | 확인된 기호와 서식을 정리한 본문 |
| `raw_req_text` | 기호 치환 전 추출 본문. 원본 파일과 함께 대조할 때 사용 |
| `warnings` | 미확인 문자나 문서 끝의 종료 항목 누락 등 검토할 사항 |

PDF 페이지는 인쇄된 쪽수가 아닌 파일의 물리 페이지 기준입니다. 결과는 실행한 컴퓨터의 `src/data/output/`에 UTF-8 JSON으로 저장되며 Git에서 제외됩니다.

## 4. 새 문서를 추가하거나 다시 추출하기

- **새 문서 추가** — `src/data/input/`에 파일을 넣고 실행하세요. 결과 JSON이 없는 입력만 처리합니다.
- **`건너뜀` 확인** — 해당 입력의 결과 JSON이 이미 있다는 뜻입니다. 예를 들어 HWP 결과가 있는 상태에서 PDF를 추가하면 HWP는 `건너뜀`, PDF는 `저장`으로 표시됩니다.
- **기존 문서 다시 추출** — 실행 명령 끝에 `--overwrite`를 붙이세요. 변환에 실패하면 기존 JSON을 보존하고 종료 코드 1을 반환합니다.
- **한 파일만 처리** — `--file "src/data/input/제안요청서.pdf"`처럼 파일 경로를 지정하세요. 출력은 항상 `src/data/output/`에 저장됩니다.

## 5. 글머리표와 깨진 기호 처리

세 형식 모두 추출한 본문을 `raw_req_text`에 보존하고, 같은 전처리를 거친 결과를 `req_text`에 저장합니다. 줄 시작에서 뒤에 공백 또는 줄 끝이 오는 `•`, `●`, `▪`, `‣`, `⁃`, `◦`, `○`, `❍`는 `●`로 통일합니다. 들여쓰기와 줄바꿈, `≥`, `≤`, `%`, `①`, 하위 항목의 `-`는 유지합니다. 본문 중간의 같은 기호는 이 글머리 처리로 바꾸지 않습니다.

### HWP/HWPX의 글머리표

표준 글머리 기호는 별도 치환표 없이 정리됩니다. `ㅇ`은 한글 자모이므로 기본 실행에서는 유지합니다. 해당 문서에서 `ㅇ`을 글머리로 사용하는 것을 확인했다면 `--bullet-chars "ㅇ"`을 지정하세요. **줄 시작 + 뒤 공백/줄 끝**인 경우에만 `●`로 바꾸고, 문장 중간의 `ㅇ`은 보존합니다. 이 옵션은 `--file`로 선택한 한 문서에만 적용됩니다.

아래는 저장소 상위 폴더에 확보한 HWPX와 저장소 안의 HWP를 각각 실행하는 PowerShell 예입니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 src/document_parser.py --file "src/data/input/제안요청서.hwp" --bullet-chars "ㅇ" --overwrite
.\.venv\Scripts\python.exe -X utf8 src/document_parser.py --file "../복권_제안요청서.hwpx" --bullet-chars "ㅇ" --overwrite
```

`--bullet-chars`에는 그 문서에서 글머리임을 확인한 문자를 나열할 수 있습니다. 의미를 확인하지 못한 사설 문자나 `�`를 임의로 추가하지 마세요. 같은 옵션은 PDF에도 사용할 수 있습니다.

### 문서별 사설 문자 치환표

폰트의 사설 영역 문자는 세 형식 모두에서 나타날 수 있으며, UTF-8 저장만으로 의미를 복원할 수 없습니다. 원문과 대조한 치환표를 해당 파일에 적용합니다. 치환표를 PDF용/HWP용/HWPX용으로 일괄 적용하는 것이 아니라 **원본 문서와 사용 폰트에 맞춰** 관리합니다.

저장소의 샘플 PDF를 대조해 확인한 치환표는 `src/data/symbol_maps/sample-rfp-pdf.json`입니다. `U+F09F → ●`, `U+F09E → ·`를 적용합니다.

```text
uv run --locked python -X utf8 src/document_parser.py --file "src/data/input/제안요청서.pdf" --symbol-map src/data/symbol_maps/sample-rfp-pdf.json --overwrite
```

PowerShell에서 pip 방식으로 설치했다면 `uv run --locked python` 대신 `.\.venv\Scripts\python.exe`를 사용하세요.

이 샘플 치환표는 **해당 PDF 전용**입니다. 같은 이름의 HWP나 다른 문서에 그대로 적용하지 마세요. `--symbol-map` 옵션 자체는 PDF/HWP/HWPX 모두 지원하며 `--file`과 함께 사용해야 합니다. `--symbol-map`은 본문 전체의 해당 문자를 바꾸므로, 줄 시작의 글머리만 바꾸려면 `--bullet-chars`를 사용하세요.

뜻을 확인하지 못한 문자는 삭제하지 않고 `warnings`에 표시합니다. 기본 실행은 PDF 사설 문자도 보존하므로, 샘플 PDF의 `●` 복원에는 위 명령을 사용하세요. `warnings`는 치환표를 대신하는 보정 지시가 아니라, 전처리 후 남은 검토 사항입니다.

페이지를 넘는 본문은 종료 항목이 없는 경우와 3페이지 이상 이어지는 경우에도 연결합니다. 앞 요구사항의 남은 본문과 다음 요구사항이 한 표로 잡힌 경우에도 나누어 보존합니다. 연속 여부가 불명확하거나 중간 페이지 표를 추출하지 못하면 조용히 생략하지 않고 오류를 알립니다.

## 6. 수정 검증

```text
uv run --locked python -X utf8 -m unittest discover -s tests -v
```

회귀 테스트는 페이지 경계, 반복 헤더, 기호 치환, 원문 보존, 중복 번호, 파일 선택, 기존 결과의 안전한 재생성을 확인합니다. HWP와 HWPX에도 같은 전처리가 적용되는지, 문서별 글머리 설정이 다른 문서에 영향을 주지 않는지 확인합니다.

실제 검증 파일은 샘플 HWP 33건, 추가 HWP 63건, 복권 HWPX 54건, 샘플 PDF 44건입니다. 네 파일 모두 기존 추출 결과와 비교해 요구사항 번호·위치·`raw_req_text`가 동일함을 확인했습니다. 샘플 PDF의 `U+F09F` 154개를 치환하고 표준 글머리도 정리한 결과 `●`는 총 171개입니다. 샘플 HWP의 `MHR-003`에 있는 `U+F0A7`와 복권 HWPX의 `DAR-008`에 있는 `U+F09E`는 원문 표시를 추가 확인해야 하므로 보존하고 경고합니다. 확장자만 보고 의미를 추측해 치환하지 않습니다.

현재 파서는 **요구사항 번호와 상세내용이 있는 표**를 대상으로 합니다. 모든 RFP 서식을 지원하는 것은 아닙니다. 확보한 이스포츠 HWP는 라이브러리에서 병합 셀 겹침 오류가 발생해 이 파서로 처리하지 못합니다. HWP/HWPX의 표 번호는 페이지 번호가 아닙니다. 글머리 전처리는 추출된 텍스트를 대상으로 하므로, 문단 자동 번호 등 텍스트 밖의 서식까지 원문과 동일하게 재현하는 기능은 아닙니다.

## 실행이 잘되지 않을 때

| 증상 | 확인할 내용 |
| --- | --- |
| `No module named 'hwpx'` 또는 `No module named 'pdfplumber'` | 2단계에서 본인 터미널에 맞는 `-m pip install` 줄을 실행한 뒤, 같은 블록의 마지막 줄로 다시 실행하세요. HWP 모듈 `hwpx`의 설치 패키지 이름은 `python-hwpx`입니다. |
| `can't open file ... document_parser.py` | 터미널의 현재 위치가 프로젝트 최상위 폴더인지 확인하세요. |
| 처리하려는 문서가 로그에 표시되지 않음 | 원본 파일이 `src/data/input/` 바로 아래에 있고 확장자가 `.pdf`, `.hwp`, `.hwpx`인지 확인하세요. |
| PDF를 넣었는데 JSON이 보이지 않음 | `src/data/output/`에서 이름에 `.pdf.requirements.json`이 붙은 파일을 찾아보세요. 터미널에 `실패`가 표시되면 같은 줄의 오류 메시지를 확인하세요. |

[Python 가상환경 문서](https://docs.python.org/3/library/venv.html) · [python-hwpx 설치 안내](https://pypi.org/project/python-hwpx/7.0.1/)
