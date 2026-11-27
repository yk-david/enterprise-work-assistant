"""Build reviewed, source-backed clause pairs; no model-generated paraphrases.

Original requirement blocks remain intact. New IDs explicitly denote clauses.
The labels below are assistant-reviewed annotations, pending team adjudication.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FIELDS = ['case_id', 'req_a_id', 'req_a_text', 'req_b_id', 'req_b_text',
          'label', 'reason', 'diff_conditions', 'split']
LABELS = {'중복', '유사·관련', '무관'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.pairs.open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream, strict=True))
    if len(original) != 71 or original[32]['case_id'] != 'CASE-033':
        raise ValueError('This reviewed manifest expects the original 71-case dataset.')
    backup = args.out / 'pairs_gt.original.csv'
    if backup.exists() and backup.read_bytes() != args.pairs.read_bytes():
        raise ValueError('Refusing to overwrite a different original backup.')
    backup.write_bytes(args.pairs.read_bytes())
    parents = {}
    hashes = {}
    for doc, filename in [('HWP', '제안요청서.requirements.json'),
                          ('PDF', '제안요청서.pdf.requirements.json')]:
        source = REPO / 'src/data/input' / ('제안요청서.' + doc.lower())
        hashes[doc] = digest(source)
        data = json.loads((REPO / 'src/data/output' / filename).read_text(encoding='utf-8'))
        for row in data:
            parents[doc, row['req_id']] = row

    import pdfplumber
    with pdfplumber.open(REPO / 'src/data/input/제안요청서.pdf') as pdf:
        for page in [61, 102]:
            parents['PDF', f'PAGE-{page:03}'] = {
                'req_text': pdf.pages[page-1].extract_text(),
                'doc_name': '제안요청서.pdf', 'doc_type': 'PDF',
                'source_location': f'p.{page}', 'source_req_id': '',
            }

    catalog = {}
    provenance = {}
    for row in original:
        for side in ['a', 'b']:
            rid = row[f'req_{side}_id']
            parent = parents['HWP', rid]
            assert row[f'req_{side}_text'] == parent['req_text'], rid
            catalog[rid] = {
                key: parent[key] for key in
                ['req_id', 'doc_name', 'doc_type', 'source_location', 'req_text']
            }
            provenance[rid] = dict(doc='HWP', parent_id=rid, split=row['split'],
                                  unit='requirement_block', source_sha256=hashes['HWP'],
                                  source_req_id=parent['source_req_id'])

    def span(doc, parent_id, suffix, start, end):
        parent = parents[doc, parent_id]
        text = parent['req_text']
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end-1].isspace(): end -= 1
        rid = f'{doc}-{parent_id}-{suffix}'
        if doc == 'HWP' and provenance[parent_id]['split'] != 'train':
            raise ValueError('Cannot use a held-out parent for training augmentation.')
        record = dict(req_id=rid, doc_name=parent['doc_name'], doc_type=doc,
                      source_location=parent['source_location'], req_text=text[start:end])
        if rid in catalog and catalog[rid] != record:
            raise ValueError(f'Conflicting source span: {rid}')
        catalog[rid] = record
        provenance[rid] = dict(doc=doc, parent_id=parent_id, split='train', unit='clause',
                               source_sha256=hashes[doc], source_req_id=parent['source_req_id'],
                               start=start, end=end, parent_text=text,
                               offset_basis='Python Unicode character offsets in parent_text',
                               extraction='verbatim contiguous span of parsed source')
        return rid

    def clause(doc, number, index):
        pid = f'REQ-{number:03}'
        text = parents[doc, pid]['req_text']
        starts = [m.start() for m in re.finditer(r'(?m)^[ \t]*[●○ㅇ](?=\s)', text)]
        assert starts and starts[0] == 0
        ends = starts[1:] + [len(text)]
        return span(doc, pid, f'C{index:02}', starts[index-1], ends[index-1])

    def excerpt(doc, pid, suffix, begin, stop):
        text = parents[doc, pid]['req_text']
        start = text.index(begin)
        end = text.index(stop, start) if stop else len(text)
        return span(doc, pid, suffix, start, end)

    h_manual_a, h_manual_b = clause('HWP', 11, 2), clause('HWP', 11, 5)
    consortium_a, consortium_b = clause('PDF', 9, 4), clause('PDF', 12, 7)
    pm_a = clause('PDF', 12, 2)
    pm_b = excerpt('PDF', 'REQ-013', 'C02-S01', '● 사업관리자(PM)는', '\n※')
    consortium_lock = clause('PDF', 12, 4)
    consortium_notice = excerpt('PDF', 'PAGE-061', 'NOTICE04',
                                 '○ 낙찰자로 결정된 이후에', '\n○ 사업 참여인력은')

    changes = []
    rows = [dict(r) for r in original]
    row = rows[32]
    before = dict(row)
    row.update(label='유사·관련',
               reason='두 항목은 동일한 업무보고 체계를 다루지만, 보고 기한과 참석자·산출물 등 서로 다른 세부 의무를 포함하므로 전체 요구사항을 중복으로 볼 수 없다.',
               diff_conditions='A: 착수 후 10일 이내 착수보고, 계약 후 10일 이내 계획서 제출 및 계획서 포함 항목 명시. B: 보고별 참석자·산출물과 주간 특정 요일·매월 마지막 주 특정 요일 등 세부 기준 명시.')
    changes.append(dict(case_id=row['case_id'], action='relabel', before=before, after=dict(row),
                        review_status='assistant_reviewed_pending_team_adjudication'))

    def add(a, b, label, reason, diff='없음'):
        case_id = f'CASE-{len(rows)+1:03}'
        record = dict(case_id=case_id, req_a_id=a, req_a_text=catalog[a]['req_text'],
                      req_b_id=b, req_b_text=catalog[b]['req_text'], label=label,
                      reason=reason, diff_conditions=diff, split='train')
        rows.append(record)
        changes.append(dict(case_id=case_id, action='add', after=record,
                            review_status='assistant_reviewed_pending_team_adjudication',
                            unit='clause_pair'))

    add(h_manual_a, h_manual_b, '중복',
        '동일 요구사항 표의 서로 다른 두 위치에서 시스템 장애 대비 비상매뉴얼 작성·정비 의무가 같은 문장으로 반복된다.')
    add(consortium_a, consortium_b, '중복',
        '공동수급체를 구성하는 경우 참여 사업자 사이의 업무수행 범위와 책임관계를 명확히 정하도록 요구하는 동일한 의무이다.')
    add(pm_a, pm_b, '중복',
        '동일 사업의 PM에게 인수인계·방향설정 등 유지관리 전반의 책임, 자사와 협력업체 총괄 및 창구 역할을 부여하는 문장이 반복된다. 비교 단위는 PM 역할 문장이다.')
    add(consortium_lock, consortium_notice, '중복',
        '낙찰자 결정 이후 컨소시엄 구성원을 변경하지 못한다는 시점·대상·금지행위가 같다. 공동수급이라는 괄호 설명의 유무만 다르다.')

    manual_a, manual_b = clause('PDF', 1, 6), clause('PDF', 40, 1)
    delivery_a, delivery_b = clause('PDF', 6, 4), clause('PDF', 36, 4)
    standard, structure = clause('PDF', 14, 1), clause('PDF', 15, 1)
    no_redelegate = excerpt('PDF', 'REQ-032', 'C01-S01', '- 개인정보처리시', '\n- 개인정보의 안전성')
    redelegate_exception = excerpt('PDF', 'PAGE-102', 'ARTICLE05-P01',
                                  '제5조 (재위탁 제한)', '\n② “수탁자”가')
    support_a, support_b = clause('PDF', 9, 5), clause('PDF', 9, 8)
    pm_qualified = clause('PDF', 13, 2)
    end_data, equipment = clause('PDF', 24, 5), clause('PDF', 26, 1)
    incident_a, incident_b = clause('PDF', 8, 4), clause('PDF', 11, 8)
    extension = clause('PDF', 34, 2)
    survey = clause('HWP', 12, 1)

    add(manual_a, manual_b, '유사·관련',
        '운영·유지관리 매뉴얼 작성·관리 및 요청 시 제공 의무가 겹치지만 매뉴얼화 대상 범위가 다르게 기술되어 있다.',
        'A: 운영·유지관리하면서 수행하는 사항 전반. B: 반복적으로 수행하는 사항으로 범위를 한정.')
    add(delivery_a, delivery_b, '유사·관련',
        '사업 중 취득한 정보의 파일 제출 의무가 겹치지만 제출 기한과 대상 및 차기 업체 인계 의무가 다르다.',
        'A: 사업 종료 전까지 취득·생산한 모든 정보를 발주기관 담당자에게 제출. B: 종료 2주 전까지 취득 정보를 제출하고 차기 업체 선정 시 인계.')
    add(standard, structure, '유사·관련',
        '둘 다 데이터 관리방안과 변경이력·정합성 점검을 요구하지만 관리 대상과 검증 내용이 다르다.',
        'A: 데이터 표준·표준사전과 항목의 일치 및 추가 데이터의 표준 준수. B: 데이터 구조·DB 형상과 물리데이터 모델의 일치.')
    add(no_redelegate, redelegate_exception, '유사·관련',
        '개인정보 재위탁 제한이라는 같은 행위를 다루지만 허용 예외가 달라 중복으로 합칠 수 없다.',
        'A: 권리·의무의 전부 또는 일부 재위탁 불가. B: 위탁자가 동의한 경우에는 재위탁 가능.')
    add(support_a, support_b, '유사·관련',
        '협상 전 요청 시 기술지원 방안을 제시하는 의무가 겹치지만 제공 주체, 계약서 제출 및 예외 대응 의무가 다르다.',
        'A: 협력업체·제조사의 기술지원 방안. B: 제조사 기술지원 방안과 계약 후 유지관리 수행 계약서 제출, 제출 불가 시 대응방안 포함.')
    add(pm_a, pm_qualified, '유사·관련',
        'PM 역할 문장은 같지만 B는 자격·경력 조건 충족 및 영업인원 배제라는 추가 조건까지 포함한다.',
        'A: PM 책임과 총괄·창구 역할. B: 동일 역할에 더해 위 자격·경력 조건 모두 충족, 영업인원은 PM 자격 불가.')
    add(end_data, equipment, '유사·관련',
        '장비 반출 시 자료 삭제와 담당자 승인 의무가 겹치지만 적용 시점 및 관리 대상·절차가 다르다.',
        'A: 사업 종료 후 모든 자료 반환 및 사용 장비 완전삭제·승인 후 반출. B: 사업공간 장비 반출입 전반의 승인·보안점검·대장·라벨 및 반출 시 삭제.')
    add(incident_a, incident_b, '유사·관련',
        '장애 보고와 재발방지 후속조치를 공통으로 다루지만 보고 시점·내용과 대책 제출 조건이 다르다.',
        'A: 장애·업무 문제 발생 시 즉시 보고 및 동일 장애 재발방지. B: 원인·조치·사후지원 보고서 제출, 기술지원 미흡 장애의 재발방지대책 제출.')

    add(pm_a, standard, '무관', 'PM의 조직 총괄·유지관리 책임과 데이터 표준·항목 정합성 관리는 수행 대상과 산출물이 다르다.', 'A: PM 역할. B: 데이터 표준 관리와 점검.')
    add(consortium_lock, delivery_a, '무관', '낙찰 후 컨소시엄 구성원 변경 금지와 사업 종료 전 정보 파일 제출은 서로 다른 의무이다.', 'A: 참여업체 구성 유지. B: 사업 정보 인계·제출.')
    add(h_manual_a, survey, '무관', '장애 대응 비상매뉴얼 관리와 사용자 만족도 조사는 목적과 수행 행위가 다르다.', 'A: 시스템 장애 대비. B: 사용자 만족도 조사.')
    add(manual_b, extension, '무관', '반복 운영업무 매뉴얼 작성과 차기 계약 미체결 시 계약 연장은 대상과 조건이 다르다.', 'A: 운영 매뉴얼 관리. B: 계약기간 연장.')

    ids, pairs = set(), set()
    split_by_id, split_by_parent, split_by_text = defaultdict(set), defaultdict(set), defaultdict(set)
    for record in rows:
        assert set(record) == set(FIELDS)
        assert all(isinstance(v, str) and v.strip() for v in record.values())
        assert record['label'] in LABELS
        assert record['case_id'] not in ids
        ids.add(record['case_id'])
        pair = tuple(sorted([record['req_a_id'], record['req_b_id']]))
        assert pair not in pairs and pair[0] != pair[1]
        pairs.add(pair)
        for side in ['a', 'b']:
            rid = record[f'req_{side}_id']
            assert record[f'req_{side}_text'] == catalog[rid]['req_text']
            info = provenance[rid]
            split_by_id[rid].add(record['split'])
            split_by_parent[(info['doc'], info['parent_id'])].add(record['split'])
            split_by_text[re.sub(r'\s+', '', catalog[rid]['req_text'])].add(record['split'])
            if info['unit'] == 'clause':
                assert info['parent_text'][info['start']:info['end']] == catalog[rid]['req_text']
    for groups in [split_by_id, split_by_parent, split_by_text]:
        assert all(len(splits) == 1 for splits in groups.values())
    assert [r for r in rows if r['split'] != 'train'] == [r for r in original if r['split'] != 'train']
    for before, after in zip(original, rows):
        if before['case_id'] != 'CASE-033': assert before == after

    write_csv(args.out / 'pairs_gt.csv', rows, FIELDS)
    write_csv(args.out / 'pairs_added.csv', rows[71:], FIELDS)
    write_csv(args.out / 'requirements.csv', list(catalog.values()),
              ['req_id', 'doc_name', 'doc_type', 'source_location', 'req_text'])
    summary = dict(original_rows=len(original), total_rows=len(rows), added_rows=len(rows)-len(original),
                   label_counts=dict(Counter(r['label'] for r in rows)),
                   added_label_counts=dict(Counter(r['label'] for r in rows[71:])),
                   split_counts=dict(Counter(r['split'] for r in rows)),
                   source_sha256=hashes, original_csv_sha256=digest(backup),
                   annotation_status='assistant_reviewed_pending_team_adjudication',
                   validation=dict(schema=True, unique_pairs=True, source_span_match=True,
                                   id_parent_exact_text_split_disjoint=True,
                                   original_val_test_unchanged=True),
                   limitations=['4 new duplicate pairs are clause-level, not full requirement blocks.',
                                'Two positive pairs repeat exactly the same sentence at distinct source positions.',
                                'Existing val/test contain no duplicate positives; duplicate-class quality cannot be measured there.',
                                'Original RFP-level grouping was already mixed across splits; preserved, not repaired.',
                                'No claim of complete semantic leakage detection.'])
    for name, data in [('provenance.json', provenance), ('changes.json', changes), ('validation.json', summary)]:
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    report = '\n'.join([
        '기존 RFP 학습 쌍 보강 결과', '',
        '71건 → 87건. 추가 16건: 중복 4, 유사·관련 8, 무관 4. 기존 CASE-033을 중복에서 유사·관련으로 정정.',
        '결과: 중복 4, 유사·관련 33, 무관 50. train 66 / val 6 / test 15.',
        '', '중복 근거 (PDF 페이지는 파일 기준, 인쇄 쪽번호 +2):',
        'CASE-072: HWP MPR-011 표 32의 비상매뉴얼 작성·정비 문장 2회 반복.',
        'CASE-073: PDF MPR-01 p.21 ↔ MHR-01 p.24 공동수급 업무범위·책임 구분.',
        'CASE-074: PDF MHR-01 p.24 ↔ MHR-02 p.25 PM 역할 문장 반복.',
        'CASE-075: PDF MHR-01 p.24 ↔ 입찰 유의사항 p.61 낙찰 후 컨소시엄 변경 금지.',
        '', '새 사례는 원문에 실제 존재하는 조항을 별도 ID로 추출했다. 원문 블록 전체를 중복으로 표시한 것이 아니다.',
        'PM 역할 문장만 비교하면 중복이지만, 자격 조건까지 포함하면 유사·관련이라는 경계 사례도 포함했다.',
        '원문을 바꾸거나 조건을 없애는 합성 문장, 동일 문서의 다른 파일 포맷 복제, 순서만 뒤집은 쌍은 추가하지 않았다.',
        'provenance.json에 원문 부모 본문·문자 위치·요구사항 번호·문서 SHA-256을 기록했다.',
        '새 사례는 모두 train이며, 기존 val/test는 그대로다. 부모 요구사항·동일 텍스트의 신규 분할 누수가 없음을 확인했다.',
        '원래 데이터는 같은 RFP를 요구사항별로 나눈 것이므로 문서 단위 독립 평가로 볼 수 없다.',
        '현재 val/test에는 중복 정답이 없어 중복 클래스 성능을 평가할 수 없다.',
        '라벨은 assistant가 원문을 대조하여 작성한 검토안이며 팀 합의가 완료된 인간 정답으로 주장하지 않는다.',
        '', '파일: pairs_gt.csv 전체 / pairs_added.csv 추가분 / pairs_gt.original.csv 원본 / requirements.csv 참조 목록',
        'changes.json 수정 이력 / provenance.json 출처 / validation.json 검사 결과',
    ])
    (args.out / '보강내역.txt').write_text(report + '\n', encoding='utf-8-sig')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
