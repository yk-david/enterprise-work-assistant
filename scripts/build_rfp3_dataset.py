"""Extract the three supplied RFPs and build a reviewed pilot training dataset."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.document_parser import parse_rfp
from scripts.augment_existing_rfps import FIELDS, write_csv

ROOT = REPO.parent
OUT = REPO / 'src/data/output/datasets/rfp3_training'
SOURCES = [
    ROOT / '20211218201-00_1639123687988_★ 4. 제안요청서.hwp',
    ROOT / '복권_제안요청서.hwpx',
    ROOT / '최종_2026 이스포츠 국가대표 훈련 지원 용역대행 제안요청서.hwp',
]


def esports_paragraphs(path):
    # The original has overlapping table spans. Read native paragraph records;
    # do not silently repair the table or assert visual/page ordering.
    from hwpx.hwp5.reader import read_hwp5
    from hwpx.hwp5.bodytext import split_text
    from hwpx.hwp5.records import PARA_TEXT
    doc = read_hwp5(path.read_bytes())
    result = []
    for section_no, section in enumerate(doc.sections):
        for record_no, record in enumerate(section.records):
            if record.tag != PARA_TEXT:
                continue
            chunks, malformed = split_text(record.payload)
            if malformed:
                raise ValueError('Malformed HWP text controls; source review required.')
            text = ''.join(c.text if c.kind == 'text' else '\n' if c.code == 10
                           else '\t' if c.code == 9 else '' for c in chunks).strip()
            if text:
                result.append(dict(paragraph=len(result)+1, section=section_no,
                                   record_index=record_no, text=text))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = REPO / 'src/data/output/datasets/existing_rfp_augmented'
    rows = list(csv.DictReader((base / 'pairs_gt.csv').open(encoding='utf-8-sig', newline='')))
    catalog = {r['req_id']: r for r in csv.DictReader((base / 'requirements.csv').open(encoding='utf-8-sig', newline=''))}
    provenance = json.loads((base / 'provenance.json').read_text(encoding='utf-8'))
    original_len = len(rows)
    clauses, source_info = {}, []
    for number, path in enumerate(SOURCES, 1):
        key = f'RFP{number:02}'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        info = dict(doc_id=key, path=str(path), sha256=sha)
        if number == 3:
            esports = esports_paragraphs(path)
            info.update(extraction='native HWP paragraph records; table conversion failed',
                        paragraph_count=len(esports))
            (OUT / f'{key}.paragraphs.json').write_text(json.dumps(esports, ensure_ascii=False, indent=2), encoding='utf-8')
            esp_by_num = {p['paragraph']: p for p in esports}
        else:
            reqs = parse_rfp(path)
            info.update(extraction='requirement tables', requirement_count=len(reqs))
            (OUT / f'{key}.requirements.json').write_text(json.dumps(reqs, ensure_ascii=False, indent=2), encoding='utf-8')
            for req in reqs:
                text = req['req_text']
                starts = [m.start() for m in re.finditer(r'(?m)^[ \t]*[ㅇ○◦●❍-](?=\s)', text)]
                if not starts or starts[0] != 0: starts.insert(0, 0)
                top = [m.start() for m in re.finditer(r'(?m)^[ \t]*[ㅇ○◦●❍](?=\s)', text)]
                if top: starts = ([0] if top[0] > 0 else []) + top
                for index, (a,b) in enumerate(zip(starts, starts[1:] + [len(text)]), 1):
                    while a < b and text[a].isspace(): a += 1
                    while b > a and text[b-1].isspace(): b -= 1
                    rid = f"{key}-{req['source_req_id']}-C{index:02}"
                    clauses[rid] = dict(req_id=rid, doc_name=path.name, doc_type=path.suffix[1:].upper(),
                                        source_location=req['source_location'], req_text=text[a:b])
                    provenance[rid] = dict(doc=key, parent_id=req['source_req_id'], unit='clause',
                                           source_sha256=sha, source_req_id=req['source_req_id'],
                                           start=a, end=b, parent_text=text)
        source_info.append(info)

    def esp(number, context=None, through=None):
        indices = list(range(number, (through or number)+1))
        if context is not None: indices.insert(0, context)
        raw = [esp_by_num[i] for i in indices]
        rid = f'RFP03-P{number:03}' + (f'-CTX{context:03}' if context else '')
        text = '\n'.join(p['text'] for p in raw)
        clauses[rid] = dict(req_id=rid, doc_name=SOURCES[2].name, doc_type='HWP',
                            source_location='HWP 원문 문단 ' + ', '.join(map(str, indices)), req_text=text)
        provenance[rid] = dict(doc='RFP03', parent_id=f'P{number:03}', unit='clause_with_context' if context else 'clause',
                               source_sha256=source_info[2]['sha256'], source_req_id='',
                               original_paragraphs=raw,
                               context_note='Only original heading and complete paragraph text; no paraphrase.')
        return rid

    additions = []
    def add(a, b, label, reason, diff='없음', split='train'):
        for rid in [a,b]:
            catalog[rid] = clauses[rid]
            previous = provenance[rid].get('split')
            if previous is not None and previous != split: raise ValueError(rid)
            provenance[rid]['split'] = split
        record = dict(case_id=f'CASE-{len(rows)+1:03}', req_a_id=a, req_a_text=catalog[a]['req_text'],
                      req_b_id=b, req_b_text=catalog[b]['req_text'], label=label, reason=reason,
                      diff_conditions=diff, split=split)
        rows.append(record)
        additions.append(dict(case_id=record['case_id'], label=label, split=split,
                              review_status='assistant_reviewed_pending_team_adjudication'))

    # Public meals RFP. Keep all workplace/personnel parents in test.
    add('RFP01-MHR-003-C03', 'RFP01-PSR-006-C01', '중복',
        '효율성과 정보보안 조건을 고려하여 발주기관과 계약대상자가 작업장소를 협의 결정한다는 문장이 동일하게 반복된다.', split='test')
    add('RFP01-MHR-002-C01', 'RFP01-MHR-003-C03', '유사·관련',
        '작업장소 협의 결정 의무는 같지만 A는 보안 등 특별한 사유가 없는 경우 별도 장소를 정할 수 있다는 추가 조건을 포함한다.',
        'A: 특별한 사유가 없으면 상호협의로 따로 정할 수 있음. B: 효율성·보안을 고려한 협의 결정만 명시.', 'test')
    add('RFP01-MHR-003-C03', 'RFP01-MAR-005-C01', '무관',
        '작업장소 결정과 홍보사이트 메인화면 이미지 디자인 관리는 수행 대상과 행위가 다르다.',
        'A: 작업장소 협의. B: 이미지 디자인 콘텐츠 관리.', 'test')
    add('RFP01-MAR-008-C03', 'RFP01-PSR-001-C03', '중복',
        '시스템 구축·신규참여 및 운영에 필요하여 발주기관이 관련 교육을 요청하면 본 사업 수행 범위 안에서 적극 지원한다는 의무가 같다.')
    add('RFP01-TER-001-C01', 'RFP01-TER-001-C02', '유사·관련',
        '테스트 수행이라는 목적은 같지만 테스트 종류와 환경, 적용 과업이 다르다.',
        'A: 기능개선 단위테스트를 개발 환경에서 수행. B: 기능개선·신규참여 통합테스트를 운영 환경에서 수행.')
    add('RFP01-PSR-003-C01', 'RFP01-PSR-003-C02', '유사·관련',
        '변경 내용을 매뉴얼에 반영하는 의무는 같지만 이용자와 매뉴얼의 내용 범위가 다르다.',
        'A: 사용자매뉴얼의 기능 변경. B: 시스템관리자 운영에 필요한 운영자매뉴얼 변경.')
    add('RFP01-QUR-002-C01', 'RFP01-COR-008-C01', '유사·관련',
        '데이터베이스 표준화 지침을 공통으로 참조하지만 계획·이행과 설계 표준 적용이라는 산출물이 다르다.',
        'A: 행정DB 데이터 품질관리계획 수립·이행. B: DB 설계 시 표준용어·코드 적용과 서울시 지침 준용.')
    add('RFP01-PMR-001-C04', 'RFP01-PMR-001-C05', '유사·관련',
        '성과지표에 따른 측정결과 제출을 다루지만 보고 시점과 발생 조건이 다르다.',
        'A: 사업 중 발주기관 요청 시 중간점검용 결과서. B: 사업 종료 시 최종 성과측정결과서.')
    add('RFP01-MAR-008-C03', 'RFP01-COR-008-C01', '무관',
        '발주기관 요청 교육 지원과 데이터베이스 설계의 용어·코드 표준 적용은 서로 다른 업무이다.',
        'A: 교육 지원. B: DB 설계 표준 준수.')
    add('RFP01-PSR-003-C01', 'RFP01-PMR-001-C05', '무관',
        '사용자매뉴얼 현행화와 계약 종료 시 성과지표 측정결과 보고는 대상과 산출물이 다르다.',
        'A: 기능 변경의 매뉴얼 반영. B: 최종 사업 성과측정 보고.')

    # Lottery RFP. Workplace clauses and package procurement are validation-only.
    add('RFP02-SER-005-C01', 'RFP02-PSR-001-C02', '중복',
        '보안요구사항을 지킨 작업장소를 사업자가 제시하면 발주기관이 우선 검토하되, 보안 미준수 장소는 거부할 수 있다는 조건까지 같다.', split='val')
    add('RFP02-SER-005-C05', 'RFP02-PSR-001-C04', '유사·관련',
        '서초 수탁사 사무실에서 지정·허가 PC로만 자료를 열람한다는 내용은 같지만 A에는 수탁사 보안규정 준수 의무가 더 있다.',
        'A: 열람 장소·PC 제한에 수탁사 보안규정 준수 추가. B: 열람 장소·PC 제한만 명시.', 'val')
    add('RFP02-SER-005-C01', 'RFP02-CSR-016-C01', '무관',
        '사업장소의 보안조건 심사와 요구사항을 충족하는 SW패키지의 존재·분리발주 조사에는 직접적인 수행 의무의 겹침이 없다.',
        'A: 보안 작업장소 검토. B: SW패키지 및 추가 개발 범위 분석.', 'val')
    add('RFP02-PMR-001-C01', 'RFP02-PMR-009-C01', '중복',
        '계약을 수행하는 업체가 사업추진·품질보증 등 전체 사업수행의 권한과 책임을 가진다는 동일한 의무이다.')
    add('RFP02-PMR-003-C03', 'RFP02-PMR-009-C01', '유사·관련',
        '전체 사업수행 책임을 규정하지만 책임 주체를 공동수급체 대표사업자로 한정하는지에 차이가 있다.',
        'A: 공동수급체 대표사업자. B: 제안업체.')
    add('RFP02-DAR-002-C01', 'RFP02-DAR-005-C01', '유사·관련',
        '현행 데이터 관리체계를 분석하고 목표 방안을 제시하는 업무이나 관리 대상이 다르다.',
        'A: 데이터 표준관리와 표준화 매뉴얼 참조. B: 데이터 구조관리.')
    add('RFP02-DAR-004-C01', 'RFP02-DAR-006-C01', '유사·관련',
        '반복되는 데이터 문제의 현황분석·개선을 요구하지만 구조 결함과 데이터 값 오류는 구분해야 한다.',
        'A: 데이터 구조로 인한 반복 문제. B: 데이터 값 진단 내역과 반복 오류.')
    add('RFP02-SER-004-C03', 'RFP02-SER-008-C02', '유사·관련',
        '비인가자의 산출물·기록 접근 금지 의무는 같지만 인가 권한 주체가 더 구체적으로 제한되어 있다.',
        'A: 발주기관 인가. B: 발주기관 보안관리 담당공무원 인가.')
    add('RFP02-PMR-001-C01', 'RFP02-DAR-004-C01', '무관',
        '사업수행 권한·책임 귀속과 데이터 구조 문제 분석·개선은 직접적인 요구 행위가 다르다.',
        'A: 계약상대자의 총괄 책임. B: 데이터 구조 개선.')
    add('RFP02-COR-004-C01', 'RFP02-DAR-006-C01', '무관',
        '지도상의 동해·독도 표기 점검과 데이터 값 오류 진단은 서로 다른 점검 대상과 절차이다.',
        'A: 지명 표기 확인·보고·수정. B: 데이터 값 오류 현황·개선.')

    # Esports RFP: keep the original event headings, which change meaning.
    settlement_a, settlement_b = esp(29), esp(185)
    add(settlement_a, settlement_b, '중복', '사업비를 국고보조금 정산 기준에 맞추어 정산하는 업무 일체를 수행한다는 동일한 요구이다.')
    venue_ag, venue_eca = esp(152,149), esp(165,160)
    add(venue_ag, venue_eca, '유사·관련',
        '장소·일정·소요예산 제안 문장은 같지만 원문 상위 제목의 대상 대회가 달라 별도 수행 범위를 가진다.',
        'A: 2026 아시안게임 국내 훈련 지원. B: 2026 아시아 이스포츠대회 국내 훈련 지원.')
    scrim_domestic, scrim_abroad = esp(172,171), esp(175,174)
    add(scrim_domestic, scrim_abroad, '유사·관련',
        '필요시 연습경기 일정 지원은 같지만 훈련·파견 범위와 환경 구축의 근거가 다르다.',
        'A: 아시안게임·ECA 국내 합숙훈련과 협회 요청. B: 국외 훈련·대회 파견과 협회 가이드.')
    report_write, report_submit = esp(28), esp(184)
    add(report_write, report_submit, '유사·관련',
        '동일 훈련지원 결과보고서를 다루지만 작성과 용역 종료에 따른 제출은 다른 단계의 의무이다.',
        'A: 결과보고서 작성. B: 용역 종료 시 결과보고서 제출.')
    handler_summary, handler_detail = esp(39), esp(167,166)
    add(handler_summary, handler_detail, '유사·관련',
        '전담 핸들러를 통한 숙소·훈련장 이동 지원이 겹치지만 A는 식음 관리도 맡기고 B는 이동 지원을 명시한다.',
        'A: 이동과 식음 관리 전담 핸들러. B: 훈련기간 이동 지원을 위한 핸들러 배치.')
    safety, meals = esp(27), esp(182,180)
    add(settlement_a, handler_detail, '무관', '국고보조금 정산과 선수단 이동 지원은 회계 처리와 현장 운영이라는 별개의 업무이다.', 'A: 사업비 정산. B: 전담 핸들러와 이동 지원.')
    add(report_write, venue_ag, '무관', '결과보고서 작성과 외부 훈련장소·일정·예산 제안은 수행 단계와 산출물이 다르다.', 'A: 훈련 결과보고서. B: 훈련장소 대관 제안.')
    add(safety, meals, '무관', '선수단이라는 대상은 같지만 안전·보안 및 불편사항 대응과 식사 제공 계획은 별개의 수행 의무이다.', 'A: 안전·보안 및 불편사항 접수·대처. B: 선수 컨디션 유지를 위한 식사 계획.')

    # Validate sources, parent groups, canonical text, and pair identities.
    seen_ids, seen_pairs = set(), set()
    groups, texts = defaultdict(set), defaultdict(set)
    for row in rows:
        assert set(row) == set(FIELDS) and all(str(v).strip() for v in row.values())
        assert row['label'] in {'중복', '유사·관련', '무관'}
        assert row['case_id'] not in seen_ids
        seen_ids.add(row['case_id'])
        pair = tuple(sorted([row['req_a_id'],row['req_b_id']]))
        assert pair[0] != pair[1] and pair not in seen_pairs
        seen_pairs.add(pair)
        for side in ['a','b']:
            rid = row[f'req_{side}_id']; p = provenance[rid]
            assert row[f'req_{side}_text'] == catalog[rid]['req_text']
            groups[p['doc'],p['parent_id']].add(row['split'])
            texts[re.sub(r'[^가-힣A-Za-z0-9]', '', catalog[rid]['req_text'])].add(row['split'])
            if p['doc'].startswith('RFP') and 'parent_text' in p:
                assert p['parent_text'][p['start']:p['end']] == catalog[rid]['req_text']
    assert all(len(v)==1 for v in groups.values()), 'parent leakage'
    assert all(len(v)==1 for v in texts.values()), 'canonical text leakage'
    provenance = {rid:provenance[rid] for rid in catalog}
    stats = dict(total=len(rows), added=len(additions), label_counts=dict(Counter(r['label'] for r in rows)),
                 split_labels={s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ['train','val','test']},
                 sources=source_info, annotation_status='assistant_reviewed_pending_team_adjudication',
                 validation='PASS: source spans, IDs, unique pairs, parent/text splits',
                 evaluation_note='Requirement-parent split, not document-held-out; new validation/test each have only one duplicate positive.')
    write_csv(OUT/'pairs_gt.csv', rows, FIELDS)
    write_csv(OUT/'pairs_added_3rfps.csv', rows[original_len:], FIELDS)
    write_csv(OUT/'requirements.csv', list(catalog.values()), ['req_id','doc_name','doc_type','source_location','req_text'])
    for name,data in [('provenance.json',provenance),('sources.json',source_info),('annotations.json',additions),('validation.json',stats)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
