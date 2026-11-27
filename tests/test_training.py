import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.train import encode_example, gold_messages, read_gold
from src.document_parser import TableBlock, parse_rfp, source_ids
from unittest.mock import patch
from scripts.compare_training_predictions import score


def sample(case, split, a, b):
    return dict(case_id=case, req_a_id=a, req_a_text='본문 '+a, req_b_id=b,
                req_b_text='본문 '+b, label='중복', reason='고유정답이유',
                diff_conditions='없음', split=split)


class FakeTokenizer:
    eos_token_id=2
    def encode(self,text,add_special_tokens=False): return list(text.encode('utf-8'))
    def apply_chat_template(self,messages,**kwargs): return json.dumps(messages,ensure_ascii=False)+'\nASSISTANT\n'


class TrainingTests(unittest.TestCase):
    def test_spreadsheet_error_body_is_rejected_before_training(self):
        rows=[sample('C1','train','A1','B1'),sample('C2','val','A2','B2'),sample('C3','test','A3','B3')]
        rows[0]['req_a_text']='#NAME?'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'pairs.csv'
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'스프레드시트 오류'):
                read_gold(path)

    def test_missing_prediction_remains_in_metric_denominator(self):
        a=sample('C1','test','A','B');b=sample('C2','test','C','D')
        pred={**a,'pred_label':'중복','status':'ok'}
        result=score([a,b],[pred])
        self.assertEqual(result['accuracy'],0.5)
        self.assertEqual(result['errors'],['C2'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['confusion_matrix']['중복']['ERROR'],1)

    def test_only_completion_is_supervised(self):
        row=sample('C','train','A','B')
        messages=gold_messages(row)
        self.assertNotIn('고유정답이유',json.dumps(messages[:-1],ensure_ascii=False))
        encoded=encode_example(FakeTokenizer(),row,20000)
        n=encoded['prompt_tokens']
        self.assertEqual(encoded['labels'][:n],[-100]*n)
        self.assertEqual(encoded['labels'][n:],encoded['input_ids'][n:])
        self.assertEqual(encoded['labels'][-1],2)
        with self.assertRaisesRegex(ValueError,'not truncated'):
            encode_example(FakeTokenizer(),row,10)

    def test_parent_leakage_rejected_even_with_distinct_clause_ids(self):
        rows=[sample('C1','train','A1','B1'),sample('C2','val','A2','B2'),sample('C3','test','A3','B3')]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'pairs.csv'
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            self.assertEqual(len(read_gold(path)),3)
            p={rid:dict(doc='DOC',parent_id=rid) for row in rows for rid in [row['req_a_id'],row['req_b_id']]}
            p['A1']['parent_id']=p['A2']['parent_id']='SAME_PARENT'
            prov=Path(temp)/'provenance.json';prov.write_text(json.dumps(p),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'crosses splits'):read_gold(path,prov)

    def test_merged_hwp_labels_and_hwpx_definition_row(self):
        rows=[['요구사항 고유번호','요구사항 고유번호','DAR-001'],
              ['요구사항 상세내용','정의','정의 설명'],
              ['요구사항 상세내용','세부내용','실제 본문']]
        self.assertEqual(source_ids(rows),{'DAR-001'})
        self.assertEqual(source_ids([['요구사항 번호','요구사항 명(또는 요구사항 내용 요약)']]),set())
        with patch('src.document_parser.read_hwp',return_value=[TableBlock('표 1',rows)]):
            parsed=parse_rfp('sample.hwpx')
        self.assertEqual(parsed[0]['req_text'],'실제 본문')
        self.assertEqual(parsed[0]['doc_type'],'HWPX')
        with self.assertRaises(ValueError):source_ids([['요구사항번호','DAR-broken']])


if __name__=='__main__':unittest.main()
