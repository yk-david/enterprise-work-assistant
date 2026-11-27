"""Compare base/adapter predictions without dropping missing or invalid cases."""
import argparse
import csv
import json
from pathlib import Path

LABELS = ['중복','유사·관련','무관']


def score(gold, predictions):
    by_id={}
    for row in predictions:
        if row['case_id'] in by_id: raise ValueError('Duplicate prediction case ID')
        by_id[row['case_id']]=row
    gold_ids={r['case_id'] for r in gold}
    if set(by_id)-gold_ids: raise ValueError('Unexpected prediction case ID')
    matrix={label:{p:0 for p in LABELS+['ERROR']} for label in LABELS}
    errors=[]
    for row in gold:
        pred=by_id.get(row['case_id'])
        if pred:
            for key in ['req_a_id','req_b_id','req_a_text','req_b_text']:
                if pred.get(key)!=row[key]: raise ValueError(f"Prediction source mismatch: {row['case_id']} {key}")
        label=pred.get('pred_label') if pred else None
        if pred is None or pred.get('status')!='ok' or label not in LABELS:
            errors.append(row['case_id']);label='ERROR'
        matrix[row['label']][label]+=1
    per_class={}
    for label in LABELS:
        tp=matrix[label][label]
        fp=sum(matrix[g][label] for g in LABELS if g!=label)
        fn=sum(n for p,n in matrix[label].items() if p!=label)
        per_class[label]=dict(support=sum(matrix[label].values()),precision=tp/(tp+fp) if tp+fp else 0,
                              recall=tp/(tp+fn) if tp+fn else 0, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(total=len(gold),valid=len(gold)-len(errors),errors=errors,
                complete=not errors,accuracy=sum(matrix[l][l] for l in LABELS)/len(gold),
                macro_f1=sum(v['f1'] for v in per_class.values())/3,
                per_class=per_class,confusion_matrix=matrix)


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream,strict=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--adapted',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    gold=[r for r in read_csv(args.gold) if r['split']=='test']
    if not gold:raise ValueError('No test data')
    result=dict(base=score(gold,read_csv(args.base)),adapted=score(gold,read_csv(args.adapted)),
                annotation_status='assistant_reviewed_pending_team_adjudication',
                limitations='18 selected test pairs; only 1 duplicate and 2 related examples; not a document-held-out or human-adjudicated benchmark.')
    result['macro_f1_delta']=result['adapted']['macro_f1']-result['base']['macro_f1']
    result['accuracy_delta']=result['adapted']['accuracy']-result['base']['accuracy']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['base']['complete'] and result['adapted']['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
