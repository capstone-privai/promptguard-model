"""Focused audit recovery with bounded replay; no historical evaluation rows trained."""
import json
from collections import defaultdict
from pathlib import Path
from privai_model.ml_span.context_data import generate_context
from privai_model.ml_span.schema import load,digest
from privai_model.ml_span.splitter import assert_disjoint
from privai_model.io import write_json,write_jsonl

FAMILIES=[
 ('v9_audit_warning','train','log','en','Security warning: authentication secret [«S»] appeared in diagnostic output. Public investigation number [«P»].','Security warning: public authentication policy identifier [«S»] appeared in diagnostic output. Public investigation number [«P»].'),
 ('v9_audit_exposed','train','tool_output','en','Exposed authentication secret found by the auditor: «S». Public reference for the audit: «P».','Public authentication documentation identifier found by the auditor: «S». Public reference for the audit: «P».'),
 ('v9_audit_table','train','markdown','en','| Security audit field | Contents |\n| leaked authentication secret | «S» |\n| public case number | «P» |','| Security audit field | Contents |\n| public authentication policy ID | «S» |\n| public case number | «P» |'),
 ('v9_audit_event','train','log','en','audit_event(kind="secret_exposure", authentication_secret="«S»", public_case="«P»")','audit_event(kind="policy_lookup", public_authentication_policy_id="«S»", public_case="«P»")'),
 ('v9_audit_after','train','file_read','en','Evidence value: «S»\nAuditor interpretation: an exposed authentication secret. Public evidence index «P».','Evidence value: «S»\nAuditor interpretation: a public authentication policy identifier. Public evidence index «P».'),
 ('v9_audit_mixed','train','prompt','mixed','보안 감사에서 authentication secret «S»가 노출됐습니다. 공개 감사 문서 번호는 «P»입니다.','보안 감사에서 public authentication policy identifier «S»를 조회했습니다. 공개 감사 문서 번호는 «P»입니다.'),
 ('v9_audit_inline','train','log','en','[SECURITY REVIEW] leaked secret used for authentication = «S» | public audit trail = «P»','[SECURITY REVIEW] public identifier for authentication policy = «S» | public audit trail = «P»'),
 ('v9_audit_pair','train','configuration','en','exposure_report:\n  authentication_secret: «S»\n  second_authentication_secret: «T»\n  public_incident_code: «P»','documentation_report:\n  public_authentication_policy_id: «S»\n  second_public_policy_id: «T»\n  public_incident_code: «P»'),
 ('v9_val_audit_note','validation','prompt','en','The security reviewer reports a disclosed authentication secret: («S»). Public review identifier: («P»).','The security reviewer reports a public authentication-policy reference: («S»). Public review identifier: («P»).'),
 ('v9_val_audit_record','validation','log','mixed','감사 기록\nauthentication secret was disclosed: «S»\npublic audit code: «P»','감사 기록\npublic authentication policy was indexed: «S»\npublic audit code: «P»'),
 ('v9_test_audit_document','test','markdown','en','Security incident evidence\nExposed value «S» is a secret used to authenticate. Public incident record «P».','Security incident evidence\nPublished value «S» is a public authentication policy reference. Public incident record «P».'),
 ('v9_test_audit_item','test','tool_output','mixed','검사 발견사항: «S» (leaked authentication secret). Public finding reference: «P».','검사 발견사항: «S» (public authentication policy identifier). Public finding reference: «P».'),
]

def main():
 out=Path('data/span-context-v9')
 if out.exists():raise FileExistsError(out)
 splits,manifest=generate_context(24,20261105,json.loads(Path('reports/ml_span/public-audit-exclusions.json').read_text()),families=FAMILIES,variable_multiline=True)
 # Deterministically retain up to 16 rows per old family; all new pairs remain intact.
 groups=defaultdict(list)
 for r in load(Path('data/span-context-v8/train.jsonl')):groups[r['family_id']].append(r)
 replay=[r for rows in groups.values() for r in rows[:16]]
 splits['train']=replay+splits['train']
 # The v8 test has been computed by tooling but not inspected to design this run.
 for split in ['validation','test']:splits[split]=load(Path('data/span-context-v8')/(split+'.jsonl'))+splits[split]
 assert_disjoint(splits)
 old=[r for directory in ['span-synthetic-v1','span-refinement-v1','span-refinement-v3','span-context-v4','span-context-v6','span-context-v7'] for split in ['validation','test'] for r in load(Path('data')/directory/(split+'.jsonl'))]
 assert_disjoint({'train':splits['train'],'historical_eval':old})
 out.mkdir()
 for split,rows in splits.items():write_jsonl(out/(split+'.jsonl'),rows)
 manifest.update(counts={k:len(v) for k,v in splits.items()},replay_rows=len(replay),replay_families=len(groups),fingerprint=digest(json.dumps(splits,sort_keys=True)),notice='Focused synthetic audit recovery plus bounded all-family replay. Old evaluations excluded. v8 hold-out retained without using its results for this revision.')
 write_json(out/'manifest.json',manifest)
if __name__=='__main__':main()
