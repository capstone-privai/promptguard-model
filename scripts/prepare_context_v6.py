"""Independent purpose-after-value and multiline challenge; prior tests remain development."""
import json
from pathlib import Path
from privai_model.ml_span.context_data import FAMILIES,generate_context
from privai_model.ml_span.schema import load,digest
from privai_model.ml_span.splitter import assert_disjoint
from privai_model.io import write_json,write_jsonl

NEW=[
 ('multiline_training_a','train','prompt','en','«S»\nThe block above is a private credential used to authenticate. Public ticket: «P».','«S»\nThe block above is a public authentication-policy document identifier. Public ticket: «P».'),
 ('multiline_training_b','train','configuration','en','login_private_material: |\n«S»\npublic_ticket: «P»','public_authentication_document: |\n«S»\npublic_ticket: «P»'),
 ('multiline_training_c','train','tool_output','mixed','비밀 인증 자료 시작\n«S»\n비밀 인증 자료 끝\n공개 요청 번호 «P»','공개 인증 문서 시작\n«S»\n공개 인증 문서 끝\n공개 요청 번호 «P»'),
 ('purpose_training_a','train','log','en','value [«S»] is the confidential login credential; public request [«P»]','value [«S»] is the public login-policy reference; public request [«P»]'),
 ('purpose_training_b','train','markdown','en','| private login credential | «S» |\n| public authentication policy ID | «P» |','| public login guide identifier | «S» |\n| public authentication policy ID | «P» |'),
 ('bearer_training_a','train','file_read','en','HTTP request dump:\nauthorization: bearer «S»\nx-trace: «P»\nEnd request.','HTTP request dump:\nx-public-auth-policy-id: «S»\nx-trace: «P»\nEnd request.'),
 ('bearer_training_b','train','prompt','en','Header name: Authorization\nHeader value: Bearer «S»\nPublic request: «P»','Header name: X-Public-Auth-Policy-ID\nHeader value: «S»\nPublic request: «P»'),
 ('purpose_training_c','train','prompt','ko','«S»는 로그인에 쓰는 비밀값입니다. «P»는 인증 정책의 공개 문서 번호입니다.','«S»는 로그인 정책의 공개 식별번호입니다. «P»는 인증 정책의 공개 문서 번호입니다.'),
 ('multiline_validate_a','validation','file_read','en','Credential block retrieved:\n«S»\nUse this private block to log in. Public ID: «P»','Documentation block retrieved:\n«S»\nUse this public block to locate the authentication guide. Public ID: «P»'),
 ('purpose_validate_a','validation','log','en','entry=«S»; classification=private authentication credential; trace=«P»','entry=«S»; classification=public authentication document identifier; trace=«P»'),
 ('bearer_validate_a','validation','configuration','en','upstream_headers:\n  Authorization: "Bearer «S»"\n  X-Public-ID: "«P»"','upstream_headers:\n  X-Authentication-Manual-ID: "«S»"\n  X-Public-ID: "«P»"'),
 ('purpose_validate_b','validation','prompt','ko','비공개 로그인 값 [«S»]과 공개 인증 설명서 번호 [«P»]를 확인했습니다.','공개 로그인 설명서 번호 [«S»]와 공개 인증 설명서 번호 [«P»]를 확인했습니다.'),
 ('multiline_sealed_a','test','markdown','en','### Login material\n«S»\nThe preceding block grants private service access. Public reference «P».','### Authentication documentation\n«S»\nThe preceding block identifies a public authentication manual. Public reference «P».'),
 ('purpose_sealed_a','test','tool_output','en','Lookup result [«S»]. Meaning: confidential value supplied for login. Public audit ID [«P»].','Lookup result [«S»]. Meaning: public identifier of a login guide. Public audit ID [«P»].'),
 ('bearer_sealed_a','test','code','en','outgoing = {"authorization": "Bearer «S»", "public-trace": "«P»"}','outgoing = {"public-auth-manual": "«S»", "public-trace": "«P»"}'),
 ('purpose_sealed_b','test','prompt','mixed','[«S»] — service 접속용 비밀 문자열. [«P»] — 공개 인증 문서 식별자.','[«S»] — service 인증 안내서 공개 식별자. [«P»] — 공개 인증 문서 식별자.'),
]
def main():
 out=Path('data/span-context-v6')
 if out.exists():raise FileExistsError(out)
 FAMILIES[:]=NEW
 splits,manifest=generate_context(24,20261102,json.loads(Path('reports/ml_span/public-audit-exclusions.json').read_text()))
 # The generator prefix remains context4 but each family name is new and unique.
 splits['train']=load(Path('data/span-context-v5/train.jsonl'))+splits['train']
 assert_disjoint(splits)
 old=[]
 for directory in ['span-synthetic-v1','span-refinement-v1','span-refinement-v3','span-context-v4']:
  for name in ['validation','test']:old.extend(load(Path('data')/directory/(name+'.jsonl')))
 assert_disjoint({'train':splits['train'],'historical_eval':old})
 out.mkdir()
 for name,rows in splits.items():write_jsonl(out/(name+'.jsonl'),rows)
 manifest.update(counts={k:len(v) for k,v in splits.items()},fingerprint=digest(json.dumps(splits,sort_keys=True)),notice='Earlier tests are development only; new families assigned before training. Synthetic only.')
 write_json(out/'manifest.json',manifest)
if __name__=='__main__':main()
