"""Repair audit-log, trailing-role and file-block development failures with new families."""
import json
from pathlib import Path
from privai_model.ml_span.context_data import generate_context
from privai_model.ml_span.schema import load, digest
from privai_model.ml_span.splitter import assert_disjoint
from privai_model.io import write_json, write_jsonl

FAMILIES=[
 ('v8_audit_exposure','train','log','en','Security review found leaked service password [«S»]. Public case reference [«P»].','Security review found public service-password policy reference [«S»]. Public case reference [«P»].'),
 ('v8_audit_event','train','log','en','event=credential_disclosure; confidential authentication value=«S»; public event number=«P»','event=document_lookup; public authentication guide reference=«S»; public event number=«P»'),
 ('v8_audit_ko','train','log','mixed','감사 결과: secret used for account login «S» was exposed. 공개 사건 번호 «P».','감사 결과: public account-login policy ID «S» was displayed. 공개 사건 번호 «P».'),
 ('v8_audit_report','train','tool_output','en','Finding: the value «S» grants private account access and appeared in a log. Public finding ID «P».','Finding: the value «S» identifies public account-access documentation and appeared in a log. Public finding ID «P».'),
 ('v8_suffix_description','train','tool_output','mixed','수집된 데이터 «S»\nDescription: private secret for signing in to the server. 공개 문서 참조 «P».','수집된 데이터 «S»\nDescription: public identifier for the server sign-in guide. 공개 문서 참조 «P».'),
 ('v8_suffix_caption','train','markdown','mixed','«S»\n위 문자열의 의미: confidential account access password. Public document «P».','«S»\n위 문자열의 의미: public account access documentation index. Public document «P».'),
 ('v8_suffix_usage','train','prompt','en','Received string: «S»\nHow it is used: present this confidential credential to establish a session. Public tracking value: «P».','Received string: «S»\nHow it is used: search the public session setup instructions by this ID. Public tracking value: «P».'),
 ('v8_suffix_record','train','configuration','mixed','record: «S»\nrecord_role: 로그인에 사용하는 private credential\npublic_index: «P»','record: «S»\nrecord_role: 로그인 안내서를 찾는 public reference\npublic_index: «P»'),
 ('v8_suffix_meaning','train','file_read','en','Captured value «S». It is the confidential credential needed to sign in. Public document index «P».','Captured value «S». It is the public index of instructions about signing in. Public document index «P».'),
 ('v8_suffix_two','train','tool_output','mixed','첫째 «S», 둘째 «T». Both strings are private credentials for opening account sessions. 공개 가이드 «P».','첫째 «S», 둘째 «T». Both strings are public references to account-session guides. 공개 가이드 «P».'),
 ('multiline_v8_path','train','tool_output','en','File contents at /vault/session.key:\n«S»\nThe whole file is a private login credential. Public lookup number «P».','File contents at /catalog/session.index:\n«S»\nThe whole file is a public login-guide reference. Public lookup number «P».'),
 ('multiline_v8_read','train','file_read','mixed','파일 읽기 결과:\n«S»\n위 전체 블록: confidential credential for service sign-in. 공개 안내서 «P».','파일 읽기 결과:\n«S»\n위 전체 블록: public reference for service sign-in instructions. 공개 안내서 «P».'),
 ('multiline_v8_stream','train','log','en','Retrieved object bytes:\n«S»\nObject metadata: confidential material that authenticates the client. Public object ID «P».','Retrieved object bytes:\n«S»\nObject metadata: publicly indexed password-policy document reference. Public object ID «P».'),
 ('multiline_v8_comment','train','configuration','en','material = """«S»"""\n# Entire material above is the private session credential.\npublic_record = "«P»"','material = """«S»"""\n# Entire material above is a public session manual reference.\npublic_record = "«P»"'),
 ('multiline_v8_first','train','prompt','mixed','계정 접속에 필요한 secret block:\n«S»\n끝. 공개 문서 식별자: «P»','계정 접속 안내서의 public reference block:\n«S»\n끝. 공개 문서 식별자: «P»'),
 ('multiline_v8_pair','train','markdown','en','Private session material for host one:\n«S»\nPrivate session material for host two:\n«T»\nPublic catalog item «P».','Public session documentation for host one:\n«S»\nPublic session documentation for host two:\n«T»\nPublic catalog item «P».'),
 ('v8_val_audit','validation','log','en','SECURITY FINDING: confidential login password «S» was printed; public ticket «P».','SECURITY FINDING: public login-password manual identifier «S» was printed; public ticket «P».'),
 ('v8_val_review','validation','tool_output','mixed','검토 메모 «S»: exposed private authentication credential. 공개 이슈 «P».','검토 메모 «S»: publicly available authentication policy reference. 공개 이슈 «P».'),
 ('v8_val_suffix','validation','file_read','mixed','텍스트 값 [«S»]\n해설: private password authorizing a service session. 공개 기록 [«P»].','텍스트 값 [«S»]\n해설: public identifier of a service-session manual. 공개 기록 [«P»].'),
 ('v8_val_session','validation','prompt','en','Code received («S»). Its purpose is to authenticate a private session. Public article («P»).','Code received («S»). Its purpose is to locate a public authentication article. Public article («P»).'),
 ('multiline_v8_val_blob','validation','tool_output','en','Fetched /secure/access.blob:\n«S»\nFetched bytes form a confidential sign-in credential. Public fetch reference «P».','Fetched /docs/access.blob:\n«S»\nFetched bytes form a public sign-in manual identifier. Public fetch reference «P».'),
 ('multiline_v8_val_ko','validation','markdown','mixed','출력 블록\n«S»\n블록의 용도: private account login material. 공개 설명서 «P».','출력 블록\n«S»\n블록의 용도: public account login documentation reference. 공개 설명서 «P».'),
 ('v8_test_audit','test','log','en','Incident analyst detected an exposed secret for client authentication: «S». Public incident code: «P».','Incident analyst detected a public reference for client authentication policy: «S». Public incident code: «P».'),
 ('v8_test_review','test','markdown','mixed','조사 결과 — «S» is a leaked confidential credential used to open sessions. Public report «P».','조사 결과 — «S» is a public documentation reference about opening sessions. Public report «P».'),
 ('v8_test_suffix','test','tool_output','mixed','추출 문자열: «S»\n분류 설명: confidential password required for server access. Public manual «P».','추출 문자열: «S»\n분류 설명: public catalog number of the server access guide. Public manual «P».'),
 ('v8_test_session','test','configuration','en','observed_value: «S»\nmeaning: secret credential supplied when starting an authenticated session\npublic_reference: «P»','observed_value: «S»\nmeaning: public reference supplied when finding session authentication documentation\npublic_reference: «P»'),
 ('multiline_v8_test_object','test','file_read','en','Object /credentials/client.bin was opened:\n«S»\nIts complete content authorizes private client access. Public object reference «P».','Object /manuals/client.bin was opened:\n«S»\nIts complete content identifies public client-access instructions. Public object reference «P».'),
 ('multiline_v8_test_ko','test','prompt','mixed','가져온 전체 내용\n«S»\n내용의 의미는 confidential authentication material for signing in입니다. 공개 색인 «P».','가져온 전체 내용\n«S»\n내용의 의미는 public authentication documentation identifier입니다. 공개 색인 «P».'),
]

def main():
 out=Path('data/span-context-v8')
 if out.exists():raise FileExistsError(out)
 splits,manifest=generate_context(24,20261104,json.loads(Path('reports/ml_span/public-audit-exclusions.json').read_text()),families=FAMILIES,variable_multiline=True)
 splits['train']=load(Path('data/span-context-v7/train.jsonl'))+splits['train']
 assert_disjoint(splits)
 previous=['span-synthetic-v1','span-refinement-v1','span-refinement-v3','span-context-v4','span-context-v6','span-context-v7']
 old=[r for directory in previous for split in ['validation','test'] for r in load(Path('data')/directory/(split+'.jsonl'))]
 assert_disjoint({'train':splits['train'],'historical_eval':old})
 out.mkdir()
 for name,rows in splits.items():write_jsonl(out/(name+'.jsonl'),rows)
 manifest.update(counts={k:len(v) for k,v in splits.items()},fingerprint=digest(json.dumps(splits,sort_keys=True)),replay='span-context-v7/train',historical_eval=previous,variable_multiline=True,notice='All previous tests are development only. New families assigned before training. Synthetic values only.')
 write_json(out/'manifest.json',manifest)
if __name__=='__main__':main()
