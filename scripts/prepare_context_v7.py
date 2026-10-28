"""Fresh semantic families; historical evaluations never enter training."""
import json
from pathlib import Path
from privai_model.ml_span.context_data import generate_context
from privai_model.ml_span.schema import load, digest
from privai_model.ml_span.splitter import assert_disjoint
from privai_model.io import write_json, write_jsonl

# Paired roles deliberately share values; split assignments precede training.
FAMILIES = [
 ('multiline_v7_packet','train','tool_output','en','Packet contents:\n«S»\nPurpose of the entire packet: authenticate the private worker. Catalog reference «P».','Packet contents:\n«S»\nPurpose of the entire packet: identify the public password manual. Catalog reference «P».'),
 ('multiline_v7_envelope','train','markdown','en','Envelope begins\n«S»\nEnvelope ends\nKeep this login credential confidential. Reference «P».','Envelope begins\n«S»\nEnvelope ends\nThis is a publicly listed authentication document reference. Reference «P».'),
 ('multiline_v7_ko','train','prompt','ko','전달 내용:\n«S»\n위 내용 전체를 서버 로그인 비밀값으로 사용합니다. 공개 문서 번호: «P»','전달 내용:\n«S»\n위 내용 전체는 누구나 볼 수 있는 비밀번호 정책 문서 식별자입니다. 공개 문서 번호: «P»'),
 ('multiline_v7_before','train','configuration','en','confidential_login_blob: |\n«S»\npublic_document_blob: |\n«P»','public_login_manual_blob: |\n«S»\npublic_document_blob: |\n«P»'),
 ('multiline_v7_reply','train','file_read','mixed','다음 블록은 private authentication credential입니다.\n«S»\n블록 종료. Public catalog: «P»','다음 블록은 public authentication guide identifier입니다.\n«S»\n블록 종료. Public catalog: «P»'),
 ('multiline_v7_rotation','train','log','en','Retired private login block:\n«S»\nReplacement private login block:\n«T»\nPublic reference: «P»','Retired public manual block:\n«S»\nReplacement public manual block:\n«T»\nPublic reference: «P»'),
 ('v7_suffix_record','train','tool_output','en','Record «S»\nRole: secret presented to the service for authentication.\nPublic document «P»','Record «S»\nRole: public index entry for the authentication instructions.\nPublic document «P»'),
 ('v7_suffix_ko','train','log','ko','수신 문자열 «S». 용도는 비공개 서비스 로그인 인증입니다. 공개 색인 번호 «P».','수신 문자열 «S». 용도는 공개 로그인 설명서 검색입니다. 공개 색인 번호 «P».'),
 ('v7_caption','train','markdown','en','`«S»`\nCaption: confidential service access token.\nPublic manual: `«P»`.','`«S»`\nCaption: publicly searchable password manual reference.\nPublic manual: `«P»`.'),
 ('v7_assignment','train','code','en','payload = "«S»"  # confidential password sent during login\npublic_doc = "«P»"','payload = "«S»"  # public identifier used to search password documentation\npublic_doc = "«P»"'),
 ('v7_use','train','prompt','en','Supply «S» as the private password when opening the service session. The manual is indexed under «P».','Supply «S» as the public search reference when opening the password manual. The manual is indexed under «P».'),
 ('v7_mixed','train','prompt','mixed','값 «S»: private login credential로 제출. 값 «P»: 공개 로그인 도움말 검색용.','값 «S»: public login guide ID로 검색. 값 «P»: 공개 로그인 도움말 검색용.'),
 ('v7_xml','train','configuration','en','<entry role="private-login-credential">«S»</entry><public-manual>«P»</public-manual>','<entry role="public-login-manual">«S»</entry><public-manual>«P»</public-manual>'),
 ('v7_reverse','train','file_read','en','Public authentication manual «P»\n«S»\nThe last line is the secret used for service login.','Public authentication manual «P»\n«S»\nThe last line is another public identifier for password documentation.'),
 ('v7_two','train','tool_output','en','Account access: «S» and «T». Both are confidential login passwords. Public guide: «P».','Document access: «S» and «T». Both are public password-policy identifiers. Public guide: «P».'),
 ('v7_header','train','code','en','headers.set("Authorization", "Bearer «S»"); // private access\n// public manual: «P»','headers.set("X-Login-Documentation", "«S»"); // public reference\n// public manual: «P»'),
 ('multiline_v7_val_transport','validation','log','en','Transported block:\n«S»\nClassification of transported block: private login credential. Public manual «P».','Transported block:\n«S»\nClassification of transported block: public login documentation identifier. Public manual «P».'),
 ('v7_val_suffix','validation','markdown','en','Item: [«S»]\nInterpretation: password granting private access to the service. Public record [«P»].','Item: [«S»]\nInterpretation: public record number of the password guide. Public record [«P»].'),
 ('v7_val_two','validation','configuration','mixed','first="«S»" # 비공개 로그인 암호\nsecond="«T»" # 비공개 로그인 암호\nmanual="«P»" # 공개','first="«S»" # 공개 로그인 문서 번호\nsecond="«T»" # 공개 로그인 문서 번호\nmanual="«P»" # 공개'),
 ('v7_val_docs','validation','prompt','en','The string («S») must be kept secret because it authenticates service requests. Public password guide («P»).','The string («S») is a public reference used to find the password instructions. Public password guide («P»).'),
 ('multiline_v7_test_archive','test','file_read','en','Archived payload follows\n«S»\nPayload purpose: confidential material for signing into the account. Public archive identifier «P».','Archived payload follows\n«S»\nPayload purpose: public reference for the account authentication handbook. Public archive identifier «P».'),
 ('v7_test_suffix','test','tool_output','mixed','항목 [«S»]\n설명: confidential credential for service sign-in. 공개 도움말 키 [«P»].','항목 [«S»]\n설명: public reference for service sign-in documentation. 공개 도움말 키 [«P»].'),
 ('v7_test_pair','test','markdown','en','Credential inventory: «S»; «T». These private values authorize account access. Public policy index: «P».','Documentation inventory: «S»; «T». These public values identify account-access manuals. Public policy index: «P».'),
 ('v7_test_lookup','test','prompt','ko','«S»를 서비스에 비밀 인증값으로 전달하세요. 공개 인증 안내서 색인은 «P»입니다.','«S»로 공개 인증 안내서를 검색하세요. 공개 인증 안내서 색인은 «P»입니다.'),
]

def main():
 out=Path('data/span-context-v7')
 if out.exists(): raise FileExistsError(out)
 splits,manifest=generate_context(32,20261103,json.loads(Path('reports/ml_span/public-audit-exclusions.json').read_text()),families=FAMILIES,variable_multiline=True)
 splits['train']=load(Path('data/span-context-v6/train.jsonl'))+splits['train']
 assert_disjoint(splits)
 old=[]
 for directory in ['span-synthetic-v1','span-refinement-v1','span-refinement-v3','span-context-v4','span-context-v6']:
  for name in ['validation','test']:old.extend(load(Path('data')/directory/(name+'.jsonl')))
 assert_disjoint({'train':splits['train'],'historical_eval':old})
 out.mkdir()
 for name,rows in splits.items():write_jsonl(out/(name+'.jsonl'),rows)
 manifest.update(counts={k:len(v) for k,v in splits.items()},fingerprint=digest(json.dumps(splits,sort_keys=True)),notice='Synthetic only. Previous tests are development. Independent family assignments frozen before training.')
 write_json(out/'manifest.json',manifest)
if __name__=='__main__':main()
