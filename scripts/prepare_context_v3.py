"""Fresh context families; preserve all previous evaluation families outside training."""
import json
from pathlib import Path
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.generator import TEMPLATES, generate
from privai_model.ml_span.schema import load
from privai_model.ml_span.splitter import assert_disjoint

# Fixed before training: eight training, four validation, four final evaluation families.
FAMILIES = [
 ('rust_auth','en','code','session.login("worker", "«V»");','session.find_public_record("worker", "«V»");'),
 ('ruby_auth','en','code','connection.authenticate(user: "demo", secret: "«V»")','connection.trace(user: "demo", public_id: "«V»")'),
 ('ko_handoff','ko','prompt','다음 담당자가 로그인할 때 입력할 비밀값을 전달합니다: «V»','다음 담당자가 공개 문서를 검색할 때 입력할 번호를 전달합니다: «V»'),
 ('curl_auth','en','shell','curl -H "X-API-Key: «V»" https://api.example.invalid','curl -H "X-Request-ID: «V»" https://api.example.invalid'),
 ('toml_auth','en','configuration','[worker.authentication]\nsecret = "«V»"','[worker.telemetry]\npublic_id = "«V»"'),
 ('log_rotation','mixed','log','인증 정보 교체 완료; new login secret [«V»]; 이전 값 폐기','추적 정보 교체 완료; new public reference [«V»]; 이전 값 폐기'),
 ('two_accounts','en','prompt','Primary login passphrase: «V»\nSecondary login passphrase: «W»','Primary public record: «V»\nSecondary public record: «W»'),
 ('tool_nested','en','tool_output','Tool result: {"account":{"authentication":{"secret":"«V»"}}}','Tool result: {"account":{"telemetry":{"reference":"«V»"}}}'),
 ('php_auth','en','code','$client->authenticate("demo", "«V»");','$client->lookupPublicRecord("demo", "«V»");'),
 ('ko_ticket','mixed','prompt','운영 인계: 로그인 비밀값은 아래와 같습니다.\n«V»\n인증 시 사용하세요.','운영 인계: 공개 문서 번호는 아래와 같습니다.\n«V»\n검색 시 사용하세요.'),
 ('properties_auth','en','configuration','worker.auth.passphrase: «V»','worker.tracing.publicReference: «V»'),
 ('audit_auth','en','log','[audit] confidential authentication material exposed: «V»','[audit] public correlation material recorded: «V»'),
 ('csharp_auth','en','code','await session.AuthenticateAsync("demo", "«V»");','await session.FindPublicRecordAsync("demo", "«V»");'),
 ('ko_runbook','mixed','markdown','### 운영 절차\n로그인 화면의 비밀 입력란에 `«V»`를 붙여 넣으세요.','### 운영 절차\n공개 자료 검색란에 `«V»`를 붙여 넣으세요.'),
 ('hcl_auth','en','configuration','service "worker" { authentication_secret = "«V»" }','service "worker" { public_reference = "«V»" }'),
 ('event_auth','en','tool_output','Event output\nclassification: confidential login credential\nvalue: «V»\nend','Event output\nclassification: public tracking identifier\nvalue: «V»\nend'),
]

def main():
    output=Path('data/span-refinement-v3')
    if output.exists(): raise FileExistsError(output)
    assert not {x[0] for x in FAMILIES} & {x[0] for x in TEMPLATES}
    TEMPLATES.extend(FAMILIES)
    rows=generate(32,20261028,json.loads(Path('reports/ml_span/public-audit-exclusions.json').read_text()),{x[0] for x in FAMILIES})
    names={'train':{x[0] for x in FAMILIES[:8]},'validation':{x[0] for x in FAMILIES[8:12]},'test':{x[0] for x in FAMILIES[12:]}}
    splits={k:[r for r in rows if r['family_id'] in v] for k,v in names.items()}
    splits['train']=load(Path('data/span-refinement-v1/train.jsonl'))+splits['train']
    assert_disjoint(splits)
    output.mkdir()
    for name,items in splits.items(): write_jsonl(output/(name+'.jsonl'),items)
    write_json(output/'manifest.json',dict(seed=20261028,new_families=names_to_lists(names),counts={k:len(v) for k,v in splits.items()},notice='Previous train only replay; previous validation/test excluded. Fixed new families assigned before training. Synthetic data only.'))

def names_to_lists(names): return {k:sorted(v) for k,v in names.items()}
if __name__=='__main__': main()
