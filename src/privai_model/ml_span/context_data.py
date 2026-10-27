"""Independent challenge families with mixed public/secret fields and varied values.

Assignments are fixed before any model evaluation. Patterns are authored here;
public benchmark text is never an input. Both semantic roles use the same values.
"""
import base64
import json
import random
import re
import string
from .schema import validate, digest, canonical_template

# name, split, source, language, positive pattern, negative pattern
FAMILIES = [
 ('http_client','train','code','en','request.headers.update({"Authorization": "Bearer «S»", "X-Request-ID": "«P»"})','request.headers.update({"X-Public-Reference": "«S»", "X-Request-ID": "«P»"})'),
 ('curl_header','train','shell','en','curl --header "Authorization: Bearer «S»" --header "X-Trace: «P»" https://example.invalid','curl --header "X-Public-ID: «S»" --header "X-Trace: «P»" https://example.invalid'),
 ('proxy_auth','train','log','en','proxy debug: outgoing authentication bearer token [«S»]; trace [«P»]','proxy debug: outgoing public record identifier [«S»]; trace [«P»]'),
 ('ko_auth_help','train','prompt','ko','연결 인증에 쓰는 토큰은 «S»이고 공개 요청 번호는 «P»입니다.','공개 검색에 쓰는 번호는 «S»이고 공개 요청 번호는 «P»입니다.'),
 ('grpc_auth','train','code','en','metadata = [("authorization", "Bearer «S»"), ("request-id", "«P»")]','metadata = [("public-reference", "«S»"), ("request-id", "«P»")]'),
 ('rust_header','train','code','en','client.bearer_auth("«S»").header("X-Trace", "«P»");','client.header("X-Public-ID", "«S»").header("X-Trace", "«P»");'),
 ('ko_rotation','train','markdown','mixed','로그인 secret 교체: 이전 값 `«S»`, 새 값 `«T»`. 공개 tracking ID: `«P»`.','공개 문서 번호 교체: 이전 값 `«S»`, 새 값 `«T»`. 공개 tracking ID: `«P»`.'),
 ('two_services','train','configuration','en','services:\n  a: {password: "«S»"}\n  b: {password: "«T»"}\npublic_id: "«P»"','services:\n  a: {public_id: "«S»"}\n  b: {public_id: "«T»"}\npublic_id: "«P»"'),
 ('audit_pair','train','log','en','request «P»: password «S» rejected; fallback password «T» accepted','request «P»: public revision «S» rejected; fallback public revision «T» accepted'),
 ('sql_mixed','train','code','en','INSERT INTO credentials (public_id, login_secret) VALUES (\'«P»\', \'«S»\');','INSERT INTO documents (public_id, public_revision) VALUES (\'«P»\', \'«S»\');'),
 ('unicode_access','train','prompt','mixed','🔒 로그인 비밀값: «S»\r\n📎 공개 식별자: «P»','🔎 공개 검색값: «S»\r\n📎 공개 식별자: «P»'),
 ('tool_credentials','train','tool_output','en','result={"public_id":"«P»","auth":{"secret":"«S»","backup":"«T»"}}','result={"public_id":"«P»","catalog":{"public_id":"«S»","revision":"«T»"}}'),
 ('multiline_note','train','file_read','en','Private login material follows:\n«S»\nEnd private material. Public request: «P»','Public diagnostic material follows:\n«S»\nEnd public material. Public request: «P»'),
 ('long_worker','train','log','en','worker trace «P»\nAuthentication credential: «S»\nReady for next task.','worker trace «P»\nPublic correlation identifier: «S»\nReady for next task.'),
 ('numeric_access','train','configuration','en','private_service_pin="«S»"\npublic_reference="«P»"','public_service_reference="«S»"\npublic_reference="«P»"'),
 ('password_comment','train','code','en','# Secret for authenticating the worker\nvalue = "«S»"\npublic_id = "«P»"','# Public identifier for the password reset documentation\nvalue = "«S»"\npublic_id = "«P»"'),
 ('auth_catalog','train','prompt','en','The authentication secret to submit is «S». The public documentation ID is «P».','The public ID of the authentication-secret documentation is «S». The public documentation ID is «P».'),
 ('repeat_login','train','log','mixed','인증 실패 credential=«S»; 재시도 credential=«S»; public trace=«P»','조회 실패 public reference=«S»; 재시도 public reference=«S»; public trace=«P»'),
 ('fetch_metadata','validation','code','en','fetch(url, {headers: {Authorization: "Bearer «S»", "X-Request-ID": "«P»"}});','fetch(url, {headers: {"X-Public-Identifier": "«S»", "X-Request-ID": "«P»"}});'),
 ('ko_sealed_note','validation','prompt','ko','인증창에 넣을 비밀 문자열은「«S»」입니다. 공개 문서 식별자는「«P»」입니다.','검색창에 넣을 공개 문자열은「«S»」입니다. 공개 문서 식별자는「«P»」입니다.'),
 ('multiline_config','validation','configuration','en','private_material = """\n«S»\n"""\npublic_reference = "«P»"','public_diagnostic = """\n«S»\n"""\npublic_reference = "«P»"'),
 ('long_monitor','validation','tool_output','en','monitor output: private credential [«S»]; public tracking [«P»]','monitor output: public build reference [«S»]; public tracking [«P»]'),
 ('backup_access','validation','shell','en','connect --secret "«S»" --backup-secret "«T»" --request "«P»"','report --public-id "«S»" --public-revision "«T»" --request "«P»"'),
 ('public_password_docs','validation','markdown','en','## Login details\nCredential to authenticate: `«S»`. Public manual ID: `«P»`.','## Login details\nPublic password-manual reference: `«S»`. Public manual ID: `«P»`.'),
 ('http_transcript','test','file_read','en','POST /session HTTP/1.1\r\nHost: example.invalid\r\nauthorization: Bearer «S»\r\nx-request-id: «P»\r\n','GET /public HTTP/1.1\r\nHost: example.invalid\r\nx-public-reference: «S»\r\nx-request-id: «P»\r\n'),
 ('ko_pair_instructions','test','prompt','mixed','Primary 로그인 암호는 [«S»], secondary 로그인 암호는 [«T»]. 공개 보고서 번호는 [«P»].','Primary 공개 자료 번호는 [«S»], secondary 공개 자료 번호는 [«T»]. 공개 보고서 번호는 [«P»].'),
 ('multiline_tool','test','tool_output','en','Read result /private/login.material\n«S»\nThis is confidential authentication material. Public trace: «P»','Read result /public/diagnostic.material\n«S»\nThis is public diagnostic material. Public trace: «P»'),
 ('long_support','test','markdown','en','## Access handoff\nUse `«S»` for confidential service authentication. Public ticket: `«P»`.','## Access handoff\nUse `«S»` to locate the public password-policy article. Public ticket: `«P»`.'),
 ('db_two_logins','test','configuration','en','[replicas]\nprimary_login_secret="«S»"\nsecondary_login_secret="«T»"\npublic_build="«P»"','[replicas]\nprimary_public_reference="«S»"\nsecondary_public_reference="«T»"\npublic_build="«P»"'),
 ('public_auth_audit','test','log','en','AUDIT security: exposed authentication secret «S»; public incident identifier «P»','AUDIT security: public authentication policy identifier «S»; public incident identifier «P»'),
]


def render(pattern, values, positive):
    text=''; valuespans=[]; gold=[]; cursor=0
    for match in re.finditer('«([SPT])»',pattern):
        text+=pattern[cursor:match.start()]
        start=len(text);text+=values[match[1]]
        span=dict(start=start,end=len(text),secret_type='SECRET')
        valuespans.append(span)
        if positive and match[1] in ('S','T'): gold.append(span.copy())
        cursor=match.end()
    return text+pattern[cursor:],gold,valuespans


def generate_context(per_family=24, seed=20261029, exclusions=None):
    if per_family<1: raise ValueError('positive generation size required')
    rng=random.Random(seed)
    rand=lambda n: ''.join(rng.choices(string.ascii_letters+string.digits,k=n))
    def value(kind):
        if kind=='alnum': return rand(rng.randrange(16,41))
        if kind=='punctuated': return rand(8)+'!'+rand(7)+'@'+rand(5)
        if kind=='hex': return ''.join(rng.choices('0123456789abcdef',k=40))
        if kind=='base64': return base64.b64encode(rng.getrandbits(224).to_bytes(28,'big')).decode()
        if kind=='url_safe': return base64.urlsafe_b64encode(rng.getrandbits(240).to_bytes(30,'big')).decode().rstrip('=')
        if kind=='dotted': return rand(10)+'.'+rand(16)+'.'+rand(12)
        if kind=='numeric': return ''.join(rng.choices(string.digits,k=16))
        if kind=='unicode': return rand(8)+'가상'+rand(10)+'!'
        if kind=='multiline': return rand(28)+'\n'+rand(28)+'\n'+rand(28)
        raise ValueError(kind)
    kinds=['alnum','punctuated','hex','base64','url_safe','dotted','numeric','unicode']
    splits={s:[] for s in ('train','validation','test')}
    rejected=0
    for family,split,source,language,pos,neg in FAMILIES:
        for i in range(per_family):
            kind='multiline' if family.startswith('multiline_') else kinds[i%len(kinds)]
            if family=='numeric_access': kind='numeric'
            vals={k:value(kind) for k in ('S','P','T')}
            pair=[]
            for positive,pattern in [(True,pos),(False,neg)]:
                if family.startswith('long_'):
                    # Diverse positions: the final quarter exceeds one 512-token window.
                    n=[8,48,96,144][i%4]
                    padding=''.join(f'status step {j}: operation complete\n' for j in range(n))
                    pattern=padding+pattern+'\n'+padding[:len(padding)//4]
                text,spans,value_spans=render(pattern,vals,positive)
                row=dict(schema_version='secret-span-v1',synthetic=True,id=f'context4:{family}:{i}:{int(positive)}',text=text,
                    spans=spans,value_spans=value_spans,source=source,language=language,
                    family_id='context4_'+family,template_id='context4_'+family,origin='independent-context4:'+family,
                    value_family='context4_'+family,kind='positive' if positive else 'hard_negative',value_format=kind)
                validate(row);pair.append(row)
            if exclusions and any(
                digest(canonical_template(r)) in exclusions['template_sha256'] or any(
                digest(r['text'][s['start']:s['end']]) in exclusions['value_sha256'] for s in r['value_spans']) for r in pair):
                rejected+=len(pair);continue
            splits[split].extend(pair)
    return splits,dict(seed=seed,per_family=per_family,rejected_by_public_exclusions=rejected,
        families={s:[f[0] for f in FAMILIES if f[1]==s] for s in splits})
