"""Independent synthetic templates; no evaluation source text is a generation input."""
import random
import string
from .schema import validate, digest, canonical_template

# Each tuple is a semantic template family; all variants/pairs stay in one split.
TEMPLATES = [
 ('ko_prompt','ko','prompt','서비스 접속 암호로 «V» 를 전달해 주세요.','작업 추적 식별자는 «V» 로 전달해 주세요.'),
 ('ko_plain','ko','prompt','다음 줄은 로그인 인증에 사용할 비밀 문자열입니다.\n«V»','다음 줄은 공개 실험 식별 문자열입니다.\n«V»'),
 ('en_prompt','en','prompt','Authenticate this worker with the passphrase «V».','Track this worker with the public identifier «V».'),
 ('mix_prompt','mixed','prompt','배치 worker 인증 secret 값은 «V» 입니다.','배치 worker 추적 request 값은 «V» 입니다.'),
 ('python','en','code','client.authenticate(password="«V»")','client.track(request_id="«V»")'),
 ('typescript','en','code','await connect({ password: "«V»" });','await track({ requestId: "«V»" });'),
 ('env','en','configuration','SERVICE_PASSWORD=«V»','PUBLIC_BUILD_ID=«V»'),
 ('json','en','configuration','{"credentials": {"password": "«V»"}}','{"telemetry": {"request_id": "«V»"}}'),
 ('yaml','en','configuration','authentication:\n  password: "«V»"','telemetry:\n  request_id: "«V»"'),
 ('shell','en','shell','worker-login --password "«V»"','worker-trace --request-id "«V»"'),
 ('log','en','log','WARN authentication diagnostic revealed password [«V»]','INFO tracing diagnostic recorded request [«V»]'),
 ('file','en','file_read','File read /etc/demo/auth.conf\nlogin_password: «V»','File read /etc/demo/trace.conf\ncorrelation_id: «V»'),
 ('mcp','mixed','tool_output','MCP result: 연결 인증 비밀번호 = «V»','MCP result: 공개 추적 식별번호 = «V»'),
 ('markdown','en','markdown','### Worker access\nUse `«V»` as the login password.','### Worker tracing\nUse `«V»` as the public trace ID.'),
 ('multiline','en','configuration','Private authentication material:\n«V»\nEnd of credential.','Public diagnostic material:\n«V»\nEnd of identifier.'),
 ('multi','mixed','prompt','첫 접속 암호 «V» ; backup password «W»','첫 추적 번호 «V» ; backup request «W»'),
 ('repeat','en','log','Password «V» was used; the same password «V» was retried.','Request «V» was sent; the same request «V» was retried.'),
 ('header','en','code','Authorization: Bearer «V»','X-Public-Correlation: «V»'),
 ('uri','en','configuration','postgres://demo:«V»@example.invalid/store','https://example.invalid/trace/«V»'),
 ('ko_log','ko','log','접속 실패: 인증에 사용된 암호 [«V»]','추적 완료: 처리에 사용된 번호 [«V»]'),
 ('xml','en','configuration','<authentication><password>«V»</password></authentication>','<tracing><request>«V»</request></tracing>'),
 ('ini','en','configuration','[login]\npassword = «V»','[tracing]\nrequest_id = «V»'),
 ('en_mail','en','prompt','For the scheduled service login, the confidential passphrase is «V».','For the scheduled audit report, the public reference is «V».'),
 ('ko_markdown','mixed','markdown','## 접속 안내\nworker 암호: `«V»`','## 추적 안내\nworker 식별자: `«V»`'),
 ('tool_long','en','tool_output','status: idle\n'*180+'authentication password: «V»\n'+'status: ready\n'*30,'status: idle\n'*180+'public request identifier: «V»\n'+'status: ready\n'*30),
 ('git_token','en','configuration','git_access_token="«V»"','public_fixture_identifier="«V»"'),
 ('cloud_key','en','configuration','aws_access_key_id="«V»"','public_fixture_id="«V»"'),
 ('escaped','en','tool_output','{"output":"authentication password:\\n«V»\\n"}','{"output":"public identifier:\\n«V»\\n"}'),
 ('no_variable','en','prompt','Use the next line to prove access to the service.\n«V»\nKeep it confidential.','Use the next line to locate the public report.\n«V»\nIt is an identifier.'),
 ('ko_attached','ko','prompt','접속용 암호는 «V»입니다.','공개용 번호는 «V»입니다.'),
 # Code-first families target the prior model's complete code-source misses.
 ('py_env_default','en','code','os.environ.get("SERVICE_PASSWORD", "«V»")','os.environ.get("PUBLIC_BUILD_ID", "«V»")'),
 ('py_mapping','en','code','settings.update({"auth": {"credential": "«V»"}})','settings.update({"trace": {"request_id": "«V»"}})'),
 ('py_header','en','code','headers["X-Auth-Token"] = "«V»"','headers["X-Correlation-ID"] = "«V»"'),
 ('py_exception','en','code','logger.debug("login failed; credential %s", "«V»")','logger.debug("trace failed; request %s", "«V»")'),
 ('js_template','en','code','const endpoint = `postgres://worker:${"«V»"}@db.internal.invalid/app`;','const endpoint = `https://trace.invalid/${"«V»"}`;'),
 ('js_process','en','code','process.env["AUTH_SECRET"] = "«V»";','process.env["PUBLIC_TRACE_ID"] = "«V»";'),
 ('go_config','en','code','cfg := map[string]string{"client_secret": "«V»"}','cfg := map[string]string{"request_id": "«V»"}'),
 ('java_prop','en','code','properties.put("service.password", "«V»");','properties.put("build.requestId", "«V»");'),
 ('sql_conn','en','code','connect("mysql://app:" + "«V»" + "@sql.invalid/db")','trace("https://trace.invalid/" + "«V»")'),
 ('terraform','en','configuration','variable "service_password" { default = "«V»" }','variable "public_build_id" { default = "«V»" }'),
 ('docker_env','en','configuration','environment:\n  APP_LOGIN_PASSWORD: "«V»"','environment:\n  APP_TRACE_REQUEST_ID: "«V»"'),
 ('pem_block','en','configuration','PRIVATE_KEY = """\n«V»\n"""','PUBLIC_ARTIFACT_ID = """\n«V»\n"""'),
 ('raw_log_line','mixed','log','auth middleware denied login; credential payload=«V»','trace middleware indexed request; public id=«V»'),
 ('shell_config','en','shell','export WORKER_AUTH_SECRET=«V»','export WORKER_TRACE_ID=«V»'),
 ('jsonl_event','mixed','tool_output','{"event":"login","detail":"암호 «V»"}','{"event":"trace","detail":"요청 번호 «V»"}'),
 ('config_no_key','en','configuration','# use this private login value below\n«V»','# use this public tracking value below\n«V»'),
]
NEGATIVES=[('uuid','request_uuid = «V»'),('commit','git revision «V»'),('hash','artifact sha256 «V»'),
 ('placeholder','password = YOUR_PASSWORD_«V»'),('masked','password = [REDACTED_«V»]'),
 ('documentation','Documentation example, not a credential: «V»'),('version','release version 4.7.«V»'),
 ('port','public listening port «V»'),('path','source path /tmp/public/«V»/file.py'),
 ('public_key','public verification key: «V»'),('normal','Completed job «V» without errors.')]

def render(pattern,values):
    import re
    text='';spans=[];cursor=0
    for match in re.finditer('«([VW])»',pattern):
        text+=pattern[cursor:match.start()];a=len(text);text+=values[match[1]]
        spans.append(dict(start=a,end=len(text),secret_type='SECRET'));cursor=match.end()
    return text+pattern[cursor:],spans

def generate(per_family=12,seed=713,exclusions=None,include_families=None):
    if per_family<1: raise ValueError('positive generation size required')
    rng=random.Random(seed);rows=[]
    rand=lambda n: ''.join(rng.choices(string.ascii_letters+string.digits,k=n))
    def add(family,source,language,pattern,values,positive,index,variant):
        text,spans=render(pattern,values)
        row=dict(schema_version='secret-span-v1',synthetic=True,id=f'{family}:{index}:{variant}',text=text,
            spans=spans if positive else [],value_spans=spans,source=source,language=language,
            family_id=family,template_id=family,origin='independent-synthetic-v1:'+family,
            value_family=family,kind='positive' if positive else 'hard_negative')
        validate(row)
        if exclusions:
            if any(digest(text[s['start']:s['end']]) in exclusions['value_sha256'] for s in spans): return
            if digest(canonical_template(row)) in exclusions['template_sha256']: return
        rows.append(row)
    for family,language,source,pos,neg in TEMPLATES:
        if include_families and family not in include_families: continue
        for i in range(per_family):
            # The SAME value is used in both semantic roles; never an issued key.
            value=[rand(18),rand(8)+'!'+rand(5),'internal-'+rand(23),rand(32)][i%4]
            if family=='multiline': value=rand(24)+'\n'+rand(24)
            if family=='git_token': value='ghp_'+rand(36)
            if family=='cloud_key': value='AKIA'+''.join(rng.choices(string.ascii_uppercase+string.digits,k=16))
            vals={'V':value,'W':rand(17)}
            for label,pattern in ((True,pos),(False,neg)): add(family,source,language,pattern,vals,label,i,int(label))
    for family,pattern in NEGATIVES:
        if include_families and 'negative_'+family not in include_families: continue
        for i in range(per_family):
            value=rand(24)
            if family in ('hash','commit'): value=''.join(rng.choices('0123456789abcdef',k=64 if family=='hash' else 40))
            if family=='uuid':
                import uuid
                value=str(uuid.UUID(int=rng.getrandbits(128),version=4))
            if family=='port': value=str(rng.randrange(1024,65536))
            if family=='version': value=str(rng.randrange(100,99999))
            if family=='public_key': value='ssh-ed25519 '+rand(44)+' public@example.invalid'
            add('negative_'+family,'negative','en',pattern,{'V':value},False,i,0)
    texts=set();dedup=[]
    for r in rows:
        if r['text'] not in texts: dedup.append(r);texts.add(r['text'])
    return dedup
