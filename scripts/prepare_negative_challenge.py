"""Fresh negative-only stress evaluation; never used for training or model selection."""
import random
import string
from pathlib import Path
from privai_model.io import write_jsonl,write_json
from privai_model.ml_span.schema import validate

PATTERNS=[
 ('env_reference','code','client = Client(api_key=os.environ["SERVICE_API_KEY"]); request_id="«P»"'),
 ('shell_reference','shell','curl -H "Authorization: Bearer ${SERVICE_TOKEN}" -H "X-Request-ID: «P»" https://example.invalid'),
 ('masked','log','auth debug: password=[REDACTED]; bearer_token=********; public trace=«P»'),
 ('empty','configuration','password=""\napi_key=null\npublic_reference="«P»"'),
 ('schema','code','class Config: password: str; access_token: str\n# public documentation identifier «P»'),
 ('policy','configuration','password_min_length=24\npassword_required=true\npublic_policy_revision="«P»"'),
 ('template','configuration','client_secret="<INSERT_SECRET_HERE>"\npublic_build="«P»"'),
 ('vault_reference','configuration','password_from="vault://services/worker/password"\npublic_request="«P»"'),
 ('runtime_lookup','code','auth = getSecret("worker-password")\nrequest_id = "«P»"'),
 ('ko_policy','prompt','비밀번호 자체는 전달하지 않았습니다. 인증 정책의 공개 문서 식별번호는 «P»입니다.'),
 ('hash_comment','code','# Password reset guide public artifact digest\nartifact_sha256="«P»"'),
 ('public_path','log','Loaded public credential-format documentation from /docs/public/«P»/guide.md'),
]

def main():
    output=Path('data/span-negative-challenge-v1')
    if output.exists():raise FileExistsError(output)
    rng=random.Random(7717);rows=[]
    for family,source,pattern in PATTERNS:
        for i in range(8):
            value=''.join(rng.choices('0123456789abcdef' if family=='hash_comment' else string.ascii_letters+string.digits,k=64 if family=='hash_comment' else 24))
            a=pattern.index('«P»');text=pattern.replace('«P»',value)
            row=dict(schema_version='secret-span-v1',synthetic=True,id=f'negative_challenge:{family}:{i}',text=text,
                spans=[],value_spans=[dict(start=a,end=a+len(value),secret_type='SECRET')],source=source,
                language='ko' if family=='ko_policy' else 'en',family_id='negative_challenge_'+family,
                template_id='negative_challenge_'+family,origin='independent-negative-challenge:'+family,
                value_family='negative_challenge_'+family,kind='hard_negative',value_format='noncredential_reference')
            validate(row);rows.append(row)
    output.mkdir();write_jsonl(output/'test.jsonl',rows)
    write_json(output/'manifest.json',dict(seed=7717,records=len(rows),families=len(PATTERNS),gold_spans=0,
        role='Unseen negative-only stress evaluation. Excluded from training, threshold selection and acceptance gate. Placeholders/references are not literal credentials.'))
if __name__=='__main__':main()
