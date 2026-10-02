"""Entirely fabricated fixtures. MASK means simulated role, never a live credential."""
import random
import string


def hard_cases(long_context=False):
    rng = random.Random(173)
    token = lambda n: ''.join(rng.choices(string.ascii_letters+string.digits, k=n))
    keep = [
        ('placeholder','your-api-key-here','API_KEY="{}"','API_KEY'),
        ('documentation','AKIAIOSFODNN7EXAMPLE','Example access key: {}','AWS_KEY'),
        ('uuid','550e8400-e29b-41d4-a716-446655440000','request_id={}','UUID'),
        ('commit','0123456789abcdef'*2+'01234567','git checkout {}','SHA'),
        ('hash','a1b2c3d4'*16,'sha512={}','HASH'),
        ('integrity','sha512-'+token(86)+'==','"integrity": "{}"','INTEGRITY'),
        ('public-key','pk_test_SYNTHETIC_PUBLIC_ONLY_'+token(16),'public_client_key="{}"','PUBLIC_KEY'),
        ('test-password','test-password-not-real','테스트 전용 비밀번호: {}','PASSWORD'),
        ('random-id',token(48),'non_secret_correlation_id="{}"','ID'),
    ]
    mask = [
        ('api-key','sk-SYNTHETIC'+token(28),'export API_KEY="{}"','API_KEY'),
        ('access-token','ghp_'+token(36),'access_token="{}" # synthetic only','ACCESS_TOKEN'),
        ('password','Synthetic!'+token(14),'DB 비밀번호는 {} 입니다.','PASSWORD'),
        ('connection','Synthetic!'+token(12),'postgres://demo:{}@example.invalid/db','PASSWORD'),
        ('bearer','synthetic.'+token(40),'curl -H "Authorization: Bearer {}" https://example.invalid','TOKEN'),
        ('cli-password','Synthetic!'+token(12),'mysql -u root -p{}','PASSWORD'),
        ('cloud','AKIA'+''.join(rng.choices(string.ascii_uppercase+string.digits,k=16)),'aws_access_key_id={}','AWS_KEY'),
        ('unicode','가상비밀값!123Ab','password="{}" # 절대 실제 자격 증명 아님','PASSWORD'),
        ('minimum','x','password="{}" # synthetic minimum-length case','PASSWORD'),
    ]
    rows=[]
    for index, pair in enumerate(zip(keep,mask)):
        for label, (kind,value,template,typ) in enumerate(pair):
            line = template.format(value)
            start = line.index(value)
            prefix = ('unrelated synthetic context words only\n'*2048 if long_context else 'synthetic fixture only\n')
            rows.append({'schema_version':'candidate-v1','candidate_id':f'synthetic:{kind}',
                'group_id':f'synthetic-pair:{index}', 'granularity':'value','value':value,
                'line':{'text':line,'value_start':start,'value_end':start+len(value)},
                'context':{'before':prefix,'after':prefix},'type':typ,
                'source':{'dataset':'hard-case-synthetic-only','agent':'fixture','channel':'tool_output',
                          'tool_name':'shell','command':'synthetic fixture','file_path':'fixtures/example.env'},
                'detector':{'name':'synthetic','rule_id':kind,'raw_type':typ,'score':0.5},
                'label':{'action':'MASK' if label else 'KEEP','provisional':True,
                         'reason':'Fabricated role label for plumbing only, not a real credential.'}})
    return rows
