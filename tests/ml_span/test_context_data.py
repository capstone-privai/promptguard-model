from privai_model.ml_span.context_data import generate_context, render
from privai_model.ml_span.schema import validate, digest
from privai_model.ml_span.splitter import assert_disjoint


def test_mixed_values_and_repeated_secrets_have_exact_offsets():
    text,gold,values=render('🔒«S»입니다; public=«P»; retry=«S»; backup=«T»',
        {'S':'aB9!가상','P':'sameShape44','T':'line1\nline2'},True)
    assert [text[s['start']:s['end']] for s in gold]==['aB9!가상','aB9!가상','line1\nline2']
    assert [text[s['start']:s['end']] for s in values]==['aB9!가상','sameShape44','aB9!가상','line1\nline2']
    assert render('«S» «P»',{'S':'secretShape','P':'publicShape'},False)[1]==[]


def test_context_groups_and_pairs_are_disjoint_and_balanced():
    splits,manifest=generate_context(per_family=8)
    assert_disjoint(splits)
    assert [len(manifest['families'][s]) for s in ['train','validation','test']]==[18,6,6]
    for rows in splits.values():
        assert len({r['id'] for r in rows})==len(rows)
        for positive,negative in zip(rows[::2],rows[1::2]):
            validate(positive);validate(negative)
            assert positive['spans'] and not negative['spans']
            assert positive['family_id']==negative['family_id']
            def values(row):return [row['text'][s['start']:s['end']] for s in row['value_spans']]
            assert values(positive)==values(negative)
        assert len({r['value_format'] for r in rows})==9
        assert any(len(r['spans'])>1 for r in rows)
        assert any('\n' in r['text'][s['start']:s['end']] for r in rows for s in r['spans'])


def test_public_value_exclusion_removes_entire_semantic_pair():
    original,_=generate_context(per_family=1)
    r=original['train'][0];s=r['value_spans'][0]
    excluded={'value_sha256':[digest(r['text'][s['start']:s['end']])],'template_sha256':[]}
    filtered,manifest=generate_context(per_family=1,exclusions=excluded)
    assert manifest['rejected_by_public_exclusions']==2
    assert len(filtered['train'])==len(original['train'])-2
    assert not {original['train'][0]['id'],original['train'][1]['id']} & {r['id'] for r in filtered['train']}


def test_paraphrase_changes_context_but_not_values_or_labels():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('prepare_context_v5',Path(__file__).resolve().parents[2]/'scripts/prepare_context_v5.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,_=generate_context(per_family=1)
    original=rows['train'][0]
    text,gold,values=render('password="«S»"; public request="«P»"',{'S':'passwordSECRET가상','P':'Passwordrequest'},True)
    original=dict(original,text=text,spans=gold,value_spans=values)
    updated=module.paraphrase(original,0)
    assert updated['text']=='passphrase="passwordSECRET가상"; public correlation="Passwordrequest"'
    assert [updated['text'][s['start']:s['end']] for s in updated['spans']]==['passwordSECRET가상']
    assert [updated['text'][s['start']:s['end']] for s in updated['value_spans']]==['passwordSECRET가상','Passwordrequest']
    assert updated['family_id']==original['family_id']


def test_variable_multiline_families_keep_pairs_and_split_isolation():
    families=[(f'multiline_independent_{split}',split,'file_read','en',
               'Private credential block:\n«S»\nPublic reference «P»',
               f'Public manual block ({split}):\n«S»\nPublic reference «P»')
              for split in ['train','validation','test']]
    # Different positive context is needed too: identical templates cannot cross splits.
    families=[(*f[:4],f[4]+f'\nDocument section: {f[1]}',f[5]) for f in families]
    splits,_=generate_context(16,123,families=families,variable_multiline=True)
    assert_disjoint(splits)
    lengths=set()
    for rows in splits.values():
        for pos,neg in zip(rows[::2],rows[1::2]):
            validate(pos);validate(neg)
            secrets=[pos['text'][s['start']:s['end']] for s in pos['spans']]
            assert len(secrets)==1 and not neg['spans']
            lengths.add(len(secrets[0].splitlines()))
            assert 2<=len(secrets[0].splitlines())<=5
            assert all(12<=len(line)<=48 for line in secrets[0].splitlines())
            assert [pos['text'][s['start']:s['end']] for s in pos['value_spans']]==[
                neg['text'][s['start']:s['end']] for s in neg['value_spans']]
    assert lengths=={2,3,4,5}
    # Injecting new families must not mutate default generation in the process.
    _,default=generate_context(1)
    assert len(default['families']['train'])==18
