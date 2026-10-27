import torch
from privai_model.ml_span.boundary import BoundaryModel,BoundaryRefiner,encode


def test_boundary_refiner_only_contracts_and_preserves_candidate_metadata(tmp_path):
    model=BoundaryModel()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.layers[-1].bias[1]=30
    path=tmp_path/'boundary.pt';torch.save(model.state_dict(),path)
    refiner=BoundaryRefiner(path)
    spans=[dict(start=0,end=5,score=.8,detector='ml'),dict(start=6,end=7,score=.7,detector='ml')]
    result=refiner.refine('"abc" x',spans)
    assert result==[dict(spans[0],start=1,end=4),spans[1]]
    assert spans[0]['start']==0
    assert refiner.refine('',[])==[]


def test_unicode_boundary_encoding_uses_codepoint_positions():
    chars,categories,direction=encode('🔒가A',1,0)
    assert chars[11]==1+ord('🔒')%2048
    assert chars[12]==1+ord('가')%2048
    assert chars[13]==1+ord('A')%2048
    assert direction==0 and categories[12]!=categories[13]
