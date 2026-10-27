from privai_model.ml_span.training import LengthGroupedBatches


def test_length_grouping_preserves_rows_and_epoch_randomness():
    rows=[{'input_ids':[0]*n} for n in [8,512,16,256,8,512,16,256]*4]
    sampler=LengthGroupedBatches(rows,4,91)
    first=list(sampler);second=list(sampler)
    assert first==list(LengthGroupedBatches(rows,4,91))
    assert first!=second
    for batches in [first,second]:
        assert sorted(i for batch in batches for i in batch)==list(range(len(rows)))
        assert len(batches)==len(sampler)
        assert all(len({len(rows[i]['input_ids']) for i in batch})==1 for batch in batches)


def test_partial_batch_kept():
    batches=list(LengthGroupedBatches([{'input_ids':[0]*i} for i in range(1,12)],4,5))
    assert sorted(map(len,batches))==[3,4,4]
    assert sorted(i for batch in batches for i in batch)==list(range(11))
