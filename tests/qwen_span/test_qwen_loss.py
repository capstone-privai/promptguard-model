import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


def tiny_qwen():
    config = transformers.Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(config).eval()


def test_completion_loss_matches_masked_full_loss_and_gradients():
    from privai_model.qwen_span.modeling import completion_loss, reference_loss
    model = tiny_qwen()
    ids = torch.randint(0, 97, (1, 40))
    a = completion_loss(model, ids, 31)
    a.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    b = reference_loss(model, ids, 31)
    b.backward()
    assert a.item() == pytest.approx(b.item(), rel=1e-5)
    for g, p in zip(grads, model.parameters()):
        assert torch.allclose(g, p.grad, atol=1e-6)


def test_completion_loss_through_lora_wrapper():
    peft = pytest.importorskip('peft')
    from privai_model.qwen_span.modeling import completion_loss, reference_loss
    model = peft.get_peft_model(tiny_qwen(), peft.LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'],
                                                             task_type='CAUSAL_LM'))
    ids = torch.randint(0, 97, (1, 24))
    loss = completion_loss(model, ids, 20)
    assert loss.item() == pytest.approx(reference_loss(model, ids, 20).item(), rel=1e-5)
    loss.backward()
    assert any(p.grad is not None for n, p in model.named_parameters() if 'lora_' in n)
