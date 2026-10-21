import pytest
import torch

from models.listener import Listener
from models.speaker import Speaker


@pytest.mark.parametrize("mode", ["gumbel", "reinforce"])
@pytest.mark.parametrize("training", [True, False])
def test_speaker_output_shape_and_range(mode, training):
    torch.manual_seed(0)
    speaker = Speaker(vocab_size=7, msg_len=5, mode=mode).train(training)
    msg = speaker(torch.rand(6, 3, 64, 64))
    assert msg.symbols.shape == (6, 5)
    assert msg.symbols.dtype == torch.long
    assert ((msg.symbols >= 0) & (msg.symbols < 7)).all()
    assert msg.onehots.shape == (6, 5, 7)
    # The forward pass is truly discrete: each one-hot has exactly one 1.
    assert torch.allclose(msg.onehots.sum(-1), torch.ones(6, 5))
    assert torch.equal(msg.onehots.argmax(-1), msg.symbols)
    assert msg.log_probs.shape == (6,) and msg.entropy.shape == (6,)


def test_listener_scores_each_candidate():
    listener = Listener(vocab_size=7)
    onehots = torch.nn.functional.one_hot(torch.randint(0, 7, (6, 5)), 7).float()
    assert listener(onehots, torch.rand(6, 4, 3, 64, 64)).shape == (6, 4)


def test_gumbel_gradient_reaches_speaker_but_reinforce_does_not():
    torch.manual_seed(0)
    images = torch.rand(4, 3, 64, 64)
    for mode, expect_grad in [("gumbel", True), ("reinforce", False)]:
        speaker, listener = Speaker(mode=mode), Listener()
        logits = listener(speaker(images).onehots, images.unsqueeze(1).repeat(1, 3, 1, 1, 1))
        logits.sum().backward()
        grad = speaker.out.weight.grad
        assert (grad is not None and grad.abs().sum() > 0) == expect_grad


def test_speaker_and_listener_do_not_share_encoders():
    speaker, listener = Speaker(), Listener()
    assert speaker.encoder is not listener.encoder
    speaker_params = {id(p) for p in speaker.parameters()}
    assert not any(id(p) in speaker_params for p in listener.parameters())
