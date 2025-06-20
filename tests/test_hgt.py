"""Tests for the MiniHGT model temperature-scaling behaviour."""

import pytest
import torch  # noqa: E402

from python.model.hgt import MiniHGT  # noqa: E402


def test_forward_shapes():
    model = MiniHGT(in_dim=8, out_dim=1)
    x = torch.randn(4, 8)
    out_prob = model(x)
    assert out_prob.shape == (4, 1)
    assert (out_prob >= 0).all() and (out_prob <= 1).all()

    logits = model(x, return_logits=True)
    assert logits.shape == (4, 1)


def test_temperature_scaling_effect():
    model = MiniHGT(in_dim=4, out_dim=1)
    x = torch.randn(2, 4)

    # baseline probs
    p1 = model(x)

    # increase temperature (log-space param positive -> exp() > 1 -> divide by >1 -> logits/ >1 -> sigmoid closer to 0.5) 
    model.calib_temperature.data.fill_(1.0)  # exp(1)=2.718...
    p2 = model(x)

    # Expect p2 closer to 0.5 than p1 in absolute distance sense
    dist1 = torch.abs(p1 - 0.5).mean()
    dist2 = torch.abs(p2 - 0.5).mean()
    assert dist2 < dist1