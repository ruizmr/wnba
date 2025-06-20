"""Tests for CalibratedKellyLoss, ECE and BrierScore metrics."""

import math

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from python.model.losses import CalibratedKellyLoss  # noqa: E402
from python.model.metrics import BrierScore, ExpectedCalibrationError  # noqa: E402


def test_calibrated_kelly_loss_zero_lambda():
    prob = torch.tensor([[0.6], [0.4]])
    odds = torch.tensor([[1.9], [2.1]])
    loss_fn = CalibratedKellyLoss(lambda_kelly=0.0)
    loss = loss_fn(prob, odds)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_calibrated_kelly_loss_decreases_with_higher_prob():
    odds = torch.tensor([[2.0]])
    loss_fn = CalibratedKellyLoss(lambda_kelly=1.0)
    low_prob = torch.tensor([[0.2]])
    high_prob = torch.tensor([[0.6]])
    assert loss_fn(high_prob, odds) < loss_fn(low_prob, odds)


def test_brier_score_perfect_prediction():
    metric = BrierScore()
    metric.update(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]))
    assert torch.allclose(metric.compute(), torch.tensor(0.0))


def test_ece_perfect_calibration():
    metric = ExpectedCalibrationError(n_bins=10)
    # Perfectly calibrated: probabilities equal targets
    prob = torch.linspace(0.0, 1.0, steps=11)
    target = prob.clone()
    metric.update(prob, target)
    assert metric.compute() < 1e-6