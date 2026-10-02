import torch

from eegwords.model import ChiscoNet
from eegwords.train import _predict


def test_predict_restores_training_mode():
    # Regression: evaluation mid-epoch must not leave the net in eval mode (no dropout,
    # frozen batch norm) for the remaining training steps.
    net = ChiscoNet(chans=4, n_times=60)
    x = torch.randn(5, 4, 60)
    net.train()
    _predict(net, x)
    assert net.training and all(m.training for m in net.modules())
    net.eval()
    _predict(net, x)
    assert not net.training
