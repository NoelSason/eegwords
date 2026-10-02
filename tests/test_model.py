import torch
import torch.nn.functional as F

from eegwords.model import ChiscoNet, SameConv1xK


def _ref(conv: SameConv1xK, x):
    return F.conv2d(x, conv.weight, padding="same", groups=conv.groups)


def test_sameconv_matches_conv2d_dense_and_depthwise():
    torch.manual_seed(0)
    for cin, cout, k, g, T in [(1, 8, 20, 1, 1651), (64, 64, 10, 64, 825), (1, 8, 7, 1, 50)]:
        conv = SameConv1xK(cin, cout, (1, k), padding="same", groups=g, bias=False)
        x = torch.randn(2, cin, 3 if cin == 1 else 1, T)
        assert torch.allclose(conv(x), _ref(conv, x), atol=1e-4, rtol=1e-5)


def test_init_identical_to_nn_conv2d():
    torch.manual_seed(1)
    a = SameConv1xK(1, 8, (1, 20), padding="same", bias=False)
    torch.manual_seed(1)
    b = torch.nn.Conv2d(1, 8, (1, 20), padding="same", bias=False)
    assert torch.equal(a.weight, b.weight)


def test_forward_shape_both_rates():
    for T in (1651, 825):
        net = ChiscoNet(chans=122, n_times=T).eval()
        assert net(torch.randn(3, 122, T)).shape == (3, 39)
