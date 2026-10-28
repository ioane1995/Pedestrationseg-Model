"""Needs torch/torchvision. Uses random weights (no downloads) on tiny synthetic data."""
import pytest

torch = pytest.importorskip("torch")

from src.data import make_splits  # noqa: E402
from src.models import build_model  # noqa: E402


def test_splits_are_disjoint_and_complete():
    tr, va, te = make_splits(170)
    assert len(tr) == 120 and len(va) == 20 and len(te) == 30
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert sorted(tr + va + te) == list(range(170))
    assert make_splits(170) == (tr, va, te)  # deterministic


@pytest.mark.parametrize("name", ["maskrcnn_r50", "maskrcnn_mobilenet"])
def test_forward_train_and_eval(name):
    model = build_model(name, 2, pretrained=False, min_size=128, max_size=256)
    img = torch.rand(3, 128, 160)
    mask = torch.zeros(1, 128, 160, dtype=torch.uint8)
    mask[0, 20:100, 40:80] = 1
    tgt = {"boxes": torch.tensor([[40., 20., 80., 100.]]), "labels": torch.tensor([1]), "masks": mask}
    model.train()
    losses = model([img], [tgt])
    assert all(torch.isfinite(v) for v in losses.values())
    model.eval()
    out = model([img])[0]
    assert set(out) >= {"boxes", "labels", "scores", "masks"}
