import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import build_model
from src.unlearning.gradient_ascent import GradientAscentUnlearner


def _mean_ce_loss(model, loader):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            total_loss += F.cross_entropy(model(x), y, reduction="sum").item()
            total_n += y.size(0)
    return total_loss / total_n


def test_gradient_ascent_increases_forget_loss():
    torch.manual_seed(0)
    model = build_model("cnn", num_classes=10)

    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    forget_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

    # Fit the model to this data first, so there's meaningful loss to
    # increase — a randomly-initialized model's loss is already high
    # and near-flat, which wouldn't test anything.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(20):
        for xb, yb in forget_loader:
            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()

    loss_before = _mean_ce_loss(model, forget_loader)

    unlearner = GradientAscentUnlearner(model, config={"lr": 0.05, "epochs": 3, "device": "cpu"})
    unlearned_model = unlearner.unlearn(forget_loader)

    loss_after = _mean_ce_loss(unlearned_model, forget_loader)

    assert loss_after > loss_before, f"expected forget loss to increase, got {loss_before} -> {loss_after}"


def test_unlearn_does_not_mutate_original_model():
    torch.manual_seed(1)
    model = build_model("cnn", num_classes=10)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    forget_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    unlearner = GradientAscentUnlearner(model, config={"lr": 0.05, "epochs": 2, "device": "cpu"})
    unlearner.unlearn(forget_loader)

    for k, v in model.state_dict().items():
        assert torch.equal(v, original_state[k]), f"original model was mutated at {k}"
