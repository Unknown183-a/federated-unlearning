import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import build_model
from src.unlearning.engine import UnlearningEngine, UnlearningResult


def _accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def _mean_ce_loss(model, loader):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            total_loss += F.cross_entropy(model(x), y, reduction="sum").item()
            total_n += y.size(0)
    return total_loss / total_n


def _make_model_and_loaders():
    torch.manual_seed(0)
    model = build_model("cnn", num_classes=10)

    xf = torch.randn(32, 1, 28, 28)
    yf = torch.randint(0, 10, (32,))
    forget_loader = DataLoader(TensorDataset(xf, yf), batch_size=8, shuffle=True)

    xr = torch.randn(64, 1, 28, 28)
    yr = torch.randint(0, 10, (64,))
    remaining_loader = DataLoader(TensorDataset(xr, yr), batch_size=8, shuffle=True)

    # Fit the model on BOTH forget and remaining data first, so M_old
    # actually knows both — otherwise there's nothing meaningful to
    # forget or preserve.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(20):
        for loader in (forget_loader, remaining_loader):
            for x, y in loader:
                optimizer.zero_grad()
                F.cross_entropy(model(x), y).backward()
                optimizer.step()

    return model, forget_loader, remaining_loader


def test_engine_returns_unlearning_result():
    m_old, forget_loader, remaining_loader = _make_model_and_loaders()
    engine = UnlearningEngine(
        m_old,
        config={"lr": 0.05, "epochs": 5, "lambda_forget": 1.0, "lambda_kd": 1.0, "temperature": 2.0, "device": "cpu"},
    )
    result = engine.run(forget_client_id=0, forget_loader=forget_loader, remaining_loader=remaining_loader)

    assert isinstance(result, UnlearningResult)
    assert result.metrics["forget_client_id"] == 0
    assert len(result.metrics["history"]) == 5


def test_joint_engine_forgets_without_collapsing_remaining_accuracy():
    """The key property Phase 06 showed plain sequential GA->KD fails at:
    forgetting and preservation happening AT THE SAME TIME, not one
    undoing the other.
    """
    m_old, forget_loader, remaining_loader = _make_model_and_loaders()

    forget_loss_before = _mean_ce_loss(m_old, forget_loader)
    remaining_acc_before = _accuracy(m_old, remaining_loader)
    assert remaining_acc_before > 0.5, "m_old didn't actually learn the remaining data; test setup is invalid"

    engine = UnlearningEngine(
        m_old,
        config={"lr": 0.05, "epochs": 10, "lambda_forget": 1.0, "lambda_kd": 1.0, "temperature": 2.0, "device": "cpu"},
    )
    result = engine.run(forget_client_id=0, forget_loader=forget_loader, remaining_loader=remaining_loader)
    m_unlearn = result.model

    forget_loss_after = _mean_ce_loss(m_unlearn, forget_loader)
    remaining_acc_after = _accuracy(m_unlearn, remaining_loader)

    assert forget_loss_after > forget_loss_before, (
        f"expected forget loss to increase, got {forget_loss_before} -> {forget_loss_after}"
    )
    assert remaining_acc_after >= remaining_acc_before - 0.3, (
        f"remaining-client accuracy collapsed: {remaining_acc_before} -> {remaining_acc_after}"
    )


def test_engine_does_not_mutate_m_old():
    m_old, forget_loader, remaining_loader = _make_model_and_loaders()
    original_state = {k: v.clone() for k, v in m_old.state_dict().items()}

    engine = UnlearningEngine(
        m_old, config={"lr": 0.05, "epochs": 2, "lambda_forget": 1.0, "lambda_kd": 1.0, "device": "cpu"}
    )
    engine.run(forget_client_id=0, forget_loader=forget_loader, remaining_loader=remaining_loader)

    for k, v in m_old.state_dict().items():
        assert torch.equal(v, original_state[k]), f"m_old was mutated at {k}"
