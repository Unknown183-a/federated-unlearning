import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import build_model
from src.unlearning.knowledge_distillation import KnowledgeDistiller


def _accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def test_distillation_preserves_remaining_accuracy():
    torch.manual_seed(0)

    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    remaining_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

    # Teacher stands in for M_old: fit it to the remaining-client data first.
    teacher = build_model("cnn", num_classes=10)
    optimizer = torch.optim.SGD(teacher.parameters(), lr=0.1)
    for _ in range(20):
        for xb, yb in remaining_loader:
            optimizer.zero_grad()
            F.cross_entropy(teacher(xb), yb).backward()
            optimizer.step()
    teacher_acc = _accuracy(teacher, remaining_loader)
    assert teacher_acc > 0.5, "teacher didn't actually learn the data; test setup is invalid"

    # Student stands in for a model that needs its remaining-client
    # knowledge (re)built via distillation, e.g. after gradient ascent
    # has damaged it. A fresh, untrained model is the worst case.
    student = build_model("cnn", num_classes=10)
    student_acc_before = _accuracy(student, remaining_loader)

    distiller = KnowledgeDistiller(
        teacher=teacher,
        student=student,
        config={"lr": 0.1, "epochs": 30, "temperature": 2.0, "lambda_kd": 1.0, "device": "cpu"},
    )
    distilled_student = distiller.distill(remaining_loader)
    student_acc_after = _accuracy(distilled_student, remaining_loader)

    assert student_acc_after > student_acc_before, "distillation should improve accuracy, not leave it unchanged"
    # "Doesn't collapse" per Definition of Done: should land reasonably
    # close to the teacher, not just marginally above the random baseline.
    assert student_acc_after >= teacher_acc - 0.3, (
        f"student ({student_acc_after}) fell too far below teacher ({teacher_acc})"
    )


def test_distill_does_not_mutate_original_student_or_teacher():
    torch.manual_seed(1)
    teacher = build_model("cnn", num_classes=10)
    student = build_model("cnn", num_classes=10)
    teacher_state = {k: v.clone() for k, v in teacher.state_dict().items()}
    student_state = {k: v.clone() for k, v in student.state_dict().items()}

    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    remaining_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    distiller = KnowledgeDistiller(
        teacher=teacher, student=student, config={"lr": 0.1, "epochs": 2, "temperature": 2.0, "device": "cpu"}
    )
    distiller.distill(remaining_loader)

    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_state[k]), f"teacher was mutated at {k}"
    for k, v in student.state_dict().items():
        assert torch.equal(v, student_state[k]), f"original student was mutated at {k}"
