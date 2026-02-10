import torch
import torch.nn as nn
from typing import List, Optional

class Distiller(nn.Module):
    """
    Knowledge Distillation wrapper for PyTorch models.
    """

    def __init__(self,
                 student: nn.Module,
                 teacher: nn.Module,
                 alpha: float = 0.5,
                 temperature: float = 4.0,
                 name: Optional[str] = None):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.requires_grad_(False)

        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.student(x)

    def train_step(self, x, y_true, optimizer):
        optimizer.zero_grad()
        y_student = self.student(x)
        y_teacher = self.teacher(x)

        student_loss = self.student_loss_fn(y_student, y_true)

        T = self.temperature
        p_teacher_t = torch.nn.functional.softmax(y_teacher / T, dim=-1)
        p_student_t = torch.nn.functional.log_softmax(y_student / T, dim=-1)
        distill_loss = self.distillation_loss_fn(p_student_t, p_teacher_t) * (T * T)

        total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss

        total_loss.backward()
        optimizer.step()

        return total_loss, student_loss, distill_loss