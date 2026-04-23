"""
model.py — Student Submission File
===================================
Define your dental image classifier here.
Your model MUST follow the interface below.

Rules:
  - Class name must be DentalClassifier
  - forward() input shape:  (batch, 3, H, W)  — RGB image tensor
  - forward() output shape: (batch, 2)         — raw logits
  - Do NOT change the class name or method signatures
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

class DentalClassifier(nn.Module):
    """
    Dental image binary classifier.
    Output classes:
        0 = non_cancerous
        1 = cancerous
    """

    def __init__(self):
        super(DentalClassifier, self).__init__()

        self.model = resnet18(pretrained = True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits tensor of shape (batch_size, 2)
        """
        out = self.model(x)

        return out 
