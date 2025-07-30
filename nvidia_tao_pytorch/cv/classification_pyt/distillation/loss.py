# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distillation Loss module for knowledge distillation."""
import math
from typing import Union, List, Tuple, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.distillation.losses import LPCriterion, KLDivCriterion
from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO
from nvidia_tao_pytorch.cv.classification_pyt.utils.loss import Cross_Entropy


class ProjectionMLP(nn.Module):
    """Multi-layer perceptron for feature projection and dimension alignment in distillation.

    This MLP is designed to project features from one dimension to another, commonly used
    in knowledge distillation to align student and teacher feature dimensions. It supports
    optional pre-normalization, configurable depth with residual connections, and spatial
    upsampling for feature map distillation.

    The architecture consists of:
    1. Optional pre-normalization (LayerNorm + GELU)
    2. Input projection layer
    3. Configurable number of inner residual blocks
    4. Final projection layer with LayerNorm + GELU
    5. Optional spatial upsampling for feature maps

    Args:
        input_size (int): Input feature dimension.
        hidden_size (int): Hidden layer dimension (before upsampling adjustment).
        output_size (int): Output feature dimension (before upsampling adjustment).
        num_inner (int, optional): Number of inner residual blocks. Default: 0.
        pre_norm (bool, optional): Whether to apply pre-normalization. Default: False.
        device (torch.device, optional): Device to place the module on. Default: None.
        upsample_factor (int, optional): Factor for spatial upsampling. Default: 1.
        upsample_rank (int, optional): Maximum rank constraint for upsampled hidden size. Default: 0.
        **kwargs: Additional arguments (unused).

    Attributes:
        pre_norm (nn.Module): Pre-normalization layer or identity.
        upsample_factor (int): Upsampling factor for spatial dimensions.
        fc1 (nn.Linear): Input projection layer.
        blocks (nn.ModuleList): List of inner residual blocks.
        final (nn.Sequential): Final projection with normalization and activation.

    Example:
        >>> # Basic projection MLP
        >>> proj = ProjectionMLP(input_size=768, hidden_size=1024, output_size=512)
        >>> x = torch.randn(32, 196, 768)  # [batch, tokens, features]
        >>> output = proj(x)  # Shape: [32, 196, 512]

        >>> # MLP with upsampling for spatial feature maps
        >>> proj = ProjectionMLP(
        ...     input_size=256, hidden_size=512, output_size=512,
        ...     upsample_factor=2, num_inner=2
        ... )
        >>> x = torch.randn(32, 49, 256)  # [batch, 7*7 tokens, features]
        >>> output = proj(x)  # Shape: [32, 196, 512] (14*14 tokens after upsampling)

    Note:
        When upsample_factor > 1, the input is assumed to represent spatial tokens
        arranged in a square grid (h = w = sqrt(num_tokens)). The output will have
        (upsample_factor^2) times more spatial tokens.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_inner: int = 0,
                 pre_norm: bool = False,
                 device: torch.device = None,
                 upsample_factor: int = 1,
                 upsample_rank: int = 0,
                 **kwargs) -> None:
        super().__init__()
        self.pre_norm = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
        ) if pre_norm else nn.Identity()

        self.upsample_factor = upsample_factor
        self._real_output_dim = output_size

        hidden_size = hidden_size * upsample_factor
        if upsample_rank:
            hidden_size = min(hidden_size, upsample_rank)
        output_size *= (upsample_factor ** 2)

        self.fc1 = nn.Linear(input_size, hidden_size, device=device)

        blocks = []
        for _ in range(num_inner):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_size, device=device),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, device=device),
            ))
        self.blocks = nn.ModuleList(blocks)

        flin = nn.Linear(hidden_size, output_size, device=device)
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size, device=device),
            nn.GELU(),
            flin,
        )
        flin.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the ProjectionMLP."""
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)

        if self.upsample_factor > 1:
            h = w = int(math.sqrt(x.shape[1]))
            x = rearrange(x, 'b (h w) (u1 u2 c) -> b (h u1 w u2) c',
                          h=h, w=w, u1=self.upsample_factor, u2=self.upsample_factor,
                          c=self._real_output_dim)

        return x


class CosineSimilarityLoss():
    """Cosine similarity loss for feature distillation."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def __call__(self, normalized_student_features: torch.Tensor, normalized_teacher_features: torch.Tensor):
        """Compute cosine similarity loss."""
        cs = nn.CosineSimilarity(dim=-1, eps=self.eps)(normalized_student_features, normalized_teacher_features)
        return 1.0 - cs.mean()


class BalancedFeatureLoss:
    """Balanced feature loss for feature distillation."""

    def __init__(self, weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, normalized_student_features: torch.Tensor, normalized_teacher_features: torch.Tensor):
        """Compute balanced feature loss."""
        loss_l1 = nn.SmoothL1Loss(beta=2.0)(normalized_student_features, normalized_teacher_features)
        loss_cos = CosineSimilarityLoss(eps=self.eps)(normalized_student_features, normalized_teacher_features)
        loss = (1 - self.weight) * loss_cos + self.weight * loss_l1
        return loss


class DistillationLoss(nn.Module):
    """A modular distillation loss module that supports various loss types for knowledge distillation.

    This module can handle both logit distillation and feature map distillation, automatically
    handling dimension mismatches between teacher and student models through projection layers.

    Supported loss types:
    - "CE": Cross Entropy loss for logit distillation
    - "KL": KL Divergence loss for logit distillation
    - "L1": L1 loss for feature distillation
    - "L2": L2 loss for feature distillation
    - "FD": Feature Distillation using Smooth L1 loss
    - "CS": Cosine Similarity loss for feature distillation
    - "BALANCED": Balanced feature loss for feature distillation
    """

    def __init__(
        self,
        loss_type: str,
        student_model: nn.Module,
        teacher_model: nn.Module,
        num_classes: int,
        distillation_mode: str = "auto",
        temperature: float = 1.0,
        normalize_features: bool = True,
        use_mlp: bool = True,
        mlp_hidden_size: int = 1024,
        mlp_num_inner: int = 2,
    ):
        """
        Initialize the DistillationLoss module.

        Args:
            loss_type (str): Type of distillation loss. One of ["CE", "KL", "L1", "L2", "FD", "CS"]
            student_model (nn.Module): Student model for distillation
            teacher_model (nn.Module): Teacher model for distillation
            num_classes (int, optional): Number of classes. Used for validation in feature distillation modes.
            distillation_mode (str): Mode for distillation. Options:
                - "logits": Use model.forward() for logit distillation
                - "summary": Use model.forward_pre_logits() for summary/cls token distillation
                - "auto": Automatically determine based on loss_type (CE/KL -> logits, others -> features)
            temperature (float): Temperature for knowledge distillation. Default: 1.0
            normalize_features (bool): Whether to apply layer normalization to teacher features for FD/CS losses.
                Default: True
            use_mlp (bool): Whether to use MLP for projection. Default: False
            mlp_hidden_size (int): Hidden size for MLP. Default: 1024
            mlp_num_inner (int): Number of inner layers for MLP. Default: 2
        """
        super().__init__()

        self.loss_type = loss_type.upper()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.normalize_features = normalize_features

        # Validate loss type
        valid_loss_types = ["CE", "KL", "L1", "L2", "FD", "CS", "BALANCED"]
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"Unsupported loss type: {loss_type}. Must be one of {valid_loss_types}")

        # Determine distillation mode
        if distillation_mode.lower() == "auto":
            # Auto-detect based on loss type
            if self.loss_type in ["CE", "KL"]:
                self.distillation_mode = "logits"
            elif self.loss_type == "BALANCED":
                self.distillation_mode = "spatial"
                # in spatial mode, we only distill the last feature map
            else:
                self.distillation_mode = "summary"
        else:
            valid_modes = ["logits", "summary", "spatial"]
            if distillation_mode.lower() not in valid_modes:
                raise ValueError(f"Invalid distillation_mode: {distillation_mode}. Must be one of {valid_modes} or 'auto'")
            self.distillation_mode = distillation_mode.lower()

        # Validate configuration for feature distillation
        if self.loss_type in ["FD", "CS", "BALANCED"] and self.distillation_mode == "logits":
            raise ValueError(f"Loss type '{self.loss_type}' requires feature distillation mode, but 'logits' mode was specified")

        if self.distillation_mode != "logits" and self.loss_type in ["CE", "KL"]:
            raise ValueError(f"Loss type '{self.loss_type}' requires logits distillation mode, but {self.distillation_mode} mode was specified")

        if self.loss_type in ["FD", "CS", "BALANCED"] and num_classes > 0:
            raise ValueError(f"Number of classes must be 0 when using '{self.loss_type}' for distillation")

        # Get model dimensions by checking available methods
        self.student_dim, self.teacher_dim = self._get_model_dimensions()
        logging.info(f"student_dim: {self.student_dim}, teacher_dim: {self.teacher_dim}")

        # Create projection layer if dimensions differ and we're doing feature distillation
        self.projection_layer = None
        if self.student_dim != self.teacher_dim:
            if use_mlp:
                self.projection_layer = ProjectionMLP(self.student_dim, mlp_hidden_size, self.teacher_dim, num_inner=mlp_num_inner)
            else:
                self.projection_layer = nn.Linear(self.student_dim, self.teacher_dim, bias=True)

        # Initialize loss functions
        self.criterions = {
            "L1": LPCriterion(p=1),
            "L2": LPCriterion(p=2),
            "KL": KLDivCriterion(),
            "CE": Cross_Entropy(soft=True, label_smoothing=False),
            "FD": nn.SmoothL1Loss(beta=2.0),
            "CS": CosineSimilarityLoss(eps=1e-8),
            "BALANCED": BalancedFeatureLoss(eps=1e-8),
        }

        # Create layer normalization for feature distillation if specified
        if self.normalize_features and self.distillation_mode == "summary":
            self.teacher_norm = nn.LayerNorm(self.teacher_dim, elementwise_affine=False)
        else:
            self.teacher_norm = None

    def _get_model_dimensions(self):
        """Get the output dimensions for student and teacher models."""
        if self.distillation_mode == "logits":
            # For logits, try to get num_classes or use a test forward pass
            student_dim = teacher_dim = self.num_classes
        elif self.distillation_mode == "summary":
            # For features, try to get num_features
            student_dim = self.student_model.num_features
            teacher_dim = self.teacher_model.num_features
        else:
            if isinstance(self.student_model, RADIO):
                student_dim = self.student_model.num_features // len(self.student_model.summary_idxs)
            else:
                student_dim = self.student_model.num_features
            if isinstance(self.teacher_model, RADIO):
                teacher_dim = self.teacher_model.num_features // len(self.teacher_model.summary_idxs)
            else:
                teacher_dim = self.teacher_model.num_features
        return student_dim, teacher_dim

    def _interpolate_to_size(self, features: Union[torch.Tensor, List[torch.Tensor]], shape: Tuple[int, int]):
        if isinstance(features, (list, tuple)):
            return [self._interpolate_to_size(ft, shape) for ft in features]

        if features.shape[2:] != shape:
            features = F.interpolate(
                features,
                size=shape,
                mode='bilinear',
                align_corners=True,
            )
        return features

    @staticmethod
    def _get_last_feature_map(features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]):
        if isinstance(features, (list, tuple)):
            return features[-1]
        elif isinstance(features, dict):
            return list(features.values())[-1]
        return features

    def forward(self, batch_input: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher outputs.

        Args:
            batch_input (torch.Tensor): Input batch data to be passed through both models

        Returns:
            torch.Tensor: Computed distillation loss
        """
        # Get outputs based on distillation mode
        if self.distillation_mode == "logits":
            # Use standard forward pass for logits
            student_output = self.student_model(batch_input)
            with torch.no_grad():
                teacher_output = self.teacher_model(batch_input)
        elif self.distillation_mode == "spatial":
            student_output = self.student_model.forward_feature_pyramid(batch_input)
            student_output = self._get_last_feature_map(student_output)
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_feature_pyramid(batch_input)
                teacher_output = self._get_last_feature_map(teacher_output)
            # align the shape of student and teacher feature maps
            if student_output.shape[2:] != teacher_output.shape[2:]:  # B, C, H, W
                max_shape = tuple(
                    max(s, t)
                    for s, t in zip(student_output.shape[2:], teacher_output.shape[2:])
                )
                student_output = self._interpolate_to_size(student_output, max_shape)
                teacher_output = self._interpolate_to_size(teacher_output, max_shape)
            # [B, C, H, W] -> [B, H*W, C]
            student_output = rearrange(student_output, 'b c h w -> b (h w) c')
            teacher_output = rearrange(teacher_output, 'b c h w -> b (h w) c')
        else:
            # Use forward_pre_logits for summary token distillation
            student_output = self.student_model.forward_pre_logits(batch_input)
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_pre_logits(batch_input)

        # Handle projection for feature distillation
        if self.distillation_mode != "logits" and self.projection_layer is not None:
            student_output = self.projection_layer(student_output)

        # Apply teacher normalization if specified
        if self.teacher_norm is not None and self.loss_type in ["FD", "CS", "BALANCED"]:
            teacher_output = self.teacher_norm(teacher_output)

        # Compute loss based on type
        if self.loss_type == "CE":
            # Cross entropy loss for logit distillation
            teacher_probs = F.softmax(teacher_output / self.temperature, dim=-1)
            loss = self.criterions["CE"](student_output / self.temperature, teacher_probs)

        elif self.loss_type == "KL":
            # KL divergence loss for logit distillation
            loss = self.criterions["KL"](student_output / self.temperature, teacher_output / self.temperature)
        else:
            # Direct loss computation for L1, L2, FD, CS, BALANCED
            loss = self.criterions[self.loss_type](student_output, teacher_output)

        return loss

    def get_loss_info(self) -> dict:
        """
        Get information about the configured loss.

        Returns:
            dict: Dictionary containing loss configuration details
        """
        return {
            "loss_type": self.loss_type,
            "distillation_mode": self.distillation_mode,
            "student_dim": self.student_dim,
            "teacher_dim": self.teacher_dim,
            "num_classes": self.num_classes,
            "temperature": self.temperature,
            "has_projection": self.projection_layer is not None,
            "normalize_features": self.normalize_features,
        }
