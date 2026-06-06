# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Model Module.

``DinoV3PlModel`` inherits the entire ``nvdinov2`` Lightning flow (training step, teacher
EMA, optimizer/scheduler config, callbacks, checkpoint saving) and overrides only
``_build_model`` to construct the DINOv3 RoPE ViT (no absolute pos-embed, per-type FFN)
using the patch-16 v3 param map.

The checkpoint remapper (Meta/timm DINOv3 -> this ViT) and the Gram-anchoring loss
(``_extra_losses``) land in later steps; this file is the build + inheritance scaffold so
the family imports, ``dinov3 --help`` resolves, and the ViT-B builds.
"""

import copy

import torch
import torch.nn as nn
from timm.layers import Mlp

import nvidia_tao_pytorch.config.dinov3.default_config as v3_params
from nvidia_tao_pytorch.ssl.nvdinov2.model.head import DinoHead
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel
from nvidia_tao_pytorch.ssl.nvdinov2.model.vit import SwiGLUFused
from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer
from nvidia_tao_pytorch.ssl.dinov3.model.loss import GramLoss

# Resolve the param-map FFN name to a layer class without importing torch in the config.
_MLP_LAYERS = {
    "mlp": Mlp,
    "swiglu": SwiGLUFused,
}


class DinoV3PlModel(DinoV2PlModel):
    """PyTorch Lightning module for DINOv3 (inherits nvdinov2)."""

    def __init__(self, experiment_spec):
        """Initialize the DINOv3 Lightning module.

        Args:
            experiment_spec: The DINOv3 experiment configuration.
        """
        super().__init__(experiment_spec)
        self.checkpoint_filename = 'dinov3_model'

        # Gram anchoring (DINOv3). The frozen Gram teacher is constructed in _build_model
        # when enabled; sync it from the (now teacher==student) weights here so it is
        # consistent even when no pretrained checkpoint is supplied. When a pretrained
        # checkpoint is loaded, restore_pretrained_weights re-syncs it from the loaded
        # teacher (the intended provenance for continual pre-training).
        if getattr(self, 'gram_teacher', None) is not None:
            self._sync_gram_teacher()

    def _resolve_arch(self, backbone_type):
        """Look up DINOv3 ViT hyper-parameters for a backbone type from the v3 param map.

        Args:
            backbone_type (str): One of the v3 ``SUPPORTED_BACKBONES`` (e.g. ``vit_b``).

        Returns:
            dict: ``embed_dim``, ``depth``, ``num_heads``, ``init_values``,
            ``drop_path_schedule``, ``num_classes`` and the resolved ``mlp_layer`` class.
        """
        mp = v3_params.map_params
        return {
            "embed_dim": mp['embed_dim'][backbone_type],
            "depth": mp['depth'][backbone_type],
            "num_heads": mp['num_heads'][backbone_type],
            "init_values": mp['init_values'][backbone_type],
            "drop_path_schedule": mp['drop_path_schedule'][backbone_type],
            "num_classes": mp['num_classes'][backbone_type],
            "mlp_layer": _MLP_LAYERS[mp['mlp_layer'][backbone_type]],
        }

    def _make_backbone(self, arch):
        """Construct a DINOv3 ViT backbone from a resolved arch dict.

        Args:
            arch (dict): Output of :meth:`_resolve_arch`.

        Returns:
            DinoV3VisionTransformer: The constructed backbone.
        """
        return DinoV3VisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=arch["embed_dim"],
            depth=arch["depth"],
            num_heads=arch["num_heads"],
            init_values=arch["init_values"],
            drop_path_schedule=arch["drop_path_schedule"],
            num_classes=arch["num_classes"],
            drop_path_rate=self.drop_path_rate,
            mlp_layer=arch["mlp_layer"],
            norm_layer=nn.LayerNorm,
            act_layer=nn.SiLU,
            register_tokens=self.register_tokens,
            use_custom_attention=self.use_custom_attention,
            rope_theta=self.model_config.backbone['rope_theta'],
        )

    def _make_head(self, embed_dim):
        """Construct a DINO/iBOT head for the given embedding dimension.

        Args:
            embed_dim (int): Backbone embedding dimension.

        Returns:
            DinoHead: The constructed head.
        """
        return DinoHead(
            in_dim=embed_dim,
            out_dim=self.num_prototypes,
            num_layers=self.head_layers,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
        )

    def _build_model(self):
        """Build the DINOv3 student and teacher (RoPE ViT) using the v3 param map.

        Overrides :meth:`DinoV2PlModel._build_model`: same student/teacher (+optional
        ``student_ema`` for distillation) structure, but built from ``DinoV3VisionTransformer``
        and the patch-16 v3 ``map_params`` (with per-type FFN selection). Dimension
        attributes are re-derived here from the v3 param map so they are self-consistent
        regardless of the (nvdinov2) param map the parent ``__init__`` read.
        """
        # Re-derive dims from the v3 (patch-16) param map.
        student_arch = self._resolve_arch(self.student_backbone_type)
        teacher_arch = self._resolve_arch(self.teacher_backbone_type)
        self.student_embed_dim = student_arch["embed_dim"]
        self.teacher_embed_dim = teacher_arch["embed_dim"]

        self.student = torch.nn.ModuleDict(
            {
                'backbone': self._make_backbone(student_arch),
                'dino_head': self._make_head(self.student_embed_dim),
                'ibot_head': self._make_head(self.student_embed_dim),
            }
        )
        self.teacher = torch.nn.ModuleDict(
            {
                'backbone': self._make_backbone(teacher_arch),
                'dino_head': self._make_head(self.teacher_embed_dim),
                'ibot_head': self._make_head(self.teacher_embed_dim),
            }
        )

        if self.model_config.distill.enable:
            # Create a student ema for distillation
            self.student_ema = copy.deepcopy(self.student)

            assert self.model_config.distill.pretrained_non_distill_pl_model_path is not None, (
                "In distillation mode, you need to provide the pretrained_non_distill_pl_model_path "
                "to initialize a frozen teacher."
            )
            pretrained_backbone_head_state_dict = torch.load(
                self.model_config.distill.pretrained_non_distill_pl_model_path, map_location="cpu"
            )['state_dict']
            teacher_state_dict = {}
            for k, v in list(pretrained_backbone_head_state_dict.items()):
                if "teacher." in k:
                    teacher_state_dict[k.replace("teacher.", "")] = v

            self.teacher.load_state_dict(teacher_state_dict)
        else:
            assert self.student_backbone_type == self.teacher_backbone_type, (
                f"In non-distillation mode, student_type and teacher_type should be the same. "
                f"Currently, the teacher_type is {self.teacher_backbone_type}, and the "
                f"student_type is {self.student_backbone_type}."
            )

        # Gram anchoring: build a separate frozen Gram teacher (a copy of the teacher
        # backbone) when enabled. It is intentionally NOT placed inside self.teacher /
        # self.student (so the parent's FSDP wrapping and CustomModelCheckpoint, which act on
        # those ModuleDicts, leave it alone) and never receives gradients or EMA updates.
        # Its weights are (re)synced from the teacher by _sync_gram_teacher.
        if self.model_config.gram.enable:
            self.gram_teacher = self._make_backbone(teacher_arch)
            for param in self.gram_teacher.parameters():
                param.requires_grad = False
            self.gram_teacher.eval()
            self.gram_loss = GramLoss()

    def _sync_gram_teacher(self):
        """Copy the current teacher backbone weights into the frozen Gram teacher.

        Called once at construction (teacher == student) and again after a pretrained
        checkpoint is loaded (``restore_pretrained_weights``), so the Gram teacher always
        anchors to the same DINOv3 weights the run starts from.
        """
        self.gram_teacher.load_state_dict(self.teacher.backbone.state_dict())
        for param in self.gram_teacher.parameters():
            param.requires_grad = False
        self.gram_teacher.eval()

    def restore_pretrained_weights(self):
        """Load pretrained weights, then re-anchor the Gram teacher to them.

        Reuses the inherited loader (the DINOv3 timm-key remapper lands in a later step) and,
        when Gram anchoring is on, snapshots the freshly-loaded teacher into the frozen Gram
        teacher so the anchor matches the run's starting weights.
        """
        super().restore_pretrained_weights()
        if getattr(self, 'gram_teacher', None) is not None:
            self._sync_gram_teacher()

    def _extra_losses(self, **ctx):
        """Inject the DINOv3 Gram-anchoring loss into the inherited student_forward.

        Returns an empty list (matching the DINOv2 base) unless Gram anchoring is enabled,
        the frozen Gram teacher has been built, and the global step has reached
        ``gram.start_step``. Otherwise it runs the frozen Gram teacher on the global crops and
        returns ``[w_gram * GramLoss(student_patches, teacher_patches)]``.

        Args:
            **ctx: Context forwarded from ``DinoV2PlModel.student_forward`` (global/local
                crops, masks, and the student global/local backbone outputs).

        Returns:
            list: ``[weighted_gram_loss]`` when active, else ``[]``.
        """
        gram_cfg = self.model_config.gram
        if not gram_cfg.enable or getattr(self, 'gram_teacher', None) is None:
            return []
        if self.global_step < gram_cfg.start_step:
            return []

        student_backbone_global_output = ctx.get("student_backbone_global_output")
        global_crops = ctx.get("global_crops")
        if student_backbone_global_output is None or global_crops is None:
            return []

        student_patch_tokens = student_backbone_global_output["x_norm_patchtokens"]

        # Frozen Gram teacher: same global crops, no grad. Force eval so the module's
        # stochastic depth / dropout stay off even though the parent module is in train mode.
        self.gram_teacher.eval()
        with torch.no_grad():
            teacher_patch_tokens = self.gram_teacher(global_crops)["x_norm_patchtokens"]

        gram_loss = self.gram_loss(student_patch_tokens, teacher_patch_tokens)
        self.log(
            "losses/gram_loss",
            gram_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )
        return [gram_cfg.w_gram * gram_loss]
