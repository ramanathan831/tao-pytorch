# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Default Config.

The DINOv3 SSL family inherits aggressively from ``nvdinov2``: the dataset, head,
distillation, scheduler, optimizer and train/inference/export schemas are reused by
subclassing the nvdinov2 dataclasses. Only the genuinely new pieces are added here:

* a v3 ``map_params`` table with **patch-16** ViT entries (ViT-B bring-up + ViT-L are
  supported; ViT-H+ is reserved for a later size step),
* RoPE backbone fields (``rope_theta`` etc.) and patch-16 defaults,
* a ``GramConfig`` for Gram anchoring (wired up in a later step), and
* a disabled ``lora`` stub for forward-compatibility.

DINOv3 is Meta IP implemented inside TAO; the family/endpoint is named ``dinov3``
(not ``nvdinov3``). ``nvdinov2`` stays frozen.
"""

from dataclasses import dataclass
from typing import Optional

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_pytorch.config.nvdinov2.default_config import (
    BackboneConfig,
    NVDINOv2HeadConfig,
    NVDINOv2ModelDistillConfig,
    NVDINOv2TransformConfig,
    NVDINOv2DatasetConfig,
    NVDINOv2TrainExpConfig,
    NVDINOv2InferenceExpConfig,
    NVDINOv2ExportExpConfig,
    GenTrtEngineExpConfig,
)
from nvidia_tao_pytorch.config.common.common_config import CommonExperimentConfig

# DINOv3 candidate / reserved architectures. ViT-B is the v1 bring-up target.
# ViT-L and ViT-H+ are reserved param-map entries for later size steps (Phase 2).
SUPPORTED_BACKBONES = [
    *["vit_b", "vit_l", "vit_h_plus"]
]

# DINOv3 ViT param map (patch-16). Distinct from the nvdinov2 (patch-14) map.
# FFN note: DINOv3 ViT-B/ViT-L use a standard MLP; only ViT-H+/7B use SwiGLU. The
# ``DinoV2VisionTransformer.__init__`` already accepts ``mlp_layer``, so the v3 build
# just passes the class named here ('mlp' -> timm Mlp, 'swiglu' -> SwiGLUFused).
map_params = {
    'embed_dim': {
        'vit_b': 768,
        'vit_l': 1024,
        'vit_h_plus': 1280,
    },
    'depth': {
        'vit_b': 12,
        'vit_l': 24,
        'vit_h_plus': 32,
    },
    'num_heads': {
        'vit_b': 12,
        'vit_l': 16,
        'vit_h_plus': 20,
    },
    'init_values': {
        'vit_b': 1e-5,
        'vit_l': 1e-5,
        'vit_h_plus': 1e-5,
    },
    'drop_path_schedule': {
        'vit_b': 'linear',
        'vit_l': 'linear',
        'vit_h_plus': 'linear',
    },
    'num_classes': {
        'vit_b': 0,
        'vit_l': 0,
        'vit_h_plus': 0,
    },
    'mlp_layer': {
        'vit_b': 'mlp',
        'vit_l': 'mlp',
        'vit_h_plus': 'swiglu',
    },
}


@dataclass
class DINOv3BackboneConfig(BackboneConfig):
    """DINOv3 backbone config (patch-16 + RoPE).

    Subclasses the nvdinov2 ``BackboneConfig`` and overrides the patch-16 defaults,
    the ViT-B default backbone type, and adds the RoPE frequency base. The absolute
    positional embedding of DINOv2 is replaced by axial RoPE, so there is no
    ``pos_embed`` field; the ``DinoV3VisionTransformer`` does not register one.
    """

    teacher_type: str = STR_FIELD(
        value="vit_b",
        default_value="vit_b",
        display_name="teacher backbone",
        description=(
            "Teacher backbone name. TAO's DINOv3 supports vit_b (bring-up) and vit_l; "
            "vit_h_plus is reserved for a later size step."
        ),
        valid_options=",".join(SUPPORTED_BACKBONES),
        popular="no"
    )
    student_type: str = STR_FIELD(
        value="vit_b",
        default_value="vit_b",
        display_name="student backbone",
        description=(
            "Student backbone name. TAO's DINOv3 supports vit_b (bring-up) and vit_l; "
            "vit_h_plus is reserved for a later size step."
        ),
        valid_options=",".join(SUPPORTED_BACKBONES),
        popular="no"
    )
    num_register_tokens: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=0,
        valid_max="inf",
        description="Number of register tokens (DINOv3 ViT-B uses 4)",
        display_name="num register tokens",
        popular="yes"
    )
    patch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        description="Size of patches (DINOv3 uses patch 16)",
        display_name="patch size",
        valid_options="16",
        popular="yes"
    )
    img_size: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Size of images for the backbone (single-res 256 in v1)",
        display_name="image size",
        valid_options="256,512,768",
        popular="yes"
    )
    rope_theta: float = FLOAT_FIELD(
        value=100.0,
        default_value=100.0,
        description=(
            "Frequency base for 2D axial RoPE. Must match the timm DINOv3 reference; "
            "verified by the step-4 feature-parity smoke test."
        ),
        display_name="RoPE theta",
        popular="yes"
    )


@dataclass
class GramConfig:
    """DINOv3 Gram-anchoring config.

    Gram anchoring regularizes the student's patch-token Gram matrix toward a frozen
    Gram teacher (initialized from the loaded DINOv3 weights). The loss term itself is
    wired in a later step; these fields configure when/how strongly it applies.
    """

    enable: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to add the Gram-anchoring loss term",
        display_name="enable gram",
        popular="yes"
    )
    w_gram: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Weight of the Gram-anchoring loss term",
        display_name="gram weight",
        popular="yes"
    )
    start_step: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Global step at which the Gram term activates",
        display_name="gram start step",
        popular="yes"
    )
    teacher_source: str = STR_FIELD(
        value="pretrained",
        default_value="pretrained",
        description="Source of the frozen Gram teacher weights",
        display_name="gram teacher source",
        valid_options="pretrained,ema",
        popular="no"
    )


@dataclass
class LoRAConfig:
    """Disabled LoRA stub for forward-compatibility (parameter-efficient SSL, Phase 2)."""

    enable: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to apply LoRA adapters (disabled in v1)",
        display_name="enable lora",
        popular="no"
    )
    rank: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="LoRA rank",
        display_name="lora rank",
        popular="no"
    )
    alpha: float = FLOAT_FIELD(
        value=16.0,
        default_value=16.0,
        description="LoRA scaling alpha",
        display_name="lora alpha",
        popular="no"
    )


@dataclass
class DINOv3ModelConfig:
    """DINOv3 model config (reuses nvdinov2 distill/head, adds gram + lora)."""

    centering_method: str = STR_FIELD(
        value="sinkhorn",
        default_value="sinkhorn",
        valid_options="sinkhorn,softmax",
        description=(
            "Teacher-output centering for the DINO/iBOT heads. DINOv3 uses Sinkhorn-Knopp "
            "(SwAV); 'softmax' is the DINOv2 fallback. If training shows instability/collapse, "
            "try 'softmax'."
        ),
        display_name="centering method",
        popular="yes"
    )
    distill: NVDINOv2ModelDistillConfig = DATACLASS_FIELD(
        NVDINOv2ModelDistillConfig(),
        description="Configuration for distillation (reused from nvdinov2)"
    )
    backbone: DINOv3BackboneConfig = DATACLASS_FIELD(
        DINOv3BackboneConfig(),
        description="Configuration for the DINOv3 backbone"
    )
    head: NVDINOv2HeadConfig = DATACLASS_FIELD(
        NVDINOv2HeadConfig(),
        description="Configuration for the DINOv3 head (reused from nvdinov2)"
    )
    gram: GramConfig = DATACLASS_FIELD(
        GramConfig(),
        description="Configuration for DINOv3 Gram anchoring"
    )
    lora: LoRAConfig = DATACLASS_FIELD(
        LoRAConfig(),
        description="Disabled LoRA stub for forward-compatibility"
    )


@dataclass
class DINOv3TransformConfig(NVDINOv2TransformConfig):
    """DINOv3 transform config (single-res 256, patch-16-friendly crop sizes)."""

    global_crops_size: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Size of global crops (single-res 256 in v1)",
        display_name="Global Crops Size",
        popular="yes"
    )
    local_crops_size: int = INT_FIELD(
        value=112,
        default_value=112,
        valid_min=1,
        valid_max="inf",
        description="Size of local crops (multiple of patch 16)",
        display_name="Local Crops Size",
        popular="yes"
    )


@dataclass
class DINOv3DatasetConfig(NVDINOv2DatasetConfig):
    """DINOv3 dataset config (reuses nvdinov2 dataset, v3 transform defaults)."""

    transform: DINOv3TransformConfig = DATACLASS_FIELD(
        DINOv3TransformConfig(),
        description="Configuration parameters for data transformation",
        display_name="transform",
    )


@dataclass
class DINOv3ConvertConfig:
    """DINOv3 backbone-export (``convert``) config.

    Converts an SSL-trained DINOv3 checkpoint into the timm-format layout that the
    ``cv/backbone_v2`` ``dinov3_vitb16`` registry entry (and downstream supervised tasks)
    consume. The EMA ``teacher`` is the recommended feature extractor for continual
    pre-training, so it is the default source.
    """

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Directory for convert results/logs",
        display_name="results dir"
    )
    checkpoint: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "SSL DINOv3 checkpoint to convert: a stripped backbone file "
            "(student_*.pth / teacher_*.pth) or a full Lightning .pth/.ckpt."
        ),
        display_name="checkpoint",
        popular="yes"
    )
    output_path: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "Output path for the timm-format backbone (.safetensors or .pth). Defaults to "
            "<results_dir>/dinov3_<arch>_backbone.safetensors."
        ),
        display_name="output path",
        popular="yes"
    )
    source: str = STR_FIELD(
        value="teacher",
        default_value="teacher",
        valid_options="student,teacher,student_ema",
        description="Which SSL sub-model's backbone to export (EMA teacher recommended).",
        display_name="source",
        popular="yes"
    )
    validate: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Validate the converted state dict against a fresh timm DINOv3 model.",
        display_name="validate"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """DINOv3 experiment config."""

    model: DINOv3ModelConfig = DATACLASS_FIELD(
        DINOv3ModelConfig(),
        description="Configurable parameters to construct the model for a DINOv3 experiment.",
    )
    dataset: DINOv3DatasetConfig = DATACLASS_FIELD(
        DINOv3DatasetConfig(),
        description="Configurable parameters to construct the dataset for a DINOv3 experiment.",
    )
    train: NVDINOv2TrainExpConfig = DATACLASS_FIELD(
        NVDINOv2TrainExpConfig(),
        description="Configurable parameters to construct the trainer for a DINOv3 experiment.",
    )
    inference: NVDINOv2InferenceExpConfig = DATACLASS_FIELD(
        NVDINOv2InferenceExpConfig(),
        description="Configurable parameters to construct the inference trainer for a DINOv3 experiment.",
    )
    export: NVDINOv2ExportExpConfig = DATACLASS_FIELD(
        NVDINOv2ExportExpConfig(),
        description="Configurable parameters to export for a DINOv3 experiment.",
    )
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(
        GenTrtEngineExpConfig(),
        description="Configurable parameters to generate TensorRT engine for a DINOv3 experiment.",
    )
    convert: DINOv3ConvertConfig = DATACLASS_FIELD(
        DINOv3ConvertConfig(),
        description="Configurable parameters to convert an SSL backbone to the backbone_v2 (timm) layout.",
    )

    def __post_init__(self):
        """Set default model name for DINOv3."""
        if self.model_name is None:
            self.model_name = "dinov3"


# Expose a single MLP-layer registry so the pl_model resolves the param-map string
# ('mlp'/'swiglu') without importing torch at config-parse time.
MLP_LAYER_NAMES = ("mlp", "swiglu")
