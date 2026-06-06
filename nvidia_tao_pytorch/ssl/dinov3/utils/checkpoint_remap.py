# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional state-dict key remapping between timm DINOv3 and TAO ``ssl/dinov3``.

Single source of truth for the DINOv3 ViT key renames, so the two directions can never
drift apart:

* **timm -> TAO** (:func:`timm_to_tao`) — used by ``DinoV3PlModel`` to load public DINOv3
  weights into the SSL backbone at the start of continual pre-training.
* **TAO -> timm** (:func:`tao_to_timm`) — used by the ``dinov3 convert`` subtask to export an
  SSL-trained backbone into the timm layout that the ``cv/backbone_v2`` ``dinov3_vitb16``
  registry entry (and downstream supervised tasks) consume.

The two ViTs are numerically identical (verified by the feature-parity smoke test), so this
is purely a key-naming translation; no weights are transformed.
"""

import os

import torch

# Exact full-key renames, timm name -> TAO name.
_EXACT_TIMM_TO_TAO = {
    "reg_token": "register_tokens",
}
# Per-block suffix renames, timm suffix -> TAO suffix (LayerScale: raw gammas vs ls modules).
_SUFFIX_TIMM_TO_TAO = {
    ".gamma_1": ".ls1.gamma",
    ".gamma_2": ".ls2.gamma",
}
# Inverses.
_EXACT_TAO_TO_TIMM = {v: k for k, v in _EXACT_TIMM_TO_TAO.items()}
_SUFFIX_TAO_TO_TIMM = {v: k for k, v in _SUFFIX_TIMM_TO_TAO.items()}

# TAO-side params/buffers absent from a timm DINOv3 (inference) checkpoint.
# ``mask_token`` is an iBOT parameter; ``rope.periods`` is a non-persistent buffer (already
# excluded from state dicts, but guarded here for robustness).
TAO_ONLY_KEYS = ("mask_token",)

# TAO backbone arch -> timm DINOv3 model name (architecture only; pretrained=False).
_TIMM_MODEL_BY_ARCH = {
    "vit_b": "vit_base_patch16_dinov3",
    "vit_l": "vit_large_patch16_dinov3",
    "vit_h_plus": "vit_huge_plus_patch16_dinov3",
}


def timm_model_name_for_arch(arch):
    """Return the timm DINOv3 model name for a TAO backbone arch (e.g. ``vit_b``).

    Args:
        arch (str): TAO backbone type (``vit_b`` / ``vit_l`` / ``vit_h_plus``).

    Returns:
        str: The corresponding timm model name.
    """
    if arch not in _TIMM_MODEL_BY_ARCH:
        raise ValueError(
            f"No timm DINOv3 model name known for arch '{arch}'. Known: {list(_TIMM_MODEL_BY_ARCH)}"
        )
    return _TIMM_MODEL_BY_ARCH[arch]


def timm_to_tao(key):
    """Translate a timm DINOv3 ViT parameter name to the TAO ``ssl/dinov3`` name.

    Args:
        key (str): timm-side parameter name.

    Returns:
        str: TAO-side parameter name (unchanged if no rule applies).
    """
    if key in _EXACT_TIMM_TO_TAO:
        return _EXACT_TIMM_TO_TAO[key]
    for src, dst in _SUFFIX_TIMM_TO_TAO.items():
        if key.endswith(src):
            return key[: -len(src)] + dst
    return key


def tao_to_timm(key):
    """Translate a TAO ``ssl/dinov3`` ViT parameter name to the timm DINOv3 name.

    Args:
        key (str): TAO-side parameter name.

    Returns:
        str: timm-side parameter name (unchanged if no rule applies).
    """
    if key in _EXACT_TAO_TO_TIMM:
        return _EXACT_TAO_TO_TIMM[key]
    for src, dst in _SUFFIX_TAO_TO_TIMM.items():
        if key.endswith(src):
            return key[: -len(src)] + dst
    return key


def load_checkpoint_file(path):
    """Load a ``.safetensors`` / ``.pth`` / ``.ckpt`` checkpoint into a dict.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        dict: The loaded object (a state dict, or a container with ``state_dict``).
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def save_state_dict(state_dict, path):
    """Save a flat state dict as ``.safetensors`` (by extension) or ``.pth``.

    Args:
        state_dict (dict): Parameter-name -> tensor mapping.
        path (str): Output path; ``.safetensors`` extension selects the safetensors writer.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file({k: v.detach().contiguous().cpu() for k, v in state_dict.items()}, path)
    else:
        torch.save(state_dict, path)


def extract_backbone_state_dict(raw, source="teacher"):
    """Normalize any DINOv3 SSL checkpoint into a backbone-level state dict (TAO naming).

    Handles two input shapes:

    * a **stripped backbone file** (``student_*.pth`` / ``teacher_*.pth`` from
      ``CustomModelCheckpoint``) whose keys are already backbone-level, or
    * a **full Lightning checkpoint** with ``<source>.backbone.*`` (and head / gram-teacher)
      keys, from which the chosen source's backbone is selected and de-prefixed.

    Args:
        raw (dict): Loaded checkpoint (a state dict, or a container with ``state_dict``).
        source (str): Which sub-model's backbone to extract from a full checkpoint
            (``student`` / ``teacher`` / ``student_ema``). Ignored for stripped files.

    Returns:
        dict: Backbone-level state dict in TAO naming.
    """
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    if any(".backbone." in k for k in state_dict):
        prefix = f"{source}.backbone."
        extracted = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not extracted:
            available = sorted({k.split(".backbone.")[0] for k in state_dict if ".backbone." in k})
            raise ValueError(
                f"No keys with prefix '{prefix}' in checkpoint. Available backbone sources: {available}"
            )
        return extracted

    # Already a stripped backbone file; drop any stray (non-backbone) head keys defensively.
    return {k: v for k, v in state_dict.items() if not k.startswith(("dino_head", "ibot_head"))}


def remap_tao_backbone_to_timm(tao_state_dict):
    """Rename a TAO backbone state dict into timm DINOv3 layout, dropping TAO-only keys.

    Args:
        tao_state_dict (dict): Backbone-level state dict in TAO naming.

    Returns:
        dict: State dict in timm naming (contiguous tensors).
    """
    out = {}
    for key, weight in tao_state_dict.items():
        if key in TAO_ONLY_KEYS or key.endswith("rope.periods"):
            continue
        out[tao_to_timm(key)] = weight.contiguous()
    return out


def validate_against_timm(timm_state_dict, timm_model_name="vit_base_patch16_dinov3"):
    """Assert a state dict matches a fresh timm DINOv3 model's keys and shapes.

    Args:
        timm_state_dict (dict): Candidate state dict in timm naming.
        timm_model_name (str): timm model whose architecture defines the expected keys.

    Raises:
        ValueError: If any key is missing, unexpected, or shape-mismatched.
    """
    import timm
    reference = timm.create_model(timm_model_name, pretrained=False).state_dict()
    missing = sorted(set(reference) - set(timm_state_dict))
    unexpected = sorted(set(timm_state_dict) - set(reference))
    shape_mismatch = [
        k for k in timm_state_dict
        if k in reference and tuple(reference[k].shape) != tuple(timm_state_dict[k].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "Converted state dict is not timm-compatible:\n"
            f"  missing={missing}\n  unexpected={unexpected}\n  shape_mismatch={shape_mismatch}"
        )


def convert_ssl_to_timm(src_path, dst_path, source="teacher", validate=True,
                        timm_model_name="vit_base_patch16_dinov3"):
    """Convert a TAO DINOv3 SSL checkpoint into a timm-format backbone file.

    The output is loadable by ``timm.create_model(..., checkpoint_path=dst_path)`` and hence by
    the ``cv/backbone_v2`` ``dinov3_vitb16`` registry entry for downstream supervised tasks.

    Args:
        src_path (str): SSL checkpoint (stripped backbone ``.pth`` or full Lightning ``.ckpt``).
        dst_path (str): Output path (``.safetensors`` or ``.pth``).
        source (str): Which backbone to export (``teacher`` recommended — the EMA teacher).
        validate (bool): If True, validate the result against a fresh timm DINOv3 model.
        timm_model_name (str): timm model name for validation / target layout.

    Returns:
        dict: The converted timm-format state dict.
    """
    raw = load_checkpoint_file(src_path)
    backbone_state_dict = extract_backbone_state_dict(raw, source=source)
    timm_state_dict = remap_tao_backbone_to_timm(backbone_state_dict)
    if validate:
        validate_against_timm(timm_state_dict, timm_model_name=timm_model_name)
    save_state_dict(timm_state_dict, dst_path)
    return timm_state_dict
