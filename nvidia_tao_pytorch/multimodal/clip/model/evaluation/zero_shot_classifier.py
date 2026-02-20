# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CLIP zero-shot classification module"""

import math
from itertools import islice
from typing import Callable, Optional, Sequence

import torch
from torch.distributed import ReduceOp

# TODO: remove dependency in these dist if not needed?
from nvidia_tao_pytorch.multimodal.clip.utils.dist import all_gather, all_reduce, get_rank, get_world_size
from nvidia_tao_pytorch.multimodal.clip.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy"""
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).sum(0, keepdim=True).long() for k in topk]


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


@torch.no_grad()
def build_zero_shot_classifier(
    model,
    tokenizer,
    classnames: Sequence[str] = None,  # TODO: Add error checks if this is none
    templates: Sequence[Callable | str] = None,
    num_classes_per_batch: Optional[int] = 10,
    device: Optional[torch.device | str] = "cuda",
    distributed: bool = True,
) -> torch.Tensor:
    """Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: The device to run on, defaults to torch.device('cuda') if available
        distributed: Whether to run in distributed mode

    Returns:
        A tensor of shape (d, c) where d is the dimensionality of the CLIP model and c is the number of classes
    """
    if not isinstance(templates, Sequence) or len(templates) == 0:
        raise ValueError(
            "templates must be a non-empty sequence of strings or callables"
        )
    if not isinstance(classnames, Sequence) or len(classnames) == 0:
        raise ValueError(
            "classnames must be a non-empty sequence of class names"
        )

    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)

    # Split classnames across all processes
    if distributed:
        lens_per_rank = math.ceil(num_classes / get_world_size())
        local_classnames = classnames[
            get_rank() * lens_per_rank: (get_rank() + 1) * lens_per_rank
        ]
        if len(local_classnames) < lens_per_rank:
            local_classnames += ("placeholder",) * (
                lens_per_rank - len(local_classnames)
            )

        rank_str = f"({len(local_classnames)} classes on each rank)"
    else:
        local_classnames = classnames
        rank_str = ""

    logger.debug(
        f"Building zero-shot classifier: {num_classes} classes, {num_templates} templates {rank_str}"
    )

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [
            template.format(c) if use_format else template(c)
            for c in batch_classnames
            for template in templates
        ]
        # Tokenize - all tokenizers return [dict] with 'input_ids' (and optionally 'attention_mask')
        tokenized = tokenizer(texts)
        texts = {k: v.to(device) for k, v in tokenized[0].items()}
        class_embeddings = model(text=texts)
        class_embeddings = (
            class_embeddings["text_features"]
            if isinstance(class_embeddings, dict)
            else class_embeddings[1]
        )
        class_embeddings = class_embeddings.reshape(
            num_batch_classes, num_templates, -1
        ).mean(dim=1)
        class_embeddings = class_embeddings / \
            class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    if num_classes_per_batch:
        batched_embeds = [
            _process_batch(batch)
            for batch in batched(local_classnames, num_classes_per_batch)
        ]
        zeroshot_weights = torch.cat(batched_embeds, dim=1)
    else:
        zeroshot_weights = _process_batch(local_classnames)

    if distributed:
        # Gather all embeddings across all processes
        tensor_list = [
            torch.zeros_like(zeroshot_weights) for _ in range(get_world_size())
        ]
        all_gather(tensor_list, zeroshot_weights, async_op=False)

        # Concatenate all embeddings
        zeroshot_weights = torch.cat(tensor_list, dim=1)

        # Trim to only the number of classes we need
        zeroshot_weights = zeroshot_weights[:, :num_classes]

    logger.debug(
        f"Zero-shot classifier ready: {len(local_classnames)} classes on rank {get_rank()}"
    )

    return zeroshot_weights


@torch.no_grad()
def run_zero_shot_classifier(
    model,
    classifier,
    dataloader,
    device: Optional[torch.device | str] = "cuda",
):
    """Run zero-shot classifier on a dataset
    Args:
        model: CLIP model instance
        classifier: Zero-shot classifier weights
        dataloader: A torch.utils.data.DataLoader instance
        device: The device to run on, defaults to torch.device('cuda') if available
    """
    record = torch.tensor([0, 0, 0], device="cpu", dtype=torch.long)
    for images, labels in dataloader:
        # Handle both tensor and dict (for SigLIP2) image formats
        if isinstance(images, dict):
            images = {k: v.to(device) for k, v in images.items()}
            batch_size = images['pixel_values'].shape[0]
        else:
            images = images.to(device)
            batch_size = images.shape[0]
        labels = labels.to(device)

        image_features = model(image=images)
        image_features = (
            image_features["image_features"]
            if isinstance(image_features, dict)
            else image_features[0]
        )
        logits = 100.0 * image_features @ classifier

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        acc1, acc5 = acc1.cpu(), acc5.cpu()
        record += torch.concat(
            [acc1, acc5, torch.tensor([batch_size], dtype=torch.long)]
        )

    # Move to distributed device, and sum across all processes
    record = record.to(device)
    all_reduce(record, op=ReduceOp.SUM, async_op=False)

    # Compute final accuracy
    acc1, acc5, n = record.cpu().tolist()
    acc1 = acc1 / n
    acc5 = acc5 / n

    logger.info(
        f"Finished running zero-shot classifier on {n} images, acc1: {acc1:.4f}, acc5: {acc5:.4f}"
    )

    return acc1, acc5
