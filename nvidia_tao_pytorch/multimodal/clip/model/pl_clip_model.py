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

"""CLIP Model PyTorch Lightning Module."""

import math

import torch

# Upper bound on logit_scale to prevent training instability.
# exp(4.6052) ≈ 100, matching OpenAI CLIP's original clamp.
_MAX_LOGIT_SCALE = math.log(100)

from open_clip.loss import ClipLoss, SigLipLoss  # noqa: E402

from nvidia_tao_pytorch.core.tlt_logging import logging  # noqa: E402

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import (  # noqa: E402
    TAOLightningModule,
)  # noqa: E402
from nvidia_tao_pytorch.core.loggers import (  # noqa: E402
    api_logging as status_logging,
)  # noqa: E402

from nvidia_tao_pytorch.multimodal.clip.model.clip import (  # noqa: E402
    build_model,
)  # noqa: E402
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.utils import (  # noqa: E402
    create_classifier_templates,
    create_classnames_mapping,
)  # noqa: E402
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (  # noqa: E402
    build_optimizer,
    compute_lr,
)  # noqa: E402
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.zero_shot_classifier import (  # noqa: E402,E501
    build_zero_shot_classifier,
    accuracy,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.zero_shot_metadata import (  # noqa: E402,E501
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
    GENERIC_TEMPLATES,
)


# pylint:disable=too-many-ancestors
class CLIPPlModel(TAOLightningModule):
    """PTL module for CLIP Model."""

    def __init__(self, experiment_spec, export=False):
        """Init training for Visual ChangeNet Model."""
        super().__init__(experiment_spec)
        # Overriding what's done in super()
        self.experiment_spec = experiment_spec
        self.checkpoint_filename = 'clip'
        clip_model = build_model(
            experiment_config=self.experiment_spec, export=export
        )
        self.model = clip_model.model
        self.tokenizer = clip_model.tokenizer
        self.preprocess_train, self.preprocess_val = (
            clip_model.preprocess_train,
            clip_model.preprocess_val,
        )

        if getattr(self.experiment_spec.train, "grad_checkpointing", False):
            self.model.set_grad_checkpointing()
            logging.info("Gradient checkpointing enabled")

        # Setup validation (optional — not needed for export/inference)
        val_cfg = getattr(self.experiment_spec.dataset, 'val', None)
        val_type = getattr(val_cfg, 'type', None) if val_cfg else None
        templates_file = getattr(val_cfg, 'templates_file', None) if val_cfg else None
        classnames_file = getattr(val_cfg, 'classnames_file', None) if val_cfg else None

        if templates_file:
            self.templates = create_classifier_templates(templates_file)
        elif val_type == 'classification':
            self.templates = OPENAI_IMAGENET_TEMPLATES
        else:
            self.templates = GENERIC_TEMPLATES

        if classnames_file:
            self.classnames, self.class_mapping = create_classnames_mapping(
                classnames_file
            )
        elif val_type == 'classification':
            self.classnames = IMAGENET_CLASSNAMES
            self.class_mapping = None
        else:
            self.classnames = None
            self.class_mapping = None

        self.loss_type = self.experiment_spec.train.loss_type

    def setup(self, stage=None):
        """Set up training after Trainer is initialized."""
        if stage == 'fit':
            # TODO: verify this for multi-node, multi-gpu
            # https://github.com/Lightning-AI/pytorch-lightning/
            # pull/11599/files
            self.max_steps = self.trainer.estimated_stepping_batches
            self._build_criterion()

    def _build_criterion(self):
        """Build the loss function."""
        # Setup loss
        if self.loss_type == 'siglip':
            self.loss = SigLipLoss(
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        elif self.loss_type == 'clip':
            self.loss = ClipLoss(
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        else:
            raise NotImplementedError(
                f"loss function {self.loss_type} is not implemented"
            )
        self.criterion = self.loss

    def configure_optimizers(self):
        """Configure optimizer with per-tower parameter groups."""
        self.optimizer = build_optimizer(
            self.model, self.experiment_spec.train
        )
        self._build_tower_schedule_config()
        return self.optimizer

    def _build_tower_schedule_config(self):
        """Pre-compute per-tower schedule configs for training_step."""
        cfg = self.experiment_spec.train.optim

        self._tower_schedules = {
            'vision': {
                'lr': cfg.vision_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
            'text': {
                'lr': cfg.text_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
            'logit': {
                'lr': cfg.text_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
        }

    def on_train_start(self):
        """Training epoch start."""
        # Only update resume_step; do NOT call setup('fit') again as it causes
        # redundant dataloader creation which can lead to process explosion
        self.trainer.datamodule.resume_step = self.trainer.global_step

    def _forward_pass(self, batch):
        """Run forward pass."""
        image, text = batch[0], batch[1]
        return self.model(image=image, text=text)

    def _backward(self, outputs):
        """Compute loss from model outputs."""
        if len(outputs) == 3:
            image_features, text_features, logit_scale = outputs
            clip_loss = self.loss(image_features, text_features, logit_scale)
        else:
            image_features, text_features, logit_scale, logit_bias = outputs
            clip_loss = self.loss(
                image_features, text_features, logit_scale, logit_bias
            )
        return clip_loss, logit_scale

    def training_step(self, batch):
        """Training step."""
        # Handle both tensor and dict (for SigLIP2) image formats
        image = batch[0]
        batch_size = (
            image['pixel_values'].shape[0]
            if isinstance(image, dict)
            else image.shape[0]
        )
        outputs = self._forward_pass(batch)
        loss, logit_scale = self._backward(outputs)

        # Update per-tower learning rates
        for param_group in self.optimizer.param_groups:
            tower = param_group.get('_tower', 'text')
            sched = self._tower_schedules.get(
                tower, self._tower_schedules['text']
            )
            param_group['lr'] = compute_lr(
                self.global_step,
                sched['lr'],
                sched['warmup'],
                self.max_steps,
                sched['scheduler'],
            )

        vision_lr = self._tower_schedules['vision']['lr']
        text_lr = self._tower_schedules['text']['lr']
        current_vision_lr = compute_lr(
            self.global_step,
            vision_lr,
            self._tower_schedules['vision']['warmup'],
            self.max_steps,
            self._tower_schedules['vision']['scheduler'],
        )
        current_text_lr = compute_lr(
            self.global_step,
            text_lr,
            self._tower_schedules['text']['warmup'],
            self.max_steps,
            self._tower_schedules['text']['scheduler'],
        )
        self.log(
            "train/vision_lr", current_vision_lr,
            on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/text_lr", current_text_lr,
            on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/lr", current_text_lr,
            on_step=True, on_epoch=False, prog_bar=True
        )
        # Handle both scalar and dict loss
        # (ClipLoss returns dict with 'contrastive_loss')
        loss_value = (
            loss['contrastive_loss'] if isinstance(loss, dict) else loss
        )
        self.log(
            "train_loss", loss_value,
            on_step=True, on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=batch_size,
        )
        self.log(
            "train/logit_scale", logit_scale.item(),
            on_step=True, on_epoch=False, prog_bar=False
        )

        with torch.no_grad():
            self.model.logit_scale.clamp_(0, _MAX_LOGIT_SCALE)
        return loss_value

    def on_train_epoch_end(self):
        """Log training metrics to status.json."""
        average_train_loss = (
            self.trainer.logged_metrics["train_loss_epoch"].item()
        )

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self) -> None:
        """Reset evaluator and rebuild classifier for validation.

        Note: Classifier is rebuilt each validation because text encoder
        weights may have changed during training. If text encoder is frozen,
        the classifier could be cached for better performance.
        """
        if self.classnames is None:
            logging.warning(
                "Skipping zero-shot classifier build: no classnames "
                "configured. Set dataset.val.classnames_file or use "
                "val.type=classification."
            )
            self.classifier = None
            self.record = torch.tensor(
                [0, 0, 0], device="cpu", dtype=torch.long
            )
            return
        self.classifier = build_zero_shot_classifier(
            self.model, self.tokenizer, device=self.device,
            distributed=False,  # TODO: Check
            classnames=self.classnames,
            templates=self.templates
        )
        self.record = torch.tensor([0, 0, 0], device="cpu", dtype=torch.long)

    def validation_step(self, batch):
        """Run zero-shot validation on a batch."""
        if self.classifier is None:
            return
        image = batch[0]
        image_features = self.model(image=image)
        image_features = (
            image_features["image_features"]
            if isinstance(image_features, dict)
            else image_features[0]
        )
        logits = 100.0 * image_features @ self.classifier
        acc1, acc5 = accuracy(logits, batch[1], topk=(1, 5))
        acc1, acc5 = acc1.cpu(), acc5.cpu()
        batch_size = (
            image['pixel_values'].shape[0]
            if isinstance(image, dict)
            else image.shape[0]
        )
        self.record += torch.concat(
            [acc1, acc5, torch.tensor([batch_size], dtype=torch.long)]
        )

    def on_validation_epoch_end(self):
        """Compute and log final validation accuracy."""
        acc1, acc5, n = self.record.cpu().tolist()
        if n == 0:
            logging.warning(
                "Validation set was empty, skipping accuracy logging."
            )
            return
        acc1 = acc1 / n
        acc5 = acc5 / n
        self.log("current_epoch", self.current_epoch, sync_dist=True)
        self.log("val/accuracy_top1", acc1, sync_dist=True)
        self.log("val/accuracy_top5", acc5, sync_dist=True)
        logging.info(f"val/accuracy_top1: {acc1:.4f}")
        logging.info(f"val/accuracy_top5: {acc5:.4f}")

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val/accuracy_top1"] = str(acc1)
            self.status_logging_dict["val/accuracy_top5"] = str(acc5)
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    # Test methods (reuse validation logic)
    def on_test_epoch_start(self) -> None:
        """Test epoch start - reuse validation setup."""
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        """Test step - reuse validation step."""
        return self.validation_step(batch)

    def on_test_epoch_end(self):
        """Test epoch end - compute and log final accuracy."""
        acc1, acc5, n = self.record.cpu().tolist()
        if n == 0:
            logging.warning("Test set was empty, skipping accuracy logging.")
            return
        acc1 = acc1 / n
        acc5 = acc5 / n
        self.log("test/accuracy_top1", acc1, sync_dist=True)
        self.log("test/accuracy_top5", acc5, sync_dist=True)
        logging.info(f"test/accuracy_top1: {acc1:.4f}")
        logging.info(f"test/accuracy_top5: {acc5:.4f}")

        self.status_logging_dict = {}
        self.status_logging_dict["test/accuracy_top1"] = str(acc1)
        self.status_logging_dict["test/accuracy_top5"] = str(acc5)
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
