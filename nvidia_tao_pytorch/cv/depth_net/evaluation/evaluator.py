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

""" Depth Net Evaluator in distributed mode. """

from typing import Tuple, List, Dict

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def align_depth_least_square(
    gt: Tensor,
    pred: Tensor,
):
    """Align depth using least square method.
    Args:
        gt (torch.Tensor): Ground truth disparity/depth tensor
        pred (torch.Tensor): Predicted disparity/depth tensor
    Returns:
        aligned_pred (torch.Tensor): Aligned disparity/depth tensor
    """
    ori_shape = pred.shape  # input shape
    gt = gt.squeeze()  # [H, W]
    pred = pred.squeeze()
    assert (
        gt.shape == pred.shape
    ), f"GT shape: {gt.shape}, Pred shape: {pred.shape} are not matched."

    gt_masked = gt.reshape((-1, 1))
    pred_masked = pred.reshape((-1, 1))

    # numpy solver
    _ones = torch.ones_like(pred_masked)
    A = torch.cat([pred_masked, _ones], dim=-1)
    X = torch.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)
    return aligned_pred


def _delta_log_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor
    Returns:
        d1 (torch.Tensor): Delta 1
        d2 (torch.Tensor): Delta 2
        d3 (torch.Tensor): Delta 3
    """
    _check_same_shape(preds, target)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo

    thresh = torch.max((target / preds), (preds / target))

    d1 = torch.sum(thresh < 1.25, dim=0)
    d2 = torch.sum(thresh < 1.25 ** 2, dim=0)
    d3 = torch.sum(thresh < 1.25 ** 3, dim=0)

    return d1, d2, d3, target.shape[0]


def _rmse_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_abs_error (torch.Tensor): Sum of root mean squared error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    mse = torch.nn.functional.mse_loss(preds[mask], target[mask])
    rmse = torch.sqrt(mse)
    return rmse, target.shape[0]


def _rmse_log_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_log_error (torch.Tensor): Sum of log error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    mse_log = torch.nn.functional.mse_loss(torch.log(preds[mask]), torch.log(target[mask]))
    rmse_log = torch.sqrt(mse_log)
    return rmse_log, target.shape[0]


def _abs_rel_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_abs_error (torch.Tensor): Sum of absolute relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo

    sum_abs_error = torch.sum(torch.abs(preds[mask] - target[mask]) / target[mask], dim=0)
    return sum_abs_error, target.shape[0]


def _sq_rel_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_sq_error (torch.Tensor): Sum of squared relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    rel_sq_error = torch.sum(torch.pow(
        (preds[mask] - target[mask]), 2)) / torch.sum(torch.abs(target[mask] - torch.mean(target[mask])), dim=0, keepdims=True)
    return rel_sq_error, target.shape[0]


def _epe_error(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Calculates and returns EPE error and other related stereo metrics.

    This private helper function computes several key metrics for stereo
    matching, including End-Point Error (EPE), D1-metric, and bad-pixel
    ratios at different thresholds. It handles batch processing and
    converts input tensors to floating point numbers.

    Args:
        preds (torch.Tensor): The predicted disparity maps. Expected shape is `(N, C, H, W)`.
        target (torch.Tensor): The ground truth disparity maps. Expected shape
            is the same as `preds`.
        max_disparity (int, optional): The maximum possible disparity value.
            Used to create a valid mask for the ground truth. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]: A tuple containing the
            sums of the following metrics across the batch:
            - sum_d1 (torch.Tensor): The sum of the D1-metric values.
            - sum_bp1 (torch.Tensor): The sum of the bad-pixel ratios with a threshold of 1.
            - sum_bp2 (torch.Tensor): The sum of the bad-pixel ratios with a threshold of 2.
            - sum_bp3 (torch.Tensor): The sum of the bad-pixel ratios with a threshold of 3.
            - sum_epe_val (torch.Tensor): The sum of the mean EPE values.
            - num_obs (int): The number of observations (i.e., the batch size).
    """
    _check_same_shape(preds, target)
    mask = (target > 0.) & (target < max_disparity)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    epe = torch.abs(preds - target)
    B = target.shape[0]
    epe_mean = (epe[mask]).reshape(B, -1).sum(dim=-1) / (mask.reshape(B, -1).sum(dim=-1) + 1e-8)
    d1, bp1, bp2, bp3, epe_val = 0.0, 0.0, 0.0, 0.0, 0.0
    # assuming batch is not 1
    for i in range(B):
        d1 += (((epe[i][mask[i]] > 3) & (epe[i][mask[i]] / target[i][mask[i]] > 0.05)) + 0.0).mean()
        bp1 += (((epe[i] > 1)[mask[i]]) + 0.0).mean()
        bp2 += (((epe[i] > 2)[mask[i]]) + 0.0).mean()
        bp3 += (((epe[i] > 3)[mask[i]]) + 0.0).mean()
        epe_val += epe_mean
    return d1, bp1, bp2, bp3, epe_val, target.shape[0]


class DepthMetric(Metric):
    """Depth Evaluation Metric Class."""

    def __init__(self, model_type: str, align_gt: bool = True, num_outputs: int = 1,
                 min_depth: float = 0.001, max_depth: float = 10, max_disparity=416, **kwargs):
        """Initialize for Depth Metric Class.
        Args:
            align_gt (bool): Whether to align the ground truth disparity/depth tensor.
            num_outputs (int): Number of outputs in multioutput setting.
            min_depth (float): Minimum depth value.
            max_depth (float): Maximum depth value.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.align_gt = align_gt
        self.num_outputs = num_outputs
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.model_type = model_type
        self.max_disparity = max_disparity
        self.add_state("sum_abs_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_sq_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse_log", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d1", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d3", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_bp1", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_bp2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_bp3", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_epe", default=torch.zeros(num_outputs), dist_reduce_fx="sum")

        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        # track static values
        self.d1 = 0.
        self.d2 = 0.
        self.d3 = 0.
        self.bp1 = 0.
        self.bp2 = 0.
        self.bp3 = 0.
        self.epe = 0.
        self.kwargs = kwargs

    def update_mono(self, post_processed_results: List[Dict]) -> None:
        """Updates and accumulates monocular depth estimation metrics.

        This function processes a list of post-processed results from a monocular
        depth estimation model. It aligns predictions to ground truth depths if
        `self.align_gt` is True, then calculates and accumulates various metrics
        including absolute relative error, squared relative error, RMSE, log RMSE,
        and Delta accuracy metrics.

        Args:
            post_processed_results (List[Dict]): A list of dictionaries, where
                each dictionary contains the results for a single image from the
                model. Each dictionary is expected to have the following keys:
                - 'depth_pred' (torch.Tensor): The predicted depth map.
                - 'disp_gt' (torch.Tensor): The ground truth disparity map.
                - 'valid_mask' (torch.Tensor): A boolean mask indicating valid
                pixels for metric calculation.

        Returns:
            None: This function modifies the object's state in-place by updating
                the accumulated metric sums and the total number of observations.
        """
        pred_list = []
        target_list = []
        for result in post_processed_results:
            pred_i = result['depth_pred']
            gt_i = result['disp_gt']
            valid_mask_i = result['valid_mask']
            if self.align_gt:
                pred_aligned = align_depth_least_square(
                    gt=gt_i[valid_mask_i],
                    pred=pred_i[valid_mask_i],
                )
                pred_aligned = torch.clip(pred_aligned, min=1e-8, max=None)  # avoid 0 disparity
                target = torch.clip(gt_i[valid_mask_i], min=1e-8, max=None)  # avoid 0 disparity
            else:
                pred_aligned = torch.clip(pred_i[valid_mask_i], min=self.min_depth, max=self.max_depth)  # avoid 0 disparity
                target = torch.clip(gt_i[valid_mask_i], min=self.min_depth, max=self.max_depth)
            pred_list.append(pred_aligned)
            target_list.append(target)
        pred_aligned = torch.concat(pred_list, dim=0)
        target = torch.concat(target_list, dim=0)
        sum_abs_rel, num_obs = _abs_rel_update(pred_aligned, target)

        sum_sq_rel, _ = _sq_rel_update(pred_aligned, target)
        sum_rmse, _ = _rmse_update(pred_aligned, target)
        sum_rmse_log, _ = _rmse_log_update(pred_aligned, target)
        sum_d1, sum_d2, sum_d3, _ = _delta_log_update(pred_aligned, target)

        self.sum_abs_rel += sum_abs_rel
        self.sum_sq_rel += sum_sq_rel
        self.sum_rmse += sum_rmse
        self.sum_rmse_log += sum_rmse_log
        self.sum_d1 += sum_d1
        self.sum_d2 += sum_d2
        self.sum_d3 += sum_d3

        self.total += num_obs

    def update_stereo(self, preds: Tensor, target: Tensor) -> None:
        """Updates the metric results for a stereo estimation model.

        This function calculates various stereo metrics such as D1-metric, EPE,
        and different relative and squared errors. It accumulates these values
        into the class attributes for later aggregation.

        Args:
            preds (torch.Tensor): The predicted disparity maps. The tensor is
                expected to have a shape of `(N, C, H, W)`, where N is the batch
                size, C is the number of output channels (usually 1 for disparity),
                and H and W are the height and width of the maps. The values
                should represent the estimated disparity.
            target (torch.Tensor): The ground truth disparity maps. The tensor
                should have the same shape as `preds` and contain the true
                disparity values. It's important that this tensor is correctly
                aligned with the `preds` tensor.

        Returns:
            None: This function does not return any value. It updates the internal
                state of the object by accumulating the calculated metrics.
        """
        sum_d1, sum_bp1, sum_bp2, sum_bp3, sum_epe_val, num_obs = _epe_error(
            preds, target, max_disparity=self.max_disparity)
        sum_sq_rel, _ = _sq_rel_update(preds, target, max_disparity=self.max_disparity)
        sum_rmse, _ = _rmse_update(preds, target, max_disparity=self.max_disparity)
        sum_rmse_log, _ = _rmse_log_update(preds, target, max_disparity=self.max_disparity)
        sum_abs_rel, _ = _abs_rel_update(preds, target, max_disparity=self.max_disparity)
        self.sum_abs_rel += sum_abs_rel
        self.sum_sq_rel += sum_sq_rel
        self.sum_rmse += sum_rmse
        self.sum_rmse_log += sum_rmse_log
        self.sum_d1 += sum_d1
        self.sum_bp1 += sum_bp1
        self.sum_bp2 += sum_bp2
        self.sum_bp3 += sum_bp3
        self.sum_epe += sum_epe_val
        self.total += num_obs
        self.d1 = sum_d1
        self.bp1 = sum_bp1
        self.bp2 = sum_bp2
        self.bp3 = sum_bp3
        self.epe = sum_epe_val

    def update(self, preds: Tensor = None, target: Tensor = None, post_processed_results=None):
        """Updates the metric results based on the model type.

        This function serves as a dispatcher to the appropriate metric update
        method (`update_stereo` or `update_mono`) based on the `self.model_type`
        attribute. It is designed to handle different types of models (e.g.,
        stereo and monocular depth estimation) by routing the input data to the
        correct processing logic.

        Args:
            preds (torch.Tensor, optional): The predicted disparity or depth maps.
                This argument is used for stereo models. Defaults to None.
            target (torch.Tensor, optional): The ground truth disparity or depth maps.
                This argument is used for stereo models. Defaults to None.
            post_processed_results (list of dict, optional): A list of dictionaries
                containing post-processed results. This argument is used for
                monocular models. Defaults to None.

        Raises:
            IndexError: If `self.model_type` does not match any of the
                implemented model types ('foundationstereo', 'metricdepthanything',
                'relativedepthanything').
        """
        if self.model_type.lower() == 'foundationstereo':
            self.update_stereo(preds, target)
        elif self.model_type.lower() in ['metricdepthanything', 'relativedepthanything']:
            self.update_mono(post_processed_results)
        else:
            raise IndexError('evaluation metric not implemented for model: {self.model_type}')

    def get_single_update(self):
        """Retrieves the most recently calculated metric values as a dictionary.

        This function returns the metric values from the latest batch processed
        by the `update` method. The values are converted to standard Python
        numbers using `.item()` for easy serialization and use.

        Returns:
            dict: A dictionary containing the following scalar metric values
                from the last processed batch:
                - 'd1' (float): The D1-metric.
                - 'd2' (float): The Delta-2 metric.
                - 'd3' (float): The Delta-3 metric.
                - 'bp1' (float): The bad-pixel metric at threshold 1.
                - 'bp2' (float): The bad-pixel metric at threshold 2.
                - 'bp3' (float): The bad-pixel metric at threshold 3.
                - 'epe' (float): The End-Point Error.
                - 'abs_rel' (float): The absolute relative error.
                - 'sq_rel' (float): The squared relative error.
                - 'rmse' (float): The Root Mean Squared Error.
                - 'rmse_log' (float): The Root Mean Squared Logarithmic Error.
        """
        return {"d1": self.d1.item(), "d2": self.d2.item(), "d3": self.d3.item(),
                "bp1": self.bp1.item(), "bp2": self.bp2.item(), "bp3": self.bp3.item(),
                "epe": self.epe.item(), "abs_rel": self.abs_rel.item(), "sq_rel": self.sq_rel.item(),
                "rmse": self.rmse.item(), 'rmse_log': self.rmse_log.item()}

    def compute(self):
        """Computes and returns the final depth evaluation metrics.

        This function aggregates the accumulated metric sums (`self.sum_*`)
        and divides them by the total number of observations (`self.total`)
        to compute the final, average metrics over the entire dataset.

        Returns:
            dict: A dictionary containing the following aggregated scalar
                metric values:
                - 'd1' (float): The final D1-metric.
                - 'd2' (float): The final Delta-2 metric.
                - 'd3' (float): The final Delta-3 metric.
                - 'bp1' (float): The final bad-pixel metric at threshold 1.
                - 'bp2' (float): The final bad-pixel metric at threshold 2.
                - 'bp3' (float): The final bad-pixel metric at threshold 3.
                - 'epe' (float): The final End-Point Error.
                - 'abs_rel' (float): The final absolute relative error.
                - 'sq_rel' (float): The final squared relative error.
                - 'rmse' (float): The final Root Mean Squared Error.
                - 'rmse_log' (float): The final Root Mean Squared Logarithmic Error.
        """
        abs_rel = self.sum_abs_rel / self.total
        sq_rel = self.sum_sq_rel / self.total
        rmse = torch.sqrt(self.sum_rmse / self.total)
        rmse_log = torch.sqrt(self.sum_rmse_log / self.total)
        d1 = self.sum_d1 / self.total
        d2 = self.sum_d2 / self.total
        d3 = self.sum_d3 / self.total
        bp1 = self.sum_bp1 / self.total
        bp2 = self.sum_bp2 / self.total
        bp3 = self.sum_bp3 / self.total
        epe = self.sum_epe / self.total

        return {"d1": d1.item(), "d2": d2.item(), "d3": d3.item(),
                "bp1": bp1.item(), "bp2": bp2.item(), "bp3": bp3.item(),
                "epe": epe.item(), "abs_rel": abs_rel.item(), "sq_rel": sq_rel.item(),
                "rmse": rmse.item(), 'rmse_log': rmse_log.item()}
