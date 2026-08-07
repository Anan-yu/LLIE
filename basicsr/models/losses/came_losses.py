"""Auxiliary objectives for CAME-SAIGFormer.

Standard image reconstruction losses remain owned by ``pixel_opt``. This module
only computes CAME-specific auxiliary objectives.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentInvarianceLoss(nn.Module):
    """Keep re-encoded content invariant to degradation interventions."""

    def __init__(self, loss_weight: float = 0.05):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        content: torch.Tensor,
        counterfactual_content: List[torch.Tensor],
    ) -> torch.Tensor:
        if not counterfactual_content:
            return content.new_zeros(())
        reference = content.detach()
        loss = torch.stack(
            [F.l1_loss(features, reference) for features in counterfactual_content]
        ).mean()
        return self.loss_weight * loss


class DisentangleLoss(nn.Module):
    """Encourage content and degradation embeddings to be orthogonal."""

    def __init__(self, loss_weight: float = 0.01):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self, content: torch.Tensor, degradation: torch.Tensor
    ) -> torch.Tensor:
        batch_size = content.shape[0]
        content_flat = content.reshape(batch_size, -1)
        degradation_flat = degradation.reshape(batch_size, -1)
        similarity = F.cosine_similarity(
            content_flat, degradation_flat, dim=-1, eps=1e-8
        )
        return self.loss_weight * similarity.abs().mean()


class CycleConsistencyLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.02):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_weight * F.l1_loss(original, reconstructed)


class InterventionDiversityLoss(nn.Module):
    """Penalize off-diagonal similarity between intervention directions."""

    def __init__(self, loss_weight: float = 0.005):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, intervention_vectors: torch.Tensor) -> torch.Tensor:
        count = intervention_vectors.shape[0]
        if count <= 1:
            return intervention_vectors.new_zeros(())
        directions = F.normalize(intervention_vectors, dim=-1, eps=1e-6)
        gram = directions @ directions.transpose(0, 1)
        off_diagonal = ~torch.eye(
            count, dtype=torch.bool, device=intervention_vectors.device
        )
        return self.loss_weight * gram[off_diagonal].square().mean()


class RAEDLoss(nn.Module):
    """Reference-ambiguity-aware exposure distribution objective."""

    def __init__(
        self,
        loss_weight: float = 0.03,
        num_quantiles: int = 5,
        ordering_margin: float = 0.05,
        ordering_tolerance: float = 1e-3,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.ordering_margin = ordering_margin
        self.ordering_tolerance = ordering_tolerance
        self.register_buffer(
            "quantile_levels", torch.linspace(0.1, 0.9, num_quantiles)
        )

    @staticmethod
    def _luminance(x: torch.Tensor) -> torch.Tensor:
        return (
            0.2126 * x[:, 0:1]
            + 0.7152 * x[:, 1:2]
            + 0.0722 * x[:, 2:3]
        )

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        prediction_luminance = self._luminance(prediction)
        target_luminance = self._luminance(target)
        batch_size = prediction.shape[0]
        # torch.quantile has stable gradients in float32 and is not implemented
        # for every half-precision backend.
        prediction_flat = prediction_luminance.float().reshape(batch_size, -1)
        target_flat = target_luminance.float().reshape(batch_size, -1)
        levels = self.quantile_levels.to(dtype=prediction_flat.dtype)
        prediction_quantiles = torch.quantile(
            prediction_flat,
            levels,
            dim=1,
        )
        target_quantiles = torch.quantile(
            target_flat,
            levels,
            dim=1,
        )
        quantile_loss = F.l1_loss(prediction_quantiles, target_quantiles)

        prediction_patches = F.adaptive_avg_pool2d(
            prediction_luminance, 8
        ).reshape(batch_size, -1)
        target_patches = F.adaptive_avg_pool2d(
            target_luminance, 8
        ).reshape(batch_size, -1)
        prediction_difference = (
            prediction_patches.unsqueeze(2) - prediction_patches.unsqueeze(1)
        )
        target_difference = (
            target_patches.unsqueeze(2) - target_patches.unsqueeze(1)
        )
        patch_count = prediction_patches.shape[1]
        non_diagonal = ~torch.eye(
            patch_count, dtype=torch.bool, device=prediction.device
        ).unsqueeze(0)
        valid_pairs = non_diagonal & (
            target_difference.abs() > self.ordering_tolerance
        )
        if valid_pairs.any():
            ordering = F.relu(
                self.ordering_margin
                - prediction_difference * torch.sign(target_difference)
            )
            ordering_loss = ordering[valid_pairs].mean()
        else:
            ordering_loss = prediction.new_zeros(())
        return self.loss_weight * (quantile_loss + 0.3 * ordering_loss)


class ObservabilityCalibrationLoss(nn.Module):
    """Calibrate recoverability against paired restoration difficulty.

    The target combines local photometric error and gradient degradation.
    Regions that differ strongly from the paired reference receive lower
    reliability, while already faithful regions remain close to one.  The
    target is detached: it supervises observability without creating a shortcut
    through the input or reference tensors.
    """

    def __init__(
        self,
        loss_weight: float = 0.01,
        temperature: float = 0.15,
        kernel_size: int = 5,
        gradient_weight: float = 0.5,
        minimum_reliability: float = 0.05,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0 <= minimum_reliability < 1:
            raise ValueError("minimum_reliability must be in [0, 1).")
        self.loss_weight = float(loss_weight)
        self.temperature = float(temperature)
        self.kernel_size = int(kernel_size)
        self.gradient_weight = float(gradient_weight)
        self.minimum_reliability = float(minimum_reliability)

    @staticmethod
    def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
        gradient_h = F.pad(
            image[:, :, 1:, :] - image[:, :, :-1, :],
            (0, 0, 0, 1),
            mode="replicate",
        )
        gradient_w = F.pad(
            image[:, :, :, 1:] - image[:, :, :, :-1],
            (0, 1, 0, 0),
            mode="replicate",
        )
        return torch.sqrt(
            gradient_h.square() + gradient_w.square() + 1e-8
        ).mean(dim=1, keepdim=True)

    def reliability_target(
        self,
        input_image: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        photometric_error = (input_image - ground_truth).abs().mean(
            dim=1,
            keepdim=True,
        )
        input_gradient = self._gradient_magnitude(input_image)
        target_gradient = self._gradient_magnitude(ground_truth)
        gradient_error = (input_gradient - target_gradient).abs()
        difficulty = photometric_error + self.gradient_weight * gradient_error
        padding = self.kernel_size // 2
        difficulty = F.pad(
            difficulty,
            (padding, padding, padding, padding),
            mode="replicate",
        )
        difficulty = F.avg_pool2d(
            difficulty,
            kernel_size=self.kernel_size,
            stride=1,
        )
        reliability = torch.exp(-difficulty / self.temperature)
        return reliability.clamp(self.minimum_reliability, 1.0).detach()

    def forward(
        self,
        observability: torch.Tensor,
        input_image: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        target = self.reliability_target(input_image, ground_truth)
        if observability.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target,
                size=observability.shape[-2:],
                mode="area",
            )
        loss = F.smooth_l1_loss(observability, target)
        return self.loss_weight * loss


class ObservabilitySmoothLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.01):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self, observability: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        gradient_h = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        gradient_w = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        weight_h = torch.exp(-10 * gradient_h)
        weight_w = torch.exp(-10 * gradient_w)
        observability_h = (
            observability[:, :, 1:, :] - observability[:, :, :-1, :]
        ).abs()
        observability_w = (
            observability[:, :, :, 1:] - observability[:, :, :, :-1]
        ).abs()
        return self.loss_weight * (
            (observability_h * weight_h).mean()
            + (observability_w * weight_w).mean()
        )


class CAMELoss(nn.Module):
    """Compute CAME auxiliary losses with stable, explicit output keys."""

    def __init__(
        self,
        content_inv_weight: float = 0.05,
        cycle_weight: float = 0.02,
        raed_weight: float = 0.03,
        obs_smooth_weight: float = 0.01,
        disentangle_weight: float = 0.01,
        intervention_diversity_weight: float = 0.005,
        obs_calibration_weight: float = 0.0,
        obs_calibration_temperature: float = 0.15,
        use_raed: bool = True,
        use_cycle: bool = True,
        use_disentangle: bool = True,
        use_intervention_diversity: bool = True,
        use_observability_calibration: bool = False,
    ):
        super().__init__()
        self.use_raed = use_raed
        self.use_cycle = use_cycle
        self.use_disentangle = use_disentangle
        self.use_intervention_diversity = use_intervention_diversity
        self.use_observability_calibration = use_observability_calibration
        self.content_inv_loss = ContentInvarianceLoss(content_inv_weight)
        self.cycle_loss = CycleConsistencyLoss(cycle_weight)
        self.raed_loss = RAEDLoss(raed_weight)
        self.obs_smooth_loss = ObservabilitySmoothLoss(obs_smooth_weight)
        self.obs_calibration_loss = ObservabilityCalibrationLoss(
            loss_weight=obs_calibration_weight,
            temperature=obs_calibration_temperature,
        )
        self.disentangle_loss = DisentangleLoss(disentangle_weight)
        self.intervention_diversity_loss = InterventionDiversityLoss(
            intervention_diversity_weight
        )

    def forward(
        self,
        model_output: Dict[str, object],
        ground_truth: torch.Tensor,
        input_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        output = model_output["output"]
        zero = output.new_zeros(())
        content: Optional[torch.Tensor] = model_output.get("content_features")
        counterfactual_content = model_output.get("cf_content_features", [])
        if content is not None and counterfactual_content:
            content_invariance = self.content_inv_loss(
                content, counterfactual_content
            )
        else:
            content_invariance = zero

        cycle_reconstruction = model_output.get("cycle_recon")
        if self.use_cycle and cycle_reconstruction is not None:
            cycle = self.cycle_loss(input_image, cycle_reconstruction)
        else:
            cycle = zero

        raed = self.raed_loss(output, ground_truth) if self.use_raed else zero
        observability = model_output.get("observability")
        observability_smoothness = (
            self.obs_smooth_loss(observability, input_image)
            if observability is not None
            else zero
        )
        observability_calibration = (
            self.obs_calibration_loss(
                observability,
                input_image,
                ground_truth,
            )
            if self.use_observability_calibration
            and observability is not None
            else zero
        )
        degradation: Optional[torch.Tensor] = model_output.get(
            "degradation_features"
        )
        if self.use_disentangle and content is not None and degradation is not None:
            disentangle = self.disentangle_loss(content, degradation)
        else:
            disentangle = zero

        intervention_vectors: Optional[torch.Tensor] = model_output.get(
            "intervention_vectors"
        )
        if (
            self.use_intervention_diversity
            and intervention_vectors is not None
        ):
            diversity = self.intervention_diversity_loss(intervention_vectors)
        else:
            diversity = zero

        losses = {
            "l_content_inv": content_invariance,
            "l_cycle": cycle,
            "l_raed": raed,
            "l_obs_smooth": observability_smoothness,
            "l_obs_calibration": observability_calibration,
            "l_disentangle": disentangle,
            "l_intervention_diversity": diversity,
        }
        losses["l_came_total"] = torch.stack(list(losses.values())).sum()
        return losses
