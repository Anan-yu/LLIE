"""Core modules for CAME-SAIGFormer.

The modules keep SAIGFormer's channel-transposed attention complexity while
conditioning color representation, illumination guidance, and feature routing
on camera/degradation observations.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CameraDescriptorEncoder(nn.Module):
    """Encode global image statistics and learned camera/degradation cues."""

    def __init__(self, in_channels: int = 3, descriptor_dim: int = 64):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # 3 RGB means + 3 RGB stds + luminance mean/std + gradient mean/std.
        self.cnn_proj = nn.Sequential(
            nn.Linear(64, descriptor_dim),
            nn.GELU(),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(10, descriptor_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(descriptor_dim * 2, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    @staticmethod
    def _global_statistics(x: torch.Tensor) -> torch.Tensor:
        rgb_mean = x.mean(dim=(2, 3))
        rgb_std = x.std(dim=(2, 3), unbiased=False)
        luminance = (
            0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        )
        lum_mean = luminance.mean(dim=(2, 3))
        lum_std = luminance.std(dim=(2, 3), unbiased=False)
        grad_h = F.pad(
            luminance[:, :, 1:, :] - luminance[:, :, :-1, :],
            (0, 0, 0, 1),
            mode="replicate",
        )
        grad_w = F.pad(
            luminance[:, :, :, 1:] - luminance[:, :, :, :-1],
            (0, 1, 0, 0),
            mode="replicate",
        )
        gradient = torch.sqrt(grad_h.square() + grad_w.square() + 1e-12)
        grad_mean = gradient.mean(dim=(2, 3))
        grad_std = gradient.std(dim=(2, 3), unbiased=False)
        return torch.cat(
            [rgb_mean, rgb_std, lum_mean, lum_std, grad_mean, grad_std], dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn_proj(self.encoder(x))
        statistic_features = self.stats_proj(self._global_statistics(x))
        return self.fusion(torch.cat([cnn_features, statistic_features], dim=1))


class InvertibleColorRotation(nn.Module):
    """Fixed luminance/chroma basis followed by descriptor-conditioned rotation."""

    def __init__(self, descriptor_dim: int = 64):
        super().__init__()
        color_basis = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [0.500, -0.419, -0.081],
                [-0.169, -0.331, 0.500],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("color_basis", color_basis)
        self.register_buffer("color_basis_inverse", torch.linalg.inv(color_basis))
        self.angle_predictor = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, 1),
        )
        nn.init.zeros_(self.angle_predictor[-1].weight)
        nn.init.zeros_(self.angle_predictor[-1].bias)

    def _rotation(self, descriptor: torch.Tensor) -> torch.Tensor:
        theta = self.angle_predictor(descriptor).squeeze(-1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        one = torch.ones_like(theta)
        zero = torch.zeros_like(theta)
        # The luminance axis is fixed; only the two chroma axes rotate.
        return torch.stack(
            [
                one,
                zero,
                zero,
                zero,
                cos_theta,
                -sin_theta,
                zero,
                sin_theta,
                cos_theta,
            ],
            dim=-1,
        ).reshape(-1, 3, 3)

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        basis = self.color_basis.to(dtype=x.dtype)
        base_color = torch.einsum("ij,bjhw->bihw", basis, x)
        rotation = self._rotation(descriptor).to(dtype=x.dtype)
        return torch.einsum("bij,bjhw->bihw", rotation, base_color)

    def inverse(
        self, transformed: torch.Tensor, descriptor: torch.Tensor
    ) -> torch.Tensor:
        rotation = self._rotation(descriptor).to(dtype=transformed.dtype)
        base_color = torch.einsum(
            "bij,bjhw->bihw", rotation.transpose(1, 2), transformed
        )
        basis_inverse = self.color_basis_inverse.to(dtype=transformed.dtype)
        return torch.einsum("ij,bjhw->bihw", basis_inverse, base_color)


class DescriptorConditionedSpline(nn.Module):
    """Strictly monotonic descriptor-conditioned piecewise-linear spline."""

    def __init__(self, descriptor_dim: int = 64, num_bins: int = 8):
        super().__init__()
        self.num_bins = num_bins
        self.param_proj = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_bins),
        )
        nn.init.zeros_(self.param_proj[-1].weight)
        nn.init.zeros_(self.param_proj[-1].bias)

    def _mapping_parameters(
        self, descriptor: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slopes = F.softplus(self.param_proj(descriptor)) + 0.5
        slopes = slopes / slopes.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        heights = slopes / self.num_bins
        cumulative_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0))
        cumulative_heights = cumulative_heights / cumulative_heights[
            :, -1:
        ].clamp_min(1e-6)
        return slopes.to(dtype=dtype), cumulative_heights.to(dtype=dtype)

    def forward(
        self, intensity: torch.Tensor, descriptor: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, height, width = intensity.shape
        _, cumulative_heights = self._mapping_parameters(
            descriptor, intensity.dtype
        )
        values = intensity.reshape(batch_size, -1).clamp(0, 1)
        bin_indices = (values * self.num_bins).long().clamp(
            0, self.num_bins - 1
        )
        left_x = bin_indices.to(values.dtype) / self.num_bins
        fraction = ((values - left_x) * self.num_bins).clamp(0, 1)
        left_y = torch.gather(cumulative_heights, 1, bin_indices)
        right_y = torch.gather(cumulative_heights, 1, bin_indices + 1)
        mapped = left_y + fraction * (right_y - left_y)
        return mapped.reshape(batch_size, 1, height, width)

    def inverse(
        self, compressed: torch.Tensor, descriptor: torch.Tensor
    ) -> torch.Tensor:
        """Invert the spline using its descriptor-conditioned cumulative heights."""
        batch_size, _, height, width = compressed.shape
        _, cumulative_heights = self._mapping_parameters(
            descriptor, compressed.dtype
        )
        values = compressed.reshape(batch_size, -1).clamp(0, 1)
        # Search is batched; gathered heights remain differentiable.
        bin_indices = torch.searchsorted(
            cumulative_heights.contiguous(), values.contiguous(), right=True
        ) - 1
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        left_y = torch.gather(cumulative_heights, 1, bin_indices)
        right_y = torch.gather(cumulative_heights, 1, bin_indices + 1)
        fraction = (values - left_y) / (right_y - left_y).clamp_min(1e-8)
        fraction = fraction.clamp(0, 1)
        restored = (bin_indices.to(values.dtype) + fraction) / self.num_bins
        return restored.reshape(batch_size, 1, height, width)


class CAMT(nn.Module):
    """Camera-Adaptive Manifold Transform with an explicit inverse."""

    def __init__(self, descriptor_dim: int = 64, num_bins: int = 8):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.descriptor_encoder = CameraDescriptorEncoder(3, descriptor_dim)
        self.color_rotation = InvertibleColorRotation(descriptor_dim)
        self.intensity_spline = DescriptorConditionedSpline(
            descriptor_dim, num_bins
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        descriptor = self.descriptor_encoder(x)
        rotated = self.color_rotation(x, descriptor)
        compressed = self.intensity_spline(rotated[:, :1], descriptor)
        manifold = torch.cat([compressed, rotated[:, 1:]], dim=1)
        return {
            "manifold": manifold,
            "descriptor": descriptor,
            "confidence": self.confidence_head(x),
        }

    def inverse(
        self, manifold: torch.Tensor, descriptor: torch.Tensor
    ) -> torch.Tensor:
        intensity = self.intensity_spline.inverse(manifold[:, :1], descriptor)
        rotated = torch.cat([intensity, manifold[:, 1:]], dim=1)
        return self.color_rotation.inverse(rotated, descriptor).clamp(0, 1)


class ManifoldAdaptiveIllumination(nn.Module):
    """Estimate spatially adaptive illumination in the learned manifold."""

    def __init__(
        self,
        in_channels: int = 3,
        descriptor_dim: int = 64,
        train_patch: int = 128,
        eps: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.train_patch = (
            list(train_patch) if isinstance(train_patch, (list, tuple))
            else [train_patch, train_patch]
        )
        conditioned_channels = in_channels + descriptor_dim
        self.offset_predict = nn.Sequential(
            nn.Conv2d(conditioned_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 4, 1),
        )
        self.modulation_predict = nn.Sequential(
            nn.Conv2d(conditioned_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 1),
        )

    @staticmethod
    def _get_center_grid(x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        coords_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(
            torch.meshgrid(coords_w, coords_h, indexing="xy"), dim=-1
        )
        normalizer = x.new_tensor([width, height])
        return coords / normalizer * 2 - 1

    def forward(
        self, manifold: torch.Tensor, descriptor: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, height, width = manifold.shape
        descriptor_map = descriptor[:, :, None, None].expand(
            -1, -1, height, width
        )
        conditioned_input = torch.cat([manifold, descriptor_map], dim=1)
        integrated = torch.cumsum(torch.cumsum(manifold, dim=-1), dim=-2)
        center_grid = self._get_center_grid(manifold).unsqueeze(0)
        normalizer = manifold.new_tensor(
            [self.train_patch[0] / width, self.train_patch[1] / height]
        ).view(1, 1, 1, 2)

        offsets = self.offset_predict(conditioned_input).permute(0, 2, 3, 1)
        offset_w, offset_h = torch.split(offsets, 2, dim=3)
        offset_w = offset_w - offset_w.mean(dim=-1, keepdim=True)
        offset_h = offset_h - offset_h.mean(dim=-1, keepdim=True)
        minimum_patch = 2
        offset_w_min = torch.minimum(
            offset_w.amin(dim=-1, keepdim=True),
            offset_w.new_full(
                offset_w.amin(dim=-1, keepdim=True).shape,
                -minimum_patch / self.train_patch[0],
            ),
        )
        offset_w_max = torch.maximum(
            offset_w.amax(dim=-1, keepdim=True),
            offset_w.new_full(
                offset_w.amax(dim=-1, keepdim=True).shape,
                minimum_patch / self.train_patch[0],
            ),
        )
        offset_h_min = torch.minimum(
            offset_h.amin(dim=-1, keepdim=True),
            offset_h.new_full(
                offset_h.amin(dim=-1, keepdim=True).shape,
                -minimum_patch / self.train_patch[1],
            ),
        )
        offset_h_max = torch.maximum(
            offset_h.amax(dim=-1, keepdim=True),
            offset_h.new_full(
                offset_h.amax(dim=-1, keepdim=True).shape,
                minimum_patch / self.train_patch[1],
            ),
        )

        area = (
            (offset_h_max - offset_h_min)
            * (offset_w_max - offset_w_min)
            * self.train_patch[0]
            * self.train_patch[1]
            / 4
        )
        area = area.reshape(batch_size, 1, height, width).clamp(1, height * width)
        scale = self.modulation_predict(conditioned_input)
        if self.eps:
            safe_sign = torch.where(
                scale >= 0, torch.ones_like(scale), -torch.ones_like(scale)
            )
            scale = torch.where(scale.abs() < self.eps, safe_sign * self.eps, scale)
        area = area * scale

        top_left = (
            torch.cat([offset_w_min, offset_h_min], dim=-1) * normalizer
            + center_grid
        ).clamp(-1, 1)
        top_right = (
            torch.cat([offset_w_max, offset_h_min], dim=-1) * normalizer
            + center_grid
        ).clamp(-1, 1)
        bottom_left = (
            torch.cat([offset_w_min, offset_h_max], dim=-1) * normalizer
            + center_grid
        ).clamp(-1, 1)
        bottom_right = (
            torch.cat([offset_w_max, offset_h_max], dim=-1) * normalizer
            + center_grid
        ).clamp(-1, 1)

        top_left_value = F.grid_sample(
            integrated, top_left, align_corners=True, padding_mode="border"
        )
        top_right_value = F.grid_sample(
            integrated, top_right, align_corners=True, padding_mode="border"
        )
        bottom_left_value = F.grid_sample(
            integrated, bottom_left, align_corners=True, padding_mode="border"
        )
        bottom_right_value = F.grid_sample(
            integrated, bottom_right, align_corners=True, padding_mode="border"
        )
        return (
            top_left_value
            + bottom_right_value
            - top_right_value
            - bottom_left_value
        ) / area


class ObservabilityEstimator(nn.Module):
    """Estimate recoverability from intensity, gradients, noise, and confidence."""

    def __init__(self):
        super().__init__()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)
        self.fusion = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        intensity = x.mean(dim=1, keepdim=True)
        diff_h = F.pad(
            x[:, :, 1:, :] - x[:, :, :-1, :],
            (0, 0, 0, 1),
            mode="replicate",
        )
        diff_w = F.pad(
            x[:, :, :, 1:] - x[:, :, :, :-1],
            (0, 1, 0, 0),
            mode="replicate",
        )
        gradient = torch.sqrt(diff_h.square() + diff_w.square() + 1e-8).mean(
            dim=1, keepdim=True
        )
        gradient = gradient / gradient.amax(dim=(2, 3), keepdim=True).clamp_min(
            1e-6
        )
        response = F.conv2d(
            intensity, self.laplacian_kernel.to(dtype=x.dtype), padding=1
        ).abs()
        noise_inverse = 1.0 - response / response.amax(
            dim=(2, 3), keepdim=True
        ).clamp_min(1e-6)
        return self.fusion(
            torch.cat([intensity, gradient, noise_inverse, confidence], dim=1)
        )


class ObservabilitySelectiveSkipFusion(nn.Module):
    """Refine encoder skips without changing the pretrained initial function.

    The decoder context, illumination guide, and observability map predict a
    correction for the encoder feature.  The correction head is zero
    initialized, so enabling this module on a pretrained model initially
    produces exactly the original skip feature.  Corrections are concentrated
    in low-observability regions where direct skips are most likely to carry
    amplified noise.
    """

    def __init__(
        self,
        dim: int,
        illumination_channels: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        hidden_dim = max(dim // 2, 8)
        input_dim = dim * 2 + illumination_channels + 1
        self.context = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                groups=hidden_dim,
                bias=bias,
            ),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.correction = nn.Conv2d(hidden_dim, dim, 1, bias=True)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(
        self,
        skip: torch.Tensor,
        decoder: torch.Tensor,
        illumination: torch.Tensor,
        observability: torch.Tensor,
    ) -> torch.Tensor:
        if skip.shape != decoder.shape:
            raise ValueError(
                "Skip and decoder features must have identical shapes, got "
                f"{tuple(skip.shape)} and {tuple(decoder.shape)}."
            )
        spatial_size = skip.shape[-2:]
        if illumination.shape[-2:] != spatial_size:
            illumination = F.interpolate(
                illumination,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        if observability.shape[-2:] != spatial_size:
            observability = F.interpolate(
                observability,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        observability = observability.clamp(0, 1)
        context = self.context(
            torch.cat(
                [skip, decoder, illumination, observability],
                dim=1,
            )
        )
        uncertainty = 1.0 - observability
        correction = uncertainty * self.gate(context) * self.correction(context)
        return skip + correction


class ReliabilityCalibratedSkipFusion(nn.Module):
    """Convexly replace unreliable skip content with decoder context.

    Unlike an unconstrained additive correction, this module predicts a
    bounded blend between the encoder skip and a decoder-conditioned
    candidate.  The blend is spatially modulated by calibrated reliability,
    so high-reliability detail is preserved while uncertain regions can draw
    more strongly on the denoised decoder representation.

    The global blend starts close to zero and the decoder projection starts as
    an identity mapping.  This keeps checkpoint fine-tuning stable while still
    providing non-zero gradients to every branch from the first update.
    """

    def __init__(
        self,
        dim: int,
        illumination_channels: int = 3,
        gate_groups: int = 8,
        initial_blend_logit: float = -5.0,
        bias: bool = False,
    ):
        super().__init__()
        if gate_groups < 1 or dim % gate_groups != 0:
            raise ValueError(
                "gate_groups must be a positive divisor of dim, got "
                f"dim={dim}, gate_groups={gate_groups}."
            )
        hidden_dim = max(dim // 2, 8)
        input_dim = dim * 2 + illumination_channels + 1
        self.dim = dim
        self.gate_groups = gate_groups
        self.channels_per_gate = dim // gate_groups
        self.context = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                groups=hidden_dim,
                bias=bias,
            ),
            nn.GELU(),
        )
        self.group_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, gate_groups, 1, bias=True),
            nn.Sigmoid(),
        )
        self.decoder_candidate = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                3,
                padding=1,
                groups=dim,
                bias=False,
            ),
            nn.Conv2d(dim, dim, 1, bias=bias),
        )
        self.context_correction = nn.Conv2d(hidden_dim, dim, 1, bias=True)
        self.blend_logit = nn.Parameter(torch.tensor(float(initial_blend_logit)))
        self._initialize_stable_candidate()

    def _initialize_stable_candidate(self) -> None:
        depthwise = self.decoder_candidate[0]
        pointwise = self.decoder_candidate[1]
        nn.init.zeros_(depthwise.weight)
        center = depthwise.kernel_size[0] // 2
        with torch.no_grad():
            depthwise.weight[:, 0, center, center] = 1.0
        nn.init.zeros_(pointwise.weight)
        with torch.no_grad():
            diagonal = torch.arange(self.dim)
            pointwise.weight[diagonal, diagonal, 0, 0] = 1.0
        if pointwise.bias is not None:
            nn.init.zeros_(pointwise.bias)
        nn.init.zeros_(self.context_correction.weight)
        nn.init.zeros_(self.context_correction.bias)

    @property
    def blend_strength(self) -> torch.Tensor:
        """Return the bounded global replacement strength."""
        return torch.sigmoid(self.blend_logit)

    def forward(
        self,
        skip: torch.Tensor,
        decoder: torch.Tensor,
        illumination: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if skip.shape != decoder.shape:
            raise ValueError(
                "Skip and decoder features must have identical shapes, got "
                f"{tuple(skip.shape)} and {tuple(decoder.shape)}."
            )
        spatial_size = skip.shape[-2:]
        if illumination.shape[-2:] != spatial_size:
            illumination = F.interpolate(
                illumination,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        if reliability.shape[-2:] != spatial_size:
            reliability = F.interpolate(
                reliability,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        reliability = reliability.clamp(0, 1)
        context = self.context(
            torch.cat([skip, decoder, illumination, reliability], dim=1)
        )
        group_gate = self.group_gate(context).repeat_interleave(
            self.channels_per_gate,
            dim=1,
        )
        candidate = self.decoder_candidate(decoder)
        candidate = candidate + self.context_correction(context)
        blend = (
            self.blend_strength
            * (1.0 - reliability)
            * group_gate
        )
        return skip + blend * (candidate - skip)


class ObservabilityConditionedAttention(nn.Module):
    """Channel-transposed attention with spatial illumination modulation."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias
        )
        self.illum_spatial_gate = nn.Conv2d(3, dim, 1, bias=True)
        self.illum_head_modulator = nn.Sequential(
            nn.Conv2d(3, num_heads, 1, bias=True),
            nn.Sigmoid(),
        )
        self.local_detail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        illum_guide: torch.Tensor,
        observability: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        spatial_gate = 0.5 + torch.sigmoid(self.illum_spatial_gate(illum_guide))
        q = q * spatial_gate
        k = k * spatial_gate

        q = rearrange(
            q, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )
        k = rearrange(
            k, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )
        v = rearrange(
            v, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        head_weight = self.illum_head_modulator(illum_guide).mean(dim=(2, 3))
        adaptive_temperature = self.temperature.unsqueeze(0) * (
            0.5 + head_weight[:, :, None, None]
        )
        attention = (q @ k.transpose(-2, -1)) * adaptive_temperature
        attention = attention.softmax(dim=-1)
        global_output = rearrange(
            attention @ v,
            "b head c (h w) -> b (head c) h w",
            h=height,
            w=width,
        )
        local_output = self.local_detail(x)
        route = observability.clamp(0, 1)
        output = route * local_output + (1 - route) * global_output
        return self.project_out(output)


class CounterfactualDisentanglement(nn.Module):
    """Intervene in a learned degradation subspace and re-encode content."""

    def __init__(self, dim: int, num_interventions: int = 2):
        super().__init__()
        if num_interventions < 1:
            raise ValueError("num_interventions must be at least 1")
        self.num_interventions = num_interventions
        self.content_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.degradation_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 1, bias=False),
        )
        self.intervention_vectors = nn.Parameter(
            torch.randn(num_interventions, dim) * 0.01
        )
        self.recombine = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.content_reencoder = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        if num_interventions == 1:
            strengths = torch.tensor([0.5])
        else:
            strengths = torch.linspace(-0.5, 0.5, num_interventions)
            strengths[strengths.abs() < 1e-6] = 0.25
        self.register_buffer("intervention_strengths", strengths)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        content = self.content_proj(x)
        degradation = self.degradation_proj(x)
        result: Dict[str, object] = {
            "content": content,
            "degradation": degradation,
            "cf_features": [],
            "cf_content_features": [],
            "intervention_vectors": self.intervention_vectors,
        }
        if self.training:
            directions = F.normalize(
                self.intervention_vectors, dim=-1, eps=1e-6
            )
            scale = degradation.std(
                dim=(2, 3), keepdim=True, unbiased=False
            ).clamp_min(1e-4)
            cf_features: List[torch.Tensor] = []
            cf_content_features: List[torch.Tensor] = []
            for index in range(self.num_interventions):
                direction = directions[index].view(1, -1, 1, 1)
                strength = self.intervention_strengths[index].to(
                    dtype=degradation.dtype
                )
                intervened_degradation = (
                    degradation + strength * direction * scale
                )
                cf_feature = self.recombine(
                    torch.cat([content, intervened_degradation], dim=1)
                )
                cf_features.append(cf_feature)
                cf_content_features.append(self.content_reencoder(cf_feature))
            result["cf_features"] = cf_features
            result["cf_content_features"] = cf_content_features
        result["output"] = self.recombine(
            torch.cat([content, degradation], dim=1)
        )
        return result


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = x.var(1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + 1e-5)
        return (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class DualGatedFFN(nn.Module):
    def __init__(
        self, dim: int, expansion: float = 2.66, bias: bool = False
    ):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj_1 = nn.Conv2d(dim, hidden, 1, bias=bias)
        self.proj_2 = nn.Conv2d(dim, hidden, 1, bias=bias)
        self.out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.proj_1(x), self.proj_2(x)
        return self.out(
            first * torch.sigmoid(second) + second * F.gelu(first)
        )


class CAMETransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = ObservabilityConditionedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = DualGatedFFN(dim, ffn_expansion, bias)

    def forward(
        self,
        x: torch.Tensor,
        illum_guide: torch.Tensor,
        observability: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), illum_guide, observability)
        return x + self.ffn(self.norm2(x))


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self, in_c: int = 3, embed_dim: int = 48, bias: bool = False
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
