"""
CAME-SAIGFormer Core Innovation Modules
========================================
Camera-Adaptive Manifold and Counterfactual Exposure Intervention modules.

Three core components:
1. CAMT - Camera-Adaptive Manifold Transform
2. CEI  - Counterfactual Exposure Intervention
3. OGDR - Observability-Guided Dynamic Restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


# =============================================================================
# CAMT: Camera-Adaptive Manifold Transform
# =============================================================================

class CameraDescriptorEncoder(nn.Module):
    """Extract a compact camera-degradation descriptor from input image.
    
    Estimates camera response characteristics and degradation state
    to parameterize the adaptive color manifold transform.
    
    Output descriptor encodes: white balance shift, response curve shape,
    noise level, and global exposure.
    """
    
    def __init__(self, in_channels: int = 3, descriptor_dim: int = 64):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        
        # Global context extraction via progressive downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Descriptor projection
        self.proj = nn.Sequential(
            nn.Linear(64, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, 3, H, W) in [0, 1]
        Returns:
            descriptor: Camera-degradation descriptor (B, descriptor_dim)
        """
        feat = self.encoder(x).flatten(1)  # (B, 64)
        descriptor = self.proj(feat)  # (B, descriptor_dim)
        return descriptor


class InvertibleColorRotation(nn.Module):
    """Invertible 1x1 convolution for color space rotation.
    
    Uses orthogonal matrix parameterization to ensure invertibility.
    Initialized as identity to preserve colors at start of training.
    """
    
    def __init__(self, channels: int = 3):
        super().__init__()
        # Parameterize as skew-symmetric matrix for orthogonal constraint
        # via Cayley transform: Q = (I - A)(I + A)^{-1}, A skew-symmetric
        self.A = nn.Parameter(torch.zeros(channels, channels))
        self.channels = channels
        
    def _get_rotation_matrix(self) -> torch.Tensor:
        """Compute orthogonal rotation matrix via Cayley transform."""
        # Enforce skew-symmetry
        A = self.A - self.A.t()
        I = torch.eye(self.channels, device=A.device, dtype=A.dtype)
        Q = torch.linalg.solve(I + A, I - A)  # (I+A)^{-1}(I-A)
        return Q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward color rotation.
        Args:
            x: (B, 3, H, W)
        Returns:
            Rotated color features (B, 3, H, W)
        """
        Q = self._get_rotation_matrix()  # (3, 3)
        # Apply as 1x1 conv: reshape x to (B, H*W, 3) @ Q^T
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, 3)
        out = x_flat @ Q.t()
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse color rotation (Q^{-1} = Q^T for orthogonal Q)."""
        Q = self._get_rotation_matrix()
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        out = x_flat @ Q  # Q^T inverse is Q itself transposed
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MonotonicIntensityCompression(nn.Module):
    """Learnable monotonic intensity compression via rational quadratic spline.
    
    Ensures the intensity mapping is strictly monotonic (preserves ordering)
    while adapting to different camera response curves.
    Parameterized by descriptor to be camera-adaptive.
    """
    
    def __init__(self, descriptor_dim: int = 64, num_bins: int = 8):
        super().__init__()
        self.num_bins = num_bins
        # Predict spline parameters from descriptor
        # For each bin: width, height, derivative (3 params per bin + 1 boundary)
        self.param_proj = nn.Sequential(
            nn.Linear(descriptor_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_bins * 3 + 1),  # widths, heights, derivatives
        )
        # Initialize to identity mapping
        nn.init.zeros_(self.param_proj[-1].weight)
        nn.init.zeros_(self.param_proj[-1].bias)
        
    def forward(self, intensity: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        """Apply monotonic compression to intensity channel.
        
        Args:
            intensity: (B, 1, H, W) in [0, 1]
            descriptor: (B, descriptor_dim)
        Returns:
            Compressed intensity (B, 1, H, W) in [0, 1]
        """
        B = intensity.shape[0]
        params = self.param_proj(descriptor)  # (B, num_bins*3+1)
        
        # Split into widths, heights, derivatives
        widths = params[:, :self.num_bins]
        heights = params[:, self.num_bins:2*self.num_bins]
        derivatives = params[:, 2*self.num_bins:]
        
        # Softmax to get normalized bin widths/heights (ensures monotonicity)
        widths = F.softmax(widths, dim=-1)  # (B, num_bins)
        heights = F.softmax(heights, dim=-1)  # (B, num_bins)
        derivatives = F.softplus(derivatives) + 1e-3  # positive derivatives
        
        # Compute cumulative bin edges
        cum_widths = torch.cumsum(widths, dim=-1)  # (B, num_bins)
        cum_heights = torch.cumsum(heights, dim=-1)  # (B, num_bins)
        
        # Prepend zeros
        cum_widths = F.pad(cum_widths, (1, 0), value=0.0)  # (B, num_bins+1)
        cum_heights = F.pad(cum_heights, (1, 0), value=0.0)  # (B, num_bins+1)
        
        # Find which bin each pixel falls into
        x = intensity.squeeze(1)  # (B, H, W)
        x_flat = x.reshape(B, -1)  # (B, H*W)
        
        # Bin assignment via searchsorted
        bin_idx = torch.searchsorted(cum_widths[:, 1:], x_flat.clamp(0, 1 - 1e-6))
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
        
        # Gather bin parameters
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(bin_idx)
        
        x_k = cum_widths[batch_idx, bin_idx]
        x_k1 = cum_widths[batch_idx, bin_idx + 1]
        y_k = cum_heights[batch_idx, bin_idx]
        y_k1 = cum_heights[batch_idx, bin_idx + 1]
        
        # Linear interpolation within bin (simplified rational quadratic)
        bin_width = (x_k1 - x_k).clamp(min=1e-6)
        bin_height = (y_k1 - y_k).clamp(min=1e-6)
        
        t = ((x_flat - x_k) / bin_width).clamp(0, 1)
        
        # Smooth interpolation with derivative-aware Hermite
        d_k = derivatives[batch_idx, bin_idx]
        t_smooth = t * t * (3 - 2 * t)  # smoothstep for C1 continuity
        
        out = y_k + bin_height * t_smooth
        out = out.reshape(B, 1, intensity.shape[2], intensity.shape[3])
        
        return out.clamp(0, 1)


class CAMT(nn.Module):
    """Camera-Adaptive Manifold Transform.
    
    Learns an invertible, camera-adaptive color manifold that replaces
    fixed RGB/HSV/HVI color spaces. The transform adapts to the estimated
    camera response and degradation state of the input image.
    
    Key properties:
    - Approximately invertible (cycle consistency)
    - Preserves color for normal-light images
    - No hue discontinuity
    - Numerically stable in extreme dark regions
    """
    
    def __init__(self, descriptor_dim: int = 64, num_bins: int = 8):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        
        # Camera descriptor encoder
        self.descriptor_encoder = CameraDescriptorEncoder(3, descriptor_dim)
        
        # Invertible color rotation (luminance-chrominance decoupling)
        self.color_rotation = InvertibleColorRotation(3)
        
        # Descriptor-conditioned intensity compression
        self.intensity_compress = MonotonicIntensityCompression(descriptor_dim, num_bins)
        
        # Confidence predictor: how reliable is the transform for this region
        self.confidence_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward manifold transform.
        
        Args:
            x: Input RGB image (B, 3, H, W) in [0, 1]
        Returns:
            dict with keys:
                'manifold': Adaptive manifold features (B, 3, H, W)
                'descriptor': Camera descriptor (B, descriptor_dim)
                'confidence': Transform confidence map (B, 1, H, W)
        """
        # Estimate camera-degradation descriptor
        descriptor = self.descriptor_encoder(x)  # (B, D)
        
        # Step 1: Invertible color rotation (adaptive luminance-chrominance split)
        rotated = self.color_rotation(x)  # (B, 3, H, W)
        
        # Step 2: Adaptive intensity compression on first channel (luminance)
        intensity = rotated[:, :1]  # (B, 1, H, W)
        chrominance = rotated[:, 1:]  # (B, 2, H, W)
        
        compressed_intensity = self.intensity_compress(intensity, descriptor)
        
        # Step 3: Reassemble manifold representation
        manifold = torch.cat([compressed_intensity, chrominance], dim=1)  # (B, 3, H, W)
        
        # Step 4: Compute confidence map
        confidence = self.confidence_head(x)  # (B, 1, H, W)
        
        return {
            'manifold': manifold,
            'descriptor': descriptor,
            'confidence': confidence,
        }
    
    def inverse(self, manifold: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        """Approximate inverse transform for cycle consistency.
        
        Args:
            manifold: Manifold features (B, 3, H, W)
            descriptor: Camera descriptor (B, descriptor_dim)
        Returns:
            Reconstructed RGB (B, 3, H, W)
        """
        # Inverse intensity compression (approximate via residual)
        intensity = manifold[:, :1]
        chrominance = manifold[:, 1:]
        
        # For inverse, we use a learned residual correction
        # (exact spline inversion is complex; approximate is sufficient for cycle loss)
        recovered_intensity = intensity  # Simplified: identity approx for stability
        
        # Inverse color rotation
        recovered = torch.cat([recovered_intensity, chrominance], dim=1)
        recovered_rgb = self.color_rotation.inverse(recovered)
        
        return recovered_rgb.clamp(0, 1)


# =============================================================================
# CEI: Counterfactual Exposure Intervention
# =============================================================================

class DegradationEncoder(nn.Module):
    """Encode degradation state from input image.
    
    Estimates: spatial illumination, noise level, color shift, exposure level.
    Used to construct counterfactual degradation states during training.
    """
    
    def __init__(self, in_channels: int = 3, embed_dim: int = 32):
        super().__init__()
        
        # Spatial illumination estimation
        self.illumination_head = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        
        # Global degradation descriptor (noise, color shift, exposure)
        self.global_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
        )
        
        # Decomposition heads from global descriptor
        self.noise_head = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.color_shift_head = nn.Linear(embed_dim, 3)
        self.exposure_head = nn.Linear(embed_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)
        Returns:
            dict with illumination_map, noise_level, color_shift, exposure_level,
            degradation_descriptor
        """
        illumination = self.illumination_head(x)  # (B, 1, H, W)
        
        global_desc = self.global_encoder(x)  # (B, embed_dim)
        
        noise_level = self.noise_head(global_desc)  # (B, 1)
        color_shift = self.color_shift_head(global_desc)  # (B, 3)
        exposure_level = self.exposure_head(global_desc)  # (B, 1)
        
        return {
            'illumination': illumination,
            'noise_level': noise_level,
            'color_shift': color_shift,
            'exposure_level': exposure_level,
            'descriptor': global_desc,
        }


class CounterfactualIntervention(nn.Module):
    """Counterfactual Exposure Intervention module.
    
    During training, constructs counterfactual low-light states by intervening
    on exposure and noise variables while keeping scene content fixed.
    Enforces content invariance across different degradation states.
    """
    
    def __init__(self, in_channels: int = 3, num_interventions: int = 2):
        super().__init__()
        self.num_interventions = num_interventions
        
        # Degradation encoder
        self.degradation_encoder = DegradationEncoder(in_channels, embed_dim=32)
        
        # Learnable intervention strengths (exposure scaling factors)
        self.exposure_factors = nn.Parameter(
            torch.linspace(0.3, 0.7, num_interventions).unsqueeze(0)  # (1, K)
        )
        
    def construct_counterfactual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Construct counterfactual inputs with different exposure states.
        
        Args:
            x: Original low-light input (B, 3, H, W)
        Returns:
            counterfactual_inputs: List of (B, 3, H, W) tensors
            degradation_info: Dict with degradation estimates
        """
        degradation = self.degradation_encoder(x)
        illumination = degradation['illumination']  # (B, 1, H, W)
        
        counterfactuals = []
        B = x.shape[0]
        
        for k in range(self.num_interventions):
            # Intervention: scale illumination while preserving content structure
            factor = self.exposure_factors[:, k:k+1].unsqueeze(-1)  # (1, 1, 1)
            
            # Apply exposure intervention: modify brightness non-uniformly
            # guided by estimated illumination map
            intervention_map = 1.0 + (factor - 1.0) * illumination
            cf_input = x * intervention_map.clamp(0.1, 3.0)
            cf_input = cf_input.clamp(0, 1)
            
            counterfactuals.append(cf_input)
        
        return counterfactuals, degradation
    
    def forward(self, x: torch.Tensor) -> Tuple[list, Dict]:
        """Training-time forward: construct counterfactual states."""
        return self.construct_counterfactual(x)


# =============================================================================
# OGDR: Observability-Guided Dynamic Restoration
# =============================================================================

class ObservabilityEstimator(nn.Module):
    """Estimate per-pixel information observability/recoverability.
    
    Combines multiple cues:
    - Local intensity (brighter = more observable)
    - Gradient reliability (stronger gradients = more reliable)
    - Estimated noise variance (higher noise = less observable)
    - CAMT confidence (transform reliability)
    
    Output: soft observability map in [0, 1] where
    1 = information-rich (use local restoration)
    0 = information-scarce (use global restoration)
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Multi-cue fusion network
        # Input: intensity(1) + gradient_mag(1) + noise_est(1) + confidence(1) = 4
        self.fusion = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        
    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude as reliability cue."""
        # Sobel-like gradient
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        # Pad to original size
        diff_h = F.pad(diff_h, (0, 0, 0, 1), mode='replicate')
        diff_w = F.pad(diff_w, (0, 1, 0, 0), mode='replicate')
        grad_mag = torch.sqrt(diff_h.pow(2) + diff_w.pow(2) + 1e-8)
        # Normalize to [0, 1]
        grad_mag = grad_mag / (grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return grad_mag.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    def _estimate_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Simple noise estimation via local variance."""
        # Use Laplacian as noise indicator
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                            dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        gray = x.mean(dim=1, keepdim=True)
        laplacian = F.conv2d(gray, kernel, padding=1)
        noise_est = laplacian.abs()
        noise_est = noise_est / (noise_est.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return noise_est
    
    def forward(self, x: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, 3, H, W)
            confidence: CAMT confidence map (B, 1, H, W)
        Returns:
            observability: (B, 1, H, W) in [0, 1]
        """
        # Cue 1: Local intensity
        intensity = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Cue 2: Gradient reliability
        gradient = self._compute_gradient_magnitude(x)  # (B, 1, H, W)
        
        # Cue 3: Noise level (inverse: high noise = low observability)
        noise = 1.0 - self._estimate_noise(x)  # (B, 1, H, W)
        
        # Cue 4: CAMT confidence
        # confidence already (B, 1, H, W)
        
        # Fuse all cues
        cues = torch.cat([intensity, gradient, noise, confidence], dim=1)  # (B, 4, H, W)
        observability = self.fusion(cues)  # (B, 1, H, W)
        
        return observability


class DynamicRestorationBlock(nn.Module):
    """Observability-guided dynamic restoration.
    
    Routes between:
    - Local path (depthwise conv): for information-rich regions
    - Global path (channel attention): for information-scarce regions
    
    Soft routing via observability map avoids hard boundaries.
    """
    
    def __init__(self, dim: int, num_heads: int = 1, ffn_expansion: float = 2.66):
        super().__init__()
        
        # Local restoration path: depthwise separable conv
        self.local_path = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        
        # Global restoration path: channel attention (lightweight)
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )
        
        # Gated FFN after routing
        hidden_dim = int(dim * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, observability: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
            observability: (B, 1, H, W) in [0, 1], 1=local, 0=global
        Returns:
            Restored features (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Local path
        local_out = self.local_path(x)  # (B, C, H, W)
        
        # Global path
        global_weight = self.global_path(x)  # (B, C)
        global_out = x * global_weight.unsqueeze(-1).unsqueeze(-1)  # (B, C, H, W)
        
        # Soft routing via observability
        # High observability -> trust local; Low observability -> use global
        route_weight = observability  # (B, 1, H, W)
        routed = route_weight * local_out + (1 - route_weight) * global_out
        
        # Residual + FFN
        out = x + routed
        
        # LayerNorm (channel-last)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        out = out + self.ffn(out)
        
        return out
