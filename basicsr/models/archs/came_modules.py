"""
CAME-SAIGFormer Core Modules (v2 - Full Replacement)
=====================================================
Completely replaces SAIGFormer's original innovations:
  - SAI2E (fixed-RGB integral image) → ManifoldAdaptiveIllumination
  - IlluminationGuideAttention (svp concat into query) → ObservabilityConditionedAttention
  - No external CEI forward pass → CounterfactualDisentanglement embedded in encoder

Scientific hypothesis: The optimal color representation for low-light enhancement
should adapt to camera response and degradation state, not remain fixed (RGB/HSV/HVI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional
from einops import rearrange


# =============================================================================
# Module 1: Camera-Adaptive Manifold Transform (CAMT)
# Replaces: Fixed RGB input space assumption
# =============================================================================

class CameraDescriptorEncoder(nn.Module):
    """Estimate a compact camera-degradation descriptor from input.
    
    Encodes: white balance shift, response curve nonlinearity,
    noise characteristics, global exposure level.
    This descriptor conditions ALL subsequent modules.
    """
    
    def __init__(self, in_channels: int = 3, descriptor_dim: int = 64):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.proj = nn.Sequential(
            nn.Linear(64, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B,3,H,W) in [0,1]. Returns: descriptor (B, D)."""
        feat = self.encoder(x)
        return self.proj(feat)


class InvertibleColorRotation(nn.Module):
    """Invertible 1x1 color rotation via Cayley transform on skew-symmetric matrix.
    
    Guarantees orthogonality (Q^T Q = I), initialized as identity.
    This adaptively decouples luminance from chrominance based on camera response.
    """
    
    def __init__(self, channels: int = 3):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(channels, channels))
        self.channels = channels
        
    def _get_rotation_matrix(self) -> torch.Tensor:
        A = self.A - self.A.t()  # enforce skew-symmetry
        I = torch.eye(self.channels, device=A.device, dtype=A.dtype)
        Q = torch.linalg.solve(I + A, I - A)
        return Q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self._get_rotation_matrix()
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        out = x_flat @ Q.t()
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        Q = self._get_rotation_matrix()
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        out = x_flat @ Q  # Q^{-1} = Q^T for orthogonal
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class DescriptorConditionedSpline(nn.Module):
    """Monotonic intensity compression conditioned on camera descriptor.
    
    Replaces fixed gamma/log curves with a learned, camera-adaptive,
    strictly monotonic mapping via piecewise linear spline.
    """
    
    def __init__(self, descriptor_dim: int = 64, num_bins: int = 8):
        super().__init__()
        self.num_bins = num_bins
        # Predict bin heights from descriptor (widths are uniform)
        self.param_proj = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_bins),
        )
        nn.init.zeros_(self.param_proj[-1].weight)
        nn.init.zeros_(self.param_proj[-1].bias)
        
    def forward(self, intensity: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        """Apply monotonic compression.
        Args: intensity (B,1,H,W) in [0,1], descriptor (B,D)
        Returns: compressed (B,1,H,W) in [0,1]
        """
        B, _, H, W = intensity.shape
        
        # Predict per-bin slopes (positive via softplus)
        raw_params = self.param_proj(descriptor)  # (B, num_bins)
        slopes = F.softplus(raw_params) + 0.5  # ensure positive, centered around 1
        
        # Normalize so total mapping spans [0,1]
        slopes = slopes / slopes.mean(dim=-1, keepdim=True)
        
        # Build cumulative mapping (piecewise linear, monotonic)
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=intensity.device)
        bin_width = 1.0 / self.num_bins
        
        # Compute output heights for each bin edge
        heights = slopes * bin_width  # (B, num_bins)
        cum_heights = torch.cumsum(heights, dim=-1)  # (B, num_bins)
        cum_heights = F.pad(cum_heights, (1, 0), value=0.0)  # (B, num_bins+1)
        # Normalize to [0,1]
        cum_heights = cum_heights / cum_heights[:, -1:].clamp(min=1e-6)
        
        # Apply mapping via interpolation
        x = intensity.reshape(B, -1).clamp(0, 1)  # (B, H*W)
        
        # Find bin index for each pixel
        bin_idx = (x * self.num_bins).long().clamp(0, self.num_bins - 1)  # (B, H*W)
        
        # Gather bin boundaries
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(bin_idx)
        x_left = bin_edges[bin_idx]  # (B, H*W)
        x_right = bin_edges[bin_idx + 1]
        y_left = cum_heights[batch_idx, bin_idx]
        y_right = cum_heights[batch_idx, bin_idx + 1]
        
        # Linear interpolation within bin
        t = ((x - x_left) / (x_right - x_left + 1e-8)).clamp(0, 1)
        out = y_left + t * (y_right - y_left)
        
        return out.reshape(B, 1, H, W)


class CAMT(nn.Module):
    """Camera-Adaptive Manifold Transform.
    
    Core innovation: the color representation adapts to camera/degradation state.
    NOT a new fixed color space — a learned, invertible, input-conditioned manifold.
    """
    
    def __init__(self, descriptor_dim: int = 64, num_bins: int = 8):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.descriptor_encoder = CameraDescriptorEncoder(3, descriptor_dim)
        self.color_rotation = InvertibleColorRotation(3)
        self.intensity_spline = DescriptorConditionedSpline(descriptor_dim, num_bins)
        
        # Confidence: how reliable is the manifold for this pixel
        self.confidence_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: x (B,3,H,W) in [0,1]
        Returns: dict with 'manifold', 'descriptor', 'confidence'
        """
        descriptor = self.descriptor_encoder(x)  # (B, D)
        
        # Adaptive luminance-chrominance decoupling
        rotated = self.color_rotation(x)  # (B, 3, H, W)
        intensity = rotated[:, :1]
        chrominance = rotated[:, 1:]
        
        # Descriptor-conditioned intensity compression
        compressed = self.intensity_spline(intensity, descriptor)
        manifold = torch.cat([compressed, chrominance], dim=1)
        
        confidence = self.confidence_head(x)
        
        return {'manifold': manifold, 'descriptor': descriptor, 'confidence': confidence}
    
    def inverse(self, manifold: torch.Tensor) -> torch.Tensor:
        """Approximate inverse for cycle consistency."""
        return self.color_rotation.inverse(manifold).clamp(0, 1)


# =============================================================================
# Module 2: ManifoldAdaptiveIllumination (replaces SAI2E)
# Key difference: operates in CAMT manifold space, not fixed RGB
# =============================================================================

class ManifoldAdaptiveIllumination(nn.Module):
    """Spatially-adaptive illumination estimation in manifold space.
    
    Replaces SAI2E which operates on fixed RGB.
    Key differences from SAI2E:
    1. Input is the adaptive manifold (not raw RGB)
    2. Offset prediction is conditioned on camera descriptor
    3. Modulation uses descriptor-aware scaling
    4. The integral image captures structure in the ADAPTIVE color space
    
    This means illumination estimation itself changes with camera response.
    """
    
    def __init__(self, in_channels: int = 3, descriptor_dim: int = 64,
                 train_patch: int = 128, eps: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.train_patch = train_patch if isinstance(train_patch, list) else [train_patch, train_patch]
        
        # Descriptor-conditioned offset prediction (replaces SAI2E's fixed offset_predict)
        self.offset_predict = nn.Sequential(
            nn.Conv2d(in_channels + descriptor_dim, 32, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 4, 1, bias=True),
        )
        
        # Descriptor-conditioned modulation (replaces SAI2E's fixed modulation_predict)
        self.modulation_predict = nn.Sequential(
            nn.Conv2d(in_channels + descriptor_dim, 32, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 1, bias=True),
        )
        
    def _get_center_grid(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='xy'), dim=-1)
        norm_coords = coords / torch.tensor([W, H], dtype=x.dtype, device=x.device) * 2 - 1
        return norm_coords
    
    def forward(self, manifold: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            manifold: Adaptive manifold features (B, 3, H, W)
            descriptor: Camera descriptor (B, D)
        Returns:
            Illumination guide in manifold space (B, 3, H, W)
        """
        B, C, H, W = manifold.shape
        
        # Expand descriptor to spatial map for conditioning
        desc_map = descriptor.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)  # (B, D, H, W)
        conditioned_input = torch.cat([manifold, desc_map], dim=1)  # (B, 3+D, H, W)
        
        # Integral image in manifold space
        integrated = torch.cumsum(manifold, dim=-1)
        integrated = torch.cumsum(integrated, dim=-2)
        
        # Descriptor-conditioned offset prediction
        center_grid = self._get_center_grid(manifold).unsqueeze(0)
        normalizer = torch.tensor(
            [self.train_patch[0]/W, self.train_patch[1]/H],
            dtype=manifold.dtype, device=manifold.device
        ).view(1, 1, 1, 2)
        
        subnet_output = self.offset_predict(conditioned_input).permute(0, 2, 3, 1)
        off_w, off_h = torch.split(subnet_output, 2, dim=3)
        off_w = off_w - off_w.mean(dim=-1, keepdim=True)
        off_h = off_h - off_h.mean(dim=-1, keepdim=True)
        
        minimum_patch = 2
        off_w_min = torch.minimum(off_w.min(dim=-1, keepdim=True)[0],
                                   torch.zeros_like(off_w.min(dim=-1, keepdim=True)[0]) - minimum_patch/self.train_patch[0])
        off_w_max = torch.maximum(off_w.max(dim=-1, keepdim=True)[0],
                                   torch.zeros_like(off_w.max(dim=-1, keepdim=True)[0]) + minimum_patch/self.train_patch[0])
        off_h_min = torch.minimum(off_h.min(dim=-1, keepdim=True)[0],
                                   torch.zeros_like(off_h.min(dim=-1, keepdim=True)[0]) - minimum_patch/self.train_patch[1])
        off_h_max = torch.maximum(off_h.max(dim=-1, keepdim=True)[0],
                                   torch.zeros_like(off_h.max(dim=-1, keepdim=True)[0]) + minimum_patch/self.train_patch[1])
        
        area = (off_h_max - off_h_min) * (off_w_max - off_w_min) * self.train_patch[0] * self.train_patch[1] / 4
        area = area.view(B, 1, H, W).clip(1, H * W)
        
        # Descriptor-conditioned modulation
        scale = self.modulation_predict(conditioned_input)
        if self.eps != 0:
            mask = (scale.abs() < self.eps)
            safe_sign = torch.where(scale >= 0, 1.0, -1.0)
            scale = torch.where(mask, safe_sign * self.eps, scale)
        area = area * scale
        
        # Corner coordinates
        off_tl = (torch.cat([off_w_min, off_h_min], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_tr = (torch.cat([off_w_max, off_h_min], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_bl = (torch.cat([off_w_min, off_h_max], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_br = (torch.cat([off_w_max, off_h_max], dim=-1) * normalizer + center_grid).clip(-1, 1)
        
        # Sample integral image
        A = F.grid_sample(integrated, off_tl, align_corners=True, padding_mode='border', mode='bilinear')
        B_val = F.grid_sample(integrated, off_tr, align_corners=True, padding_mode='border', mode='bilinear')
        C_val = F.grid_sample(integrated, off_bl, align_corners=True, padding_mode='border', mode='bilinear')
        D_val = F.grid_sample(integrated, off_br, align_corners=True, padding_mode='border', mode='bilinear')
        
        res = (A + D_val - B_val - C_val) / area
        return res


# =============================================================================
# Module 3: ObservabilityConditionedAttention (replaces IlluminationGuideAttention)
# Key difference: attention routing driven by region recoverability, not fixed concat
# =============================================================================

class ObservabilityEstimator(nn.Module):
    """Estimate per-pixel information recoverability.
    
    Fuses: local intensity, gradient reliability, noise level, CAMT confidence.
    Output in [0,1]: 1=information-rich (trust local), 0=information-scarce (need global).
    """
    
    def __init__(self):
        super().__init__()
        # 4 cues: intensity(1) + gradient(1) + noise_inverse(1) + confidence(1)
        self.fusion = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """Args: x (B,3,H,W), confidence (B,1,H,W). Returns: (B,1,H,W)."""
        intensity = x.mean(dim=1, keepdim=True)
        
        # Gradient magnitude
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        diff_h = F.pad(diff_h, (0, 0, 0, 1), mode='replicate')
        diff_w = F.pad(diff_w, (0, 1, 0, 0), mode='replicate')
        grad = (diff_h.pow(2) + diff_w.pow(2) + 1e-8).sqrt().mean(dim=1, keepdim=True)
        grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + 1e-6)
        
        # Noise indicator (Laplacian)
        gray = x.mean(dim=1, keepdim=True)
        kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
        noise = 1.0 - (F.conv2d(gray, kernel, padding=1).abs() / 
                       (F.conv2d(gray, kernel, padding=1).abs().amax(dim=(2,3), keepdim=True) + 1e-6))
        
        cues = torch.cat([intensity, grad, noise, confidence], dim=1)
        return self.fusion(cues)


class ObservabilityConditionedAttention(nn.Module):
    """Attention mechanism where query construction is driven by observability.
    
    Replaces IlluminationGuideAttention which blindly concatenates illumination
    features into the query. Instead:
    - Information-rich regions: query emphasizes LOCAL detail preservation
    - Information-scarce regions: query emphasizes GLOBAL context gathering
    
    The illumination guide modulates attention temperature (not query content),
    and observability controls the local/global balance.
    """
    
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, 
                                     padding=1, groups=dim * 3, bias=bias)
        
        # Illumination guide modulates attention temperature (not query content)
        self.illum_modulator = nn.Sequential(
            nn.Conv2d(3, num_heads, 1, bias=True),
            nn.Sigmoid(),
        )
        
        # Observability-conditioned local detail injection
        self.local_detail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor, illum_guide: torch.Tensor, 
                observability: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
            illum_guide: Manifold illumination guide (B, 3, H, W)
            observability: Recoverability map (B, 1, H, W), 1=rich, 0=scarce
        Returns:
            Attended features (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Illumination modulates attention temperature per head
        illum_weight = self.illum_modulator(illum_guide)  # (B, num_heads, H, W)
        illum_weight = illum_weight.mean(dim=(2, 3))  # (B, num_heads) global per head
        # temperature: (num_heads, 1, 1) -> (1, num_heads, 1, 1) for batch broadcast
        adaptive_temp = self.temperature.unsqueeze(0) * (0.5 + illum_weight.unsqueeze(-1).unsqueeze(-1))
        # adaptive_temp: (B, num_heads, 1, 1)
        
        attn = (q @ k.transpose(-2, -1)) * adaptive_temp  # (B, head, c, c) * (B, head, 1, 1)
        attn = attn.softmax(dim=-1)
        
        # Global attention output
        global_out = attn @ v  # (B, head, c, h*w)
        global_out = rearrange(global_out, 'b head c (h w) -> b (head c) h w', h=H, w=W)
        
        # Local detail path (for information-rich regions)
        local_out = self.local_detail(x)  # (B, C, H, W)
        
        # Observability-driven soft routing
        # High observability → trust local detail; Low → rely on global attention
        route = observability  # (B, 1, H, W)
        out = route * local_out + (1 - route) * global_out
        
        return self.project_out(out)


# =============================================================================
# Module 4: CounterfactualDisentanglement (embedded in encoder, replaces external CEI)
# =============================================================================

class CounterfactualDisentanglement(nn.Module):
    """Feature disentanglement via counterfactual intervention.
    
    Embedded at the encoder bottleneck. Decomposes features into:
    - Content representation (scene structure, invariant to degradation)
    - Degradation representation (illumination + noise + color shift)
    
    During training: intervenes on degradation variables to construct
    counterfactual states, enforcing content invariance.
    """
    
    def __init__(self, dim: int, num_interventions: int = 2):
        super().__init__()
        self.num_interventions = num_interventions
        
        # Decomposition: split features into content and degradation subspaces
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
        
        # Learnable intervention directions in degradation space
        self.intervention_vectors = nn.Parameter(
            torch.randn(num_interventions, dim) * 0.01
        )
        
        # Recombination after intervention
        self.recombine = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: x (B, C, H, W) encoder bottleneck features
        Returns: dict with 'content', 'degradation', 'cf_features' (training only)
        """
        content = self.content_proj(x)  # (B, C, H, W)
        degradation = self.degradation_proj(x)  # (B, C, H, W)
        
        result = {'content': content, 'degradation': degradation}
        
        if self.training:
            cf_features = []
            for k in range(self.num_interventions):
                # Intervene: shift degradation along learned direction
                direction = self.intervention_vectors[k].view(1, -1, 1, 1)  # (1, C, 1, 1)
                intervened_deg = degradation + direction * degradation.mean(dim=(2, 3), keepdim=True)
                
                # Recombine content + intervened degradation
                cf = self.recombine(torch.cat([content, intervened_deg], dim=1))
                cf_features.append(cf)
            
            result['cf_features'] = cf_features
        
        # Normal recombination (no intervention)
        out = self.recombine(torch.cat([content, degradation], dim=1))
        result['output'] = out
        
        return result


# =============================================================================
# Transformer Block (replaces SAIGTransformer)
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mu = x.mean(1, keepdim=True)
        sigma = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DualGatedFFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj_1 = nn.Conv2d(dim, hidden, 1, bias=bias)
        self.proj_2 = nn.Conv2d(dim, hidden, 1, bias=bias)
        self.out = nn.Conv2d(hidden, dim, 1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1, p2 = self.proj_1(x), self.proj_2(x)
        return self.out(p1 * torch.sigmoid(p2) + p2 * F.gelu(p1))


class CAMETransformerBlock(nn.Module):
    """Transformer block with ObservabilityConditionedAttention.
    
    Replaces SAIGTransformer which uses IlluminationGuideAttention.
    """
    
    def __init__(self, dim: int, num_heads: int, ffn_expansion: float = 2.66, bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = ObservabilityConditionedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = DualGatedFFN(dim, ffn_expansion, bias)
        
    def forward(self, x: torch.Tensor, illum_guide: torch.Tensor, 
                observability: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), illum_guide, observability)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Resize modules (same as original, standard components)
# =============================================================================

class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x): return self.body(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)
    def forward(self, x): return self.proj(x)
