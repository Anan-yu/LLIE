"""
CAME-SAIGFormer Loss Functions
================================
Novel losses for Camera-Adaptive Manifold and Counterfactual Exposure Intervention:

1. ContentInvarianceLoss - Enforces scene content consistency across counterfactual states
2. CycleConsistencyLoss - Ensures CAMT forward-inverse cycle preserves information
3. RAEDLoss - Reference-Ambiguity-Aware Exposure Distribution Loss
4. ObservabilitySmoothLoss - Encourages spatially smooth observability transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ContentInvarianceLoss(nn.Module):
    """Content Invariance Loss for Counterfactual Exposure Intervention.
    
    Enforces that the enhanced outputs from different counterfactual degradation
    states (same content, different exposure/noise) should converge to the same
    scene content representation.
    
    This is NOT standard contrastive learning - the positive pairs are defined by
    physical intervention (same scene content under different exposure), and the
    loss directly measures output consistency.
    
    L_content_inv = (1/K) * sum_k || E(x) - E(cf_k) ||_1
    where E(x) is the enhanced output and cf_k are counterfactual variants.
    """
    
    def __init__(self, loss_weight: float = 0.1):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, output: torch.Tensor, cf_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            output: Main enhanced output (B, 3, H, W)
            cf_outputs: List of K counterfactual enhanced outputs, each (B, 3, H, W)
        Returns:
            Scalar loss value
        """
        if not cf_outputs:
            return torch.tensor(0.0, device=output.device)
        
        loss = 0.0
        for cf_out in cf_outputs:
            # L1 consistency between main output and counterfactual output
            loss += F.l1_loss(output, cf_out)
        
        loss = loss / len(cf_outputs)
        return self.loss_weight * loss


class CycleConsistencyLoss(nn.Module):
    """Cycle Consistency Loss for CAMT.
    
    Ensures the Camera-Adaptive Manifold Transform is approximately invertible:
    L_cycle = || x - CAMT^{-1}(CAMT(x)) ||_1
    
    This prevents the transform from losing information and ensures
    the learned manifold preserves color fidelity.
    """
    
    def __init__(self, loss_weight: float = 0.05):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original: Original input image (B, 3, H, W)
            reconstructed: Cycle-reconstructed image (B, 3, H, W)
        Returns:
            Scalar loss value
        """
        return self.loss_weight * F.l1_loss(original, reconstructed)


class RAEDLoss(nn.Module):
    """Reference-Ambiguity-Aware Exposure Distribution Loss.
    
    Addresses the problem that a single low-light image may correspond to
    multiple reasonable exposure results. Instead of forcing output to match
    a fixed GT mean brightness, RAED constrains the exposure distribution:
    
    1. Multi-scale luminance quantile matching
    2. Local exposure ordering consistency  
    3. Highlight/shadow ratio constraint
    
    Does NOT use GT-Mean preprocessing or modify test images.
    """
    
    def __init__(self, loss_weight: float = 0.05, num_quantiles: int = 5):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_quantiles = num_quantiles
        # Quantile levels to match
        self.register_buffer('quantile_levels', 
                           torch.linspace(0.1, 0.9, num_quantiles))
        
    def _luminance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute perceptual luminance from RGB."""
        # ITU-R BT.709 luminance
        return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    
    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-scale quantile matching loss."""
        B = pred.shape[0]
        loss = 0.0
        
        for q in self.quantile_levels:
            # Compute quantile values per image
            pred_flat = pred.view(B, -1)
            target_flat = target.view(B, -1)
            
            k = max(1, int(q * pred_flat.shape[1]))
            
            pred_q = torch.topk(pred_flat, k, dim=1, largest=True).values.mean(dim=1)
            target_q = torch.topk(target_flat, k, dim=1, largest=True).values.mean(dim=1)
            
            loss += F.l1_loss(pred_q, target_q)
        
        return loss / self.num_quantiles
    
    def _exposure_ordering_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Local exposure ordering consistency.
        
        Ensures that relative brightness ordering between local patches
        is preserved (brighter regions in GT should remain brighter in output).
        """
        # Downsample to get local patches
        pred_lum = self._luminance(pred)
        target_lum = self._luminance(target)
        
        # Average pool to get patch-level brightness
        patch_size = 16
        if pred_lum.shape[-1] < patch_size:
            patch_size = max(4, pred_lum.shape[-1] // 4)
            
        pred_patches = F.adaptive_avg_pool2d(pred_lum, (pred_lum.shape[-2]//patch_size, 
                                                         pred_lum.shape[-1]//patch_size))
        target_patches = F.adaptive_avg_pool2d(target_lum, (target_lum.shape[-2]//patch_size,
                                                             target_lum.shape[-1]//patch_size))
        
        # Compute pairwise ordering differences
        B = pred_patches.shape[0]
        pred_flat = pred_patches.view(B, -1)
        target_flat = target_patches.view(B, -1)
        
        # Sample pairs for efficiency
        N = pred_flat.shape[1]
        if N > 64:
            idx = torch.randperm(N, device=pred.device)[:64]
            pred_flat = pred_flat[:, idx]
            target_flat = target_flat[:, idx]
            N = 64
        
        # Pairwise differences
        pred_diff = pred_flat.unsqueeze(2) - pred_flat.unsqueeze(1)  # (B, N, N)
        target_diff = target_flat.unsqueeze(2) - target_flat.unsqueeze(1)
        
        # Hinge loss on ordering: sign should match
        ordering_loss = F.relu(0.1 - pred_diff * torch.sign(target_diff)).mean()
        
        return ordering_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Enhanced output (B, 3, H, W) in [0, 1]
            target: Ground truth (B, 3, H, W) in [0, 1]
        Returns:
            Scalar loss value
        """
        pred_lum = self._luminance(pred)
        target_lum = self._luminance(target)
        
        # Component 1: Quantile matching
        l_quantile = self._quantile_loss(pred_lum, target_lum)
        
        # Component 2: Exposure ordering
        l_ordering = self._exposure_ordering_loss(pred, target)
        
        loss = l_quantile + 0.5 * l_ordering
        return self.loss_weight * loss


class ObservabilitySmoothLoss(nn.Module):
    """Observability Map Smoothness Loss.
    
    Encourages the observability map to have spatially smooth transitions
    (no hard boundaries between local/global restoration regions) while
    allowing discontinuities at actual image edges.
    
    Edge-aware total variation regularization.
    """
    
    def __init__(self, loss_weight: float = 0.01):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, observability: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observability: Observability map (B, 1, H, W)
            image: Input image for edge-aware weighting (B, 3, H, W)
        Returns:
            Scalar loss value
        """
        # Compute image gradient magnitude as edge weight
        gray = image.mean(dim=1, keepdim=True)
        grad_h = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        grad_w = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        
        # Edge-aware weights (low weight at edges, high weight in flat regions)
        weight_h = torch.exp(-grad_h * 10)
        weight_w = torch.exp(-grad_w * 10)
        
        # Observability gradients
        obs_h = (observability[:, :, 1:, :] - observability[:, :, :-1, :]).abs()
        obs_w = (observability[:, :, :, 1:] - observability[:, :, :, :-1]).abs()
        
        # Weighted TV
        loss = (obs_h * weight_h).mean() + (obs_w * weight_w).mean()
        
        return self.loss_weight * loss


class CAMELoss(nn.Module):
    """Combined CAME-SAIGFormer loss.
    
    Aggregates all loss components with configurable weights.
    Designed for progressive training: start with reconstruction,
    then add regularization losses.
    
    L_total = L_rec + λ1*L_ssim + λ2*L_content_inv + λ3*L_cycle 
              + λ4*L_raed + λ5*L_obs_smooth
    """
    
    def __init__(
        self,
        rec_weight: float = 1.0,
        ssim_weight: float = 1.0,
        content_inv_weight: float = 0.1,
        cycle_weight: float = 0.05,
        raed_weight: float = 0.05,
        obs_smooth_weight: float = 0.01,
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.ssim_weight = ssim_weight
        
        self.content_inv_loss = ContentInvarianceLoss(loss_weight=content_inv_weight)
        self.cycle_loss = CycleConsistencyLoss(loss_weight=cycle_weight)
        self.raed_loss = RAEDLoss(loss_weight=raed_weight)
        self.obs_smooth_loss = ObservabilitySmoothLoss(loss_weight=obs_smooth_weight)
        
    def forward(self, model_output: Dict, gt: torch.Tensor, 
                inp_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_output: Dict from CAME_SAIGFormer forward (training mode)
            gt: Ground truth (B, 3, H, W)
            inp_img: Original input image (B, 3, H, W)
        Returns:
            Dict of individual losses and total loss
        """
        output = model_output['output']
        losses = {}
        
        # Reconstruction loss (L1)
        l_rec = F.l1_loss(output, gt)
        losses['l_rec'] = l_rec
        
        # Content invariance loss (counterfactual consistency)
        cf_outputs = model_output.get('cf_outputs', [])
        l_content_inv = self.content_inv_loss(output, cf_outputs)
        losses['l_content_inv'] = l_content_inv
        
        # Cycle consistency loss
        cycle_recon = model_output.get('cycle_recon', None)
        if cycle_recon is not None:
            l_cycle = self.cycle_loss(inp_img, cycle_recon)
        else:
            l_cycle = torch.tensor(0.0, device=output.device)
        losses['l_cycle'] = l_cycle
        
        # RAED loss
        l_raed = self.raed_loss(output, gt)
        losses['l_raed'] = l_raed
        
        # Observability smoothness
        observability = model_output.get('observability', None)
        if observability is not None:
            l_obs = self.obs_smooth_loss(observability, inp_img)
        else:
            l_obs = torch.tensor(0.0, device=output.device)
        losses['l_obs_smooth'] = l_obs
        
        # Total loss
        total = (self.rec_weight * l_rec + 
                 losses['l_content_inv'] + 
                 losses['l_cycle'] + 
                 losses['l_raed'] + 
                 losses['l_obs_smooth'])
        losses['l_total'] = total
        
        return losses
