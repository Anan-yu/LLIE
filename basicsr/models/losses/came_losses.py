"""
CAME-SAIGFormer Loss Functions (v2 - Full Replacement)
=======================================================
Losses designed for the new architecture where:
- CEI is embedded (cf_features are bottleneck features, not full images)
- Content/degradation decomposition is explicit
- Observability map is produced by the network

Loss components:
1. L_rec: L1 reconstruction
2. L_ssim: Structural similarity
3. L_content_inv: Content invariance across counterfactual interventions
4. L_cycle: CAMT forward-inverse cycle consistency
5. L_raed: Reference-Ambiguity-Aware Exposure Distribution
6. L_obs_smooth: Edge-aware observability smoothness
7. L_disentangle: Content-degradation orthogonality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ContentInvarianceLoss(nn.Module):
    """Content invariance across counterfactual degradation interventions.
    
    The content features from the bottleneck should remain consistent
    when degradation variables are intervened upon.
    
    L = (1/K) * sum_k || content - recombine(content, intervened_deg) ||_1
    
    This is NOT contrastive learning. Positive pairs are defined by
    physical intervention on the degradation subspace.
    """
    
    def __init__(self, loss_weight: float = 0.1):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, output_features: torch.Tensor, 
                cf_features: List[torch.Tensor]) -> torch.Tensor:
        if not cf_features:
            return torch.tensor(0.0, device=output_features.device)
        
        loss = 0.0
        for cf in cf_features:
            loss += F.l1_loss(output_features, cf)
        return self.loss_weight * loss / len(cf_features)


class DisentangleLoss(nn.Module):
    """Content-degradation orthogonality constraint.
    
    Ensures content and degradation representations capture different information
    by encouraging orthogonality in their spatial feature maps.
    
    L = |cos_sim(content_flat, degradation_flat)|
    """
    
    def __init__(self, loss_weight: float = 0.02):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, content: torch.Tensor, degradation: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims
        B = content.shape[0]
        c_flat = content.reshape(B, -1)
        d_flat = degradation.reshape(B, -1)
        
        # Cosine similarity (want it near 0 = orthogonal)
        cos_sim = F.cosine_similarity(c_flat, d_flat, dim=-1)
        return self.loss_weight * cos_sim.abs().mean()


class CycleConsistencyLoss(nn.Module):
    """CAMT forward-inverse cycle: || x - CAMT^{-1}(CAMT(x)) ||_1"""
    
    def __init__(self, loss_weight: float = 0.05):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.l1_loss(original, reconstructed)


class RAEDLoss(nn.Module):
    """Reference-Ambiguity-Aware Exposure Distribution Loss.
    
    Constrains exposure distribution rather than forcing exact GT mean match.
    Components: quantile matching + local ordering consistency.
    Does NOT use GT-Mean preprocessing.
    """
    
    def __init__(self, loss_weight: float = 0.05, num_quantiles: int = 5):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_quantiles = num_quantiles
        self.register_buffer('quantile_levels', torch.linspace(0.1, 0.9, num_quantiles))
        
    def _luminance(self, x: torch.Tensor) -> torch.Tensor:
        return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lum = self._luminance(pred)
        target_lum = self._luminance(target)
        
        B = pred.shape[0]
        loss = 0.0
        
        # Quantile matching
        pred_flat = pred_lum.reshape(B, -1)
        target_flat = target_lum.reshape(B, -1)
        
        for q in self.quantile_levels:
            k = max(1, int(q * pred_flat.shape[1]))
            pred_q = torch.topk(pred_flat, k, dim=1, largest=True).values.mean(dim=1)
            target_q = torch.topk(target_flat, k, dim=1, largest=True).values.mean(dim=1)
            loss += F.l1_loss(pred_q, target_q)
        
        loss = loss / self.num_quantiles
        
        # Local ordering: downsampled patch brightness ordering should match
        patch_pred = F.adaptive_avg_pool2d(pred_lum, 8)
        patch_target = F.adaptive_avg_pool2d(target_lum, 8)
        pf = patch_pred.reshape(B, -1)
        tf = patch_target.reshape(B, -1)
        
        # Pairwise ordering hinge
        pred_diff = pf.unsqueeze(2) - pf.unsqueeze(1)
        target_diff = tf.unsqueeze(2) - tf.unsqueeze(1)
        ordering_loss = F.relu(0.05 - pred_diff * torch.sign(target_diff)).mean()
        
        return self.loss_weight * (loss + 0.3 * ordering_loss)


class ObservabilitySmoothLoss(nn.Module):
    """Edge-aware smoothness for observability map.
    
    Encourages smooth transitions except at actual image edges.
    """
    
    def __init__(self, loss_weight: float = 0.01):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, observability: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        grad_h = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        grad_w = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        
        weight_h = torch.exp(-grad_h * 10)
        weight_w = torch.exp(-grad_w * 10)
        
        obs_h = (observability[:, :, 1:, :] - observability[:, :, :-1, :]).abs()
        obs_w = (observability[:, :, :, 1:] - observability[:, :, :, :-1]).abs()
        
        return self.loss_weight * ((obs_h * weight_h).mean() + (obs_w * weight_w).mean())


class CAMELoss(nn.Module):
    """Combined loss for CAME-SAIGFormer v2.
    
    L_total = L_rec + L_ssim + λ1*L_content_inv + λ2*L_cycle 
              + λ3*L_raed + λ4*L_obs_smooth + λ5*L_disentangle
    """
    
    def __init__(
        self,
        content_inv_weight: float = 0.1,
        cycle_weight: float = 0.05,
        raed_weight: float = 0.05,
        obs_smooth_weight: float = 0.01,
        disentangle_weight: float = 0.02,
    ):
        super().__init__()
        self.content_inv_loss = ContentInvarianceLoss(content_inv_weight)
        self.cycle_loss = CycleConsistencyLoss(cycle_weight)
        self.raed_loss = RAEDLoss(raed_weight)
        self.obs_smooth_loss = ObservabilitySmoothLoss(obs_smooth_weight)
        self.disentangle_loss = DisentangleLoss(disentangle_weight)
        
    def forward(self, model_output: Dict, gt: torch.Tensor, 
                inp_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_output: Dict from CAME_SAIGFormer training forward
            gt: Ground truth (B, 3, H, W)
            inp_img: Original input (B, 3, H, W)
        Returns:
            Dict of losses including 'l_total'
        """
        output = model_output['output']
        losses = {}
        
        # Reconstruction
        losses['l_rec'] = F.l1_loss(output, gt)
        
        # Content invariance (counterfactual consistency at bottleneck)
        cf_features = model_output.get('cf_features', [])
        output_features = model_output.get('content_features', None)
        if output_features is not None and cf_features:
            # Compare content features with counterfactual recombined features
            losses['l_content_inv'] = self.content_inv_loss(output_features, cf_features)
        else:
            losses['l_content_inv'] = torch.tensor(0.0, device=output.device)
        
        # Cycle consistency
        cycle_recon = model_output.get('cycle_recon', None)
        if cycle_recon is not None:
            losses['l_cycle'] = self.cycle_loss(inp_img, cycle_recon)
        else:
            losses['l_cycle'] = torch.tensor(0.0, device=output.device)
        
        # RAED
        losses['l_raed'] = self.raed_loss(output, gt)
        
        # Observability smoothness
        observability = model_output.get('observability', None)
        if observability is not None:
            losses['l_obs_smooth'] = self.obs_smooth_loss(observability, inp_img)
        else:
            losses['l_obs_smooth'] = torch.tensor(0.0, device=output.device)
        
        # Disentanglement orthogonality
        content = model_output.get('content_features', None)
        degradation = model_output.get('degradation_features', None)
        if content is not None and degradation is not None:
            losses['l_disentangle'] = self.disentangle_loss(content, degradation)
        else:
            losses['l_disentangle'] = torch.tensor(0.0, device=output.device)
        
        # Total (excluding l_rec which is handled by pixel_opt in training model)
        losses['l_came_total'] = (
            losses['l_content_inv'] + 
            losses['l_cycle'] + 
            losses['l_raed'] + 
            losses['l_obs_smooth'] + 
            losses['l_disentangle']
        )
        
        return losses
