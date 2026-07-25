"""
CAME-SAIGFormer: Camera-Adaptive Manifold and Counterfactual Exposure
Intervention Network for Robust Low-Light Image Enhancement.

COMPLETE REPLACEMENT of SAIGFormer's original innovations:
  - SAI2E → ManifoldAdaptiveIllumination (integral image in adaptive manifold space)
  - IlluminationGuideAttention → ObservabilityConditionedAttention (recoverability-driven routing)
  - External augmentation → CounterfactualDisentanglement (embedded feature decomposition)

The U-Net encoder-decoder skeleton is retained as standard infrastructure,
but ALL novel mechanisms are replaced with the new scientific contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from basicsr.models.archs.came_modules import (
    CAMT,
    ManifoldAdaptiveIllumination,
    ObservabilityEstimator,
    ObservabilityConditionedAttention,
    CounterfactualDisentanglement,
    CAMETransformerBlock,
    Downsample,
    Upsample,
    OverlapPatchEmbed,
)


class CAME_SAIGFormer(nn.Module):
    """CAME-SAIGFormer: fully replaced core innovations.
    
    Architecture:
        Input → CAMT (adaptive manifold) → ManifoldAdaptiveIllumination (guide)
              → ObservabilityEstimator → Encoder (CAMETransformerBlocks)
              → CounterfactualDisentanglement (bottleneck)
              → Decoder (CAMETransformerBlocks) → Output + residual
    
    Args:
        embed_dim: Base channel dimension
        k_s: Output conv kernel size
        encoder_num_blocks: Blocks per encoder level
        decoder_num_blocks: Blocks per decoder level
        ffn_expansion_factor: FFN expansion ratio
        heads: Attention heads per level
        train_patch: Patch size for manifold adaptive integral
        eps: Clamping threshold for modulation
        descriptor_dim: CAMT descriptor dimension
        num_interventions: Counterfactual intervention count
    """
    
    def __init__(
        self,
        embed_dim: int = 32,
        k_s: int = 3,
        encoder_num_blocks: List[int] = [4, 6, 6, 8],
        decoder_num_blocks: List[int] = [6, 6, 4, 4],
        ffn_expansion_factor: float = 2.66,
        heads: List[int] = [1, 2, 4, 8],
        train_patch: int = 128,
        eps: float = 0.1,
        descriptor_dim: int = 64,
        num_interventions: int = 2,
    ):
        super().__init__()
        
        inp_channels = 3
        out_channels = 3
        bias = False
        
        # =================================================================
        # Innovation 1: CAMT — Camera-Adaptive Manifold Transform
        # (replaces fixed RGB input assumption)
        # =================================================================
        self.camt = CAMT(descriptor_dim=descriptor_dim, num_bins=8)
        
        # =================================================================
        # Innovation 2: ManifoldAdaptiveIllumination
        # (replaces SAI2E — integral image now in adaptive manifold space,
        #  conditioned on camera descriptor)
        # =================================================================
        self.manifold_illumination = ManifoldAdaptiveIllumination(
            in_channels=inp_channels,
            descriptor_dim=descriptor_dim,
            train_patch=train_patch,
            eps=eps,
        )
        
        # =================================================================
        # Innovation 3: ObservabilityEstimator
        # (drives attention routing — replaces blind illumination concat)
        # =================================================================
        self.observability_estimator = ObservabilityEstimator()
        
        # =================================================================
        # Encoder (uses CAMETransformerBlock, NOT SAIGTransformer)
        # =================================================================
        self.patch_embed = OverlapPatchEmbed(inp_channels, embed_dim, bias=bias)
        
        self.encoder_level1 = nn.ModuleList([
            CAMETransformerBlock(embed_dim, heads[0], ffn_expansion_factor, bias)
            for _ in range(encoder_num_blocks[0])
        ])
        
        self.illum_down1_2 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.obs_down1_2 = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        self.down1_2 = Downsample(embed_dim)
        self.encoder_level2 = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**1), heads[1], ffn_expansion_factor, bias)
            for _ in range(encoder_num_blocks[1])
        ])
        
        self.illum_down2_3 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.obs_down2_3 = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        self.down2_3 = Downsample(int(embed_dim * 2**1))
        self.encoder_level3 = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**2), heads[2], ffn_expansion_factor, bias)
            for _ in range(encoder_num_blocks[2])
        ])
        
        self.illum_down3_4 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.obs_down3_4 = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        self.down3_4 = Downsample(int(embed_dim * 2**2))
        self.latent = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**3), heads[3], ffn_expansion_factor, bias)
            for _ in range(encoder_num_blocks[3])
        ])
        
        # =================================================================
        # Innovation 4: CounterfactualDisentanglement at bottleneck
        # (replaces external CEI forward pass — embedded feature decomposition)
        # =================================================================
        self.disentangle = CounterfactualDisentanglement(
            dim=int(embed_dim * 2**3),
            num_interventions=num_interventions,
        )
        
        # =================================================================
        # Decoder (uses CAMETransformerBlock)
        # =================================================================
        self.decoder_latent = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**3), heads[3], ffn_expansion_factor, bias)
            for _ in range(1)
        ])
        
        self.up4_3 = Upsample(int(embed_dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(embed_dim * 2**3), int(embed_dim * 2**2), 1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**2), heads[2], ffn_expansion_factor, bias)
            for _ in range(decoder_num_blocks[0])
        ])
        
        self.up3_2 = Upsample(int(embed_dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(embed_dim * 2**2), int(embed_dim * 2**1), 1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**1), heads[1], ffn_expansion_factor, bias)
            for _ in range(decoder_num_blocks[1])
        ])
        
        self.up2_1 = Upsample(int(embed_dim * 2**1))
        self.decoder_level1 = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**1), heads[0], ffn_expansion_factor, bias)
            for _ in range(decoder_num_blocks[2])
        ])
        
        # Refinement
        self.refinement = nn.ModuleList([
            CAMETransformerBlock(int(embed_dim * 2**1), heads[0], ffn_expansion_factor, bias)
            for _ in range(decoder_num_blocks[-1])
        ])
        
        # Output
        self.output = nn.Conv2d(int(embed_dim * 2**1), out_channels, k_s, 1, k_s//2, bias=bias)
    
    def forward(self, inp_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Training: returns dict with output + auxiliary info for losses.
        Inference: returns enhanced image tensor directly.
        """
        # =============================================================
        # Step 1: CAMT — transform to adaptive manifold
        # =============================================================
        camt_out = self.camt(inp_img)
        manifold = camt_out['manifold']       # (B, 3, H, W)
        descriptor = camt_out['descriptor']   # (B, D)
        confidence = camt_out['confidence']   # (B, 1, H, W)
        
        # =============================================================
        # Step 2: ManifoldAdaptiveIllumination (replaces SAI2E)
        # Integral image computed in ADAPTIVE manifold space
        # =============================================================
        illum_guide_1 = self.manifold_illumination(manifold, descriptor)  # (B, 3, H, W)
        
        # =============================================================
        # Step 3: Observability estimation (drives attention routing)
        # =============================================================
        observability_1 = self.observability_estimator(inp_img, confidence)  # (B, 1, H, W)
        
        # =============================================================
        # Encoder
        # =============================================================
        # Level 1
        enc1 = self.patch_embed(manifold)
        for block in self.encoder_level1:
            enc1 = block(enc1, illum_guide_1, observability_1)
        out_enc1 = enc1
        
        # Level 2
        enc2 = self.down1_2(out_enc1)
        illum_2 = self.illum_down1_2(illum_guide_1)
        obs_2 = self.obs_down1_2(observability_1)
        for block in self.encoder_level2:
            enc2 = block(enc2, illum_2, obs_2)
        out_enc2 = enc2
        
        # Level 3
        enc3 = self.down2_3(out_enc2)
        illum_3 = self.illum_down2_3(illum_2)
        obs_3 = self.obs_down2_3(obs_2)
        for block in self.encoder_level3:
            enc3 = block(enc3, illum_3, obs_3)
        out_enc3 = enc3
        
        # Level 4 (bottleneck)
        enc4 = self.down3_4(out_enc3)
        illum_4 = self.illum_down3_4(illum_3)
        obs_4 = self.obs_down3_4(obs_3)
        for block in self.latent:
            enc4 = block(enc4, illum_4, obs_4)
        
        # =============================================================
        # Step 4: CounterfactualDisentanglement at bottleneck
        # =============================================================
        disentangle_out = self.disentangle(enc4)
        latent = disentangle_out['output']
        
        # =============================================================
        # Decoder
        # =============================================================
        for block in self.decoder_latent:
            latent = block(latent, illum_4, obs_4)
        
        dec3 = self.up4_3(latent)
        dec3 = torch.cat([dec3, out_enc3], 1)
        dec3 = self.reduce_chan_level3(dec3)
        for block in self.decoder_level3:
            dec3 = block(dec3, illum_3, obs_3)
        
        dec2 = self.up3_2(dec3)
        dec2 = torch.cat([dec2, out_enc2], 1)
        dec2 = self.reduce_chan_level2(dec2)
        for block in self.decoder_level2:
            dec2 = block(dec2, illum_2, obs_2)
        
        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, out_enc1], 1)
        for block in self.decoder_level1:
            dec1 = block(dec1, illum_guide_1, observability_1)
        
        # Refinement
        for block in self.refinement:
            dec1 = block(dec1, illum_guide_1, observability_1)
        
        # Output + residual (residual on original input, not manifold)
        output = self.output(dec1) + inp_img
        output = output.clamp(0, 1)
        
        # =============================================================
        # Pack results
        # =============================================================
        if self.training:
            # Cycle consistency: reconstruct input from manifold
            cycle_recon = self.camt.inverse(manifold)
            
            return {
                'output': output,
                'cycle_recon': cycle_recon,
                'observability': observability_1,
                'content_features': disentangle_out['content'],
                'degradation_features': disentangle_out['degradation'],
                'cf_features': disentangle_out.get('cf_features', []),
                'descriptor': descriptor,
            }
        else:
            return output
