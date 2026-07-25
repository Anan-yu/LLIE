"""
CAME-SAIGFormer: Camera-Adaptive Manifold and Counterfactual Exposure
Intervention Network for Robust Low-Light Image Enhancement.

Built upon SAIGFormer backbone with three core innovations:
1. CAMT - Camera-Adaptive Manifold Transform (replaces fixed RGB processing)
2. CEI  - Counterfactual Exposure Intervention (training-time disentanglement)
3. OGDR - Observability-Guided Dynamic Restoration (adaptive local/global routing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

import basicsr.models.archs.transformer_block as transformer
from basicsr.models.archs.SAI2E import SAI2E
from basicsr.models.archs.came_modules import (
    CAMT,
    CounterfactualIntervention,
    ObservabilityEstimator,
    DynamicRestorationBlock,
)


class CAME_SAIGFormer(nn.Module):
    """CAME-SAIGFormer: Camera-Adaptive Manifold Enhanced SAIGFormer.
    
    Architecture overview:
        Input RGB -> CAMT (adaptive manifold) -> SAI2E (illumination guide)
                  -> Encoder (SAIGTransformer + OGDR) -> Decoder -> Output
                  + CEI (counterfactual training) + Cycle consistency
    
    Args:
        embed_dim: Base embedding dimension (default: 32)
        k_s: Output convolution kernel size (default: 3)
        encoder_num_blocks: Number of transformer blocks per encoder level
        decoder_num_blocks: Number of transformer blocks per decoder level
        ffn_expansion_factor: FFN hidden dimension multiplier
        heads: Number of attention heads per level
        train_patch: Patch size for SAI2E integral image
        eps: Clamping threshold for SAI2E modulation
        descriptor_dim: CAMT camera descriptor dimension
        num_interventions: Number of counterfactual interventions (training only)
        use_ogdr: Whether to use Observability-Guided Dynamic Restoration
        use_camt: Whether to use Camera-Adaptive Manifold Transform
        use_cei: Whether to use Counterfactual Exposure Intervention (training only)
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
        eps: float = 0,
        descriptor_dim: int = 64,
        num_interventions: int = 2,
        use_ogdr: bool = True,
        use_camt: bool = True,
        use_cei: bool = True,
    ):
        super().__init__()
        
        self.use_ogdr = use_ogdr
        self.use_camt = use_camt
        self.use_cei = use_cei
        
        inp_channels = 3
        out_channels = 3
        bias = False
        
        # =====================================================================
        # Innovation 1: CAMT - Camera-Adaptive Manifold Transform
        # =====================================================================
        if self.use_camt:
            self.camt = CAMT(descriptor_dim=descriptor_dim, num_bins=8)
        
        # =====================================================================
        # Innovation 2: CEI - Counterfactual Exposure Intervention
        # =====================================================================
        if self.use_cei:
            self.cei = CounterfactualIntervention(
                in_channels=inp_channels, 
                num_interventions=num_interventions
            )
        
        # =====================================================================
        # Innovation 3: OGDR - Observability-Guided Dynamic Restoration
        # =====================================================================
        if self.use_ogdr:
            self.observability_estimator = ObservabilityEstimator(in_channels=3)
            # OGDR blocks at decoder level 1 and refinement (full resolution)
            self.ogdr_decoder = DynamicRestorationBlock(
                dim=int(embed_dim * 2**1), 
                num_heads=heads[0],
                ffn_expansion=ffn_expansion_factor
            )
            self.ogdr_refinement = DynamicRestorationBlock(
                dim=int(embed_dim * 2**1),
                num_heads=heads[0],
                ffn_expansion=ffn_expansion_factor
            )
        
        # =====================================================================
        # SAIGFormer Backbone (preserved from original)
        # =====================================================================
        
        # SAI2E illumination guidance
        self.svp = SAI2E(in_channels=inp_channels, train_patch=train_patch, eps=eps)
        
        # Encoder
        self.patch_embed = transformer.OverlapPatchEmbed(inp_channels, embed_dim, bias=False)
        
        self.encoder_level1 = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=embed_dim, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(encoder_num_blocks[0])
        ])
        
        self.svp_down1_2 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.down1_2 = transformer.Downsample(embed_dim)
        self.encoder_level2 = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**1), num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(encoder_num_blocks[1])
        ])
        
        self.svp_down2_3 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.down2_3 = transformer.Downsample(int(embed_dim * 2**1))
        self.encoder_level3 = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**2), num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(encoder_num_blocks[2])
        ])
        
        self.svp_down3_4 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=bias, groups=inp_channels)
        self.down3_4 = transformer.Downsample(int(embed_dim * 2**2))
        self.latent = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**3), num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(encoder_num_blocks[3])
        ])
        
        # Decoder
        self.decoder_latent = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**3), num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(1)
        ])
        
        self.up4_3 = transformer.Upsample(int(embed_dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(embed_dim * 2**3), int(embed_dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**2), num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(decoder_num_blocks[0])
        ])
        
        self.up3_2 = transformer.Upsample(int(embed_dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(embed_dim * 2**2), int(embed_dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**1), num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(decoder_num_blocks[1])
        ])
        
        self.up2_1 = transformer.Upsample(int(embed_dim * 2**1))
        self.decoder_level1 = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**1), num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(decoder_num_blocks[2])
        ])
        
        # Refinement
        self.refinement = nn.ModuleList([
            transformer.SAIGTransformer(
                dim=int(embed_dim * 2**1), num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias
            ) for _ in range(decoder_num_blocks[-1])
        ])
        
        # Output projection
        self.output = nn.Conv2d(int(embed_dim * 2**1), out_channels, kernel_size=k_s, stride=1, padding=k_s//2, bias=bias)
        
    def _encode_decode(self, inp_img: torch.Tensor, svp_img_1: torch.Tensor,
                       observability: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core encoder-decoder forward pass.
        
        Args:
            inp_img: Input image (B, 3, H, W)
            svp_img_1: SAI2E illumination guide at full resolution (B, 3, H, W)
            observability: Optional observability map (B, 1, H, W)
        Returns:
            Enhanced output (B, 3, H, W)
        """
        # Encoder Level 1
        inp_enc_level1 = self.patch_embed(inp_img)
        for block in self.encoder_level1:
            inp_enc_level1 = block((inp_enc_level1, svp_img_1))
        out_enc_level1 = inp_enc_level1
        
        # Encoder Level 2
        inp_enc_level2 = self.down1_2(out_enc_level1)
        svp_img_2 = self.svp_down1_2(svp_img_1)
        for block in self.encoder_level2:
            inp_enc_level2 = block((inp_enc_level2, svp_img_2))
        out_enc_level2 = inp_enc_level2
        
        # Encoder Level 3
        inp_enc_level3 = self.down2_3(out_enc_level2)
        svp_img_3 = self.svp_down2_3(svp_img_2)
        for block in self.encoder_level3:
            inp_enc_level3 = block((inp_enc_level3, svp_img_3))
        out_enc_level3 = inp_enc_level3
        
        # Encoder Level 4 (Latent)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        svp_img_4 = self.svp_down3_4(svp_img_3)
        for block in self.latent:
            inp_enc_level4 = block((inp_enc_level4, svp_img_4))
        latent = inp_enc_level4
        
        # Decoder
        for block in self.decoder_latent:
            latent = block((latent, svp_img_4))
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        for block in self.decoder_level3:
            inp_dec_level3 = block((inp_dec_level3, svp_img_3))
        out_dec_level3 = inp_dec_level3
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        for block in self.decoder_level2:
            inp_dec_level2 = block((inp_dec_level2, svp_img_2))
        out_dec_level2 = inp_dec_level2
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        for block in self.decoder_level1:
            inp_dec_level1 = block((inp_dec_level1, svp_img_1))
        out_dec_level1 = inp_dec_level1
        
        # Refinement with SAIGTransformer
        for block in self.refinement:
            out_dec_level1 = block((out_dec_level1, svp_img_1))
        
        # OGDR: Observability-guided dynamic restoration at full resolution
        if self.use_ogdr and observability is not None:
            out_dec_level1 = self.ogdr_decoder(out_dec_level1, observability)
            out_dec_level1 = self.ogdr_refinement(out_dec_level1, observability)
        
        # Output projection + residual
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1
    
    def forward(self, inp_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        During training, returns a dict with:
            'output': Enhanced image (B, 3, H, W)
            'cf_outputs': List of counterfactual enhanced images (training only)
            'camt_info': CAMT transform information (if use_camt)
            'observability': Observability map (if use_ogdr)
            
        During inference, returns only the enhanced image tensor.
        
        Args:
            inp_img: Input low-light image (B, 3, H, W) in [0, 1]
        Returns:
            Training: Dict with multiple outputs for loss computation
            Inference: Enhanced image tensor (B, 3, H, W)
        """
        result = {}
        
        # =====================================================================
        # Step 1: CAMT - Adaptive color manifold transform
        # =====================================================================
        camt_info = None
        if self.use_camt:
            camt_info = self.camt(inp_img)
            # Use manifold representation as input to backbone
            backbone_input = camt_info['manifold']
            confidence = camt_info['confidence']
        else:
            backbone_input = inp_img
            confidence = torch.ones(inp_img.shape[0], 1, inp_img.shape[2], inp_img.shape[3],
                                   device=inp_img.device)
        
        # =====================================================================
        # Step 2: SAI2E illumination guidance (on manifold or RGB)
        # =====================================================================
        svp_img_1 = self.svp(backbone_input)
        
        # =====================================================================
        # Step 3: OGDR - Compute observability map
        # =====================================================================
        observability = None
        if self.use_ogdr:
            observability = self.observability_estimator(inp_img, confidence)
        
        # =====================================================================
        # Step 4: Main encoder-decoder restoration
        # =====================================================================
        output = self._encode_decode(backbone_input, svp_img_1, observability)
        
        # If using CAMT, the output is in manifold space; 
        # residual connection already adds backbone_input, so output is manifold-enhanced
        # We keep it in the same space as input for loss computation
        if self.use_camt:
            # The residual learning ensures output stays close to input space
            # Clamp to valid range
            output = output.clamp(0, 1)
        
        # =====================================================================
        # Step 5: CEI - Counterfactual intervention (training only)
        # =====================================================================
        cf_outputs = []
        if self.use_cei and self.training:
            counterfactuals, degradation_info = self.cei(inp_img)
            
            for cf_input in counterfactuals:
                # Process counterfactual through same pipeline
                if self.use_camt:
                    cf_camt = self.camt(cf_input)
                    cf_backbone_input = cf_camt['manifold']
                    cf_confidence = cf_camt['confidence']
                else:
                    cf_backbone_input = cf_input
                    cf_confidence = confidence
                
                cf_svp = self.svp(cf_backbone_input)
                
                cf_obs = None
                if self.use_ogdr:
                    cf_obs = self.observability_estimator(cf_input, cf_confidence)
                
                cf_output = self._encode_decode(cf_backbone_input, cf_svp, cf_obs)
                cf_outputs.append(cf_output.clamp(0, 1))
            
            result['cf_outputs'] = cf_outputs
            result['degradation_info'] = degradation_info
        
        # =====================================================================
        # Step 6: Cycle consistency (training only)
        # =====================================================================
        if self.use_camt and self.training:
            # Reconstruct input from manifold representation
            cycle_recon = self.camt.inverse(camt_info['manifold'], camt_info['descriptor'])
            result['cycle_recon'] = cycle_recon
        
        # Pack results
        if self.training:
            result['output'] = output
            result['camt_info'] = camt_info
            result['observability'] = observability
            return result
        else:
            # Inference: return only the enhanced image
            return output
