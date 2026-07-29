"""CAME-SAIGFormer for low-light image enhancement.

The model retains SAIGFormer's encoder-decoder backbone and augments it with a
camera-adaptive manifold, manifold illumination, recoverability-aware routing,
and counterfactual degradation interventions. Each research component has an
explicit ablation switch.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from basicsr.models.archs.SAI2E import SAI2E
from basicsr.models.archs.came_modules import (
    CAMT,
    CAMETransformerBlock,
    CounterfactualDisentanglement,
    Downsample,
    ManifoldAdaptiveIllumination,
    ObservabilityEstimator,
    OverlapPatchEmbed,
    Upsample,
)


class CAME_SAIGFormer(nn.Module):
    """Camera-adaptive, illumination-guided restoration network."""

    def __init__(
        self,
        embed_dim: int = 32,
        k_s: int = 3,
        encoder_num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8),
        decoder_num_blocks: Tuple[int, int, int, int] = (6, 6, 4, 4),
        ffn_expansion_factor: float = 2.66,
        heads: Tuple[int, int, int, int] = (1, 2, 4, 8),
        train_patch: int = 128,
        eps: float = 0.1,
        descriptor_dim: int = 64,
        num_interventions: int = 2,
        use_camt: bool = True,
        use_manifold_illumination: bool = True,
        use_observability: bool = True,
        use_counterfactual: bool = True,
    ):
        super().__init__()
        if not (
            len(encoder_num_blocks)
            == len(decoder_num_blocks)
            == len(heads)
            == 4
        ):
            raise ValueError("Encoder, decoder, and head settings need four levels.")
        self.descriptor_dim = descriptor_dim
        self.use_camt = use_camt
        self.use_manifold_illumination = use_manifold_illumination
        self.use_observability = use_observability
        self.use_counterfactual = use_counterfactual
        inp_channels = 3
        out_channels = 3
        bias = False

        self.camt = (
            CAMT(descriptor_dim=descriptor_dim, num_bins=8)
            if use_camt
            else None
        )
        if use_manifold_illumination:
            self.manifold_illumination = ManifoldAdaptiveIllumination(
                in_channels=inp_channels,
                descriptor_dim=descriptor_dim,
                train_patch=train_patch,
                eps=eps,
            )
            self.rgb_illumination = None
        else:
            self.manifold_illumination = None
            self.rgb_illumination = SAI2E(
                in_channels=inp_channels, train_patch=train_patch, eps=eps
            )
        self.observability_estimator = (
            ObservabilityEstimator() if use_observability else None
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, embed_dim, bias=bias)
        self.encoder_level1 = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim, heads[0], ffn_expansion_factor, bias
                )
                for _ in range(encoder_num_blocks[0])
            ]
        )
        self.illum_down1_2 = nn.Conv2d(
            inp_channels,
            inp_channels,
            4,
            2,
            1,
            bias=bias,
            groups=inp_channels,
        )
        self.obs_down1_2 = nn.AvgPool2d(2, 2)
        self.down1_2 = Downsample(embed_dim)
        self.encoder_level2 = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 2, heads[1], ffn_expansion_factor, bias
                )
                for _ in range(encoder_num_blocks[1])
            ]
        )
        self.illum_down2_3 = nn.Conv2d(
            inp_channels,
            inp_channels,
            4,
            2,
            1,
            bias=bias,
            groups=inp_channels,
        )
        self.obs_down2_3 = nn.AvgPool2d(2, 2)
        self.down2_3 = Downsample(embed_dim * 2)
        self.encoder_level3 = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 4, heads[2], ffn_expansion_factor, bias
                )
                for _ in range(encoder_num_blocks[2])
            ]
        )
        self.illum_down3_4 = nn.Conv2d(
            inp_channels,
            inp_channels,
            4,
            2,
            1,
            bias=bias,
            groups=inp_channels,
        )
        self.obs_down3_4 = nn.AvgPool2d(2, 2)
        self.down3_4 = Downsample(embed_dim * 4)
        self.latent = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 8, heads[3], ffn_expansion_factor, bias
                )
                for _ in range(encoder_num_blocks[3])
            ]
        )
        self.disentangle = (
            CounterfactualDisentanglement(
                dim=embed_dim * 8, num_interventions=num_interventions
            )
            if use_counterfactual
            else None
        )

        self.decoder_latent = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 8, heads[3], ffn_expansion_factor, bias
                )
            ]
        )
        self.up4_3 = Upsample(embed_dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(
            embed_dim * 8, embed_dim * 4, 1, bias=bias
        )
        self.decoder_level3 = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 4, heads[2], ffn_expansion_factor, bias
                )
                for _ in range(decoder_num_blocks[0])
            ]
        )
        self.up3_2 = Upsample(embed_dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(
            embed_dim * 4, embed_dim * 2, 1, bias=bias
        )
        self.decoder_level2 = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 2, heads[1], ffn_expansion_factor, bias
                )
                for _ in range(decoder_num_blocks[1])
            ]
        )
        self.up2_1 = Upsample(embed_dim * 2)
        self.decoder_level1 = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 2, heads[0], ffn_expansion_factor, bias
                )
                for _ in range(decoder_num_blocks[2])
            ]
        )
        self.refinement = nn.ModuleList(
            [
                CAMETransformerBlock(
                    embed_dim * 2, heads[0], ffn_expansion_factor, bias
                )
                for _ in range(decoder_num_blocks[3])
            ]
        )
        self.output = nn.Conv2d(
            embed_dim * 2, out_channels, k_s, 1, k_s // 2, bias=bias
        )

    def _prepare_manifold(
        self, inp_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.camt is None:
            batch_size, _, height, width = inp_img.shape
            descriptor = inp_img.new_zeros((batch_size, self.descriptor_dim))
            confidence = inp_img.new_ones((batch_size, 1, height, width))
            return inp_img, descriptor, confidence
        camt_output = self.camt(inp_img)
        return (
            camt_output["manifold"],
            camt_output["descriptor"],
            camt_output["confidence"],
        )

    def forward(
        self, inp_img: torch.Tensor
    ) -> Union[Dict[str, object], torch.Tensor]:
        manifold, descriptor, confidence = self._prepare_manifold(inp_img)
        if self.manifold_illumination is not None:
            illumination_1 = self.manifold_illumination(manifold, descriptor)
        else:
            # The baseline SAI2E fallback is defined on the original RGB input.
            illumination_1 = self.rgb_illumination(inp_img)

        if self.observability_estimator is not None:
            observability_1 = self.observability_estimator(inp_img, confidence)
        else:
            # Fixed routing is the observability ablation mode.
            observability_1 = inp_img.new_full(
                (inp_img.shape[0], 1, inp_img.shape[2], inp_img.shape[3]), 0.5
            )

        encoder_1 = self.patch_embed(manifold)
        for block in self.encoder_level1:
            encoder_1 = block(encoder_1, illumination_1, observability_1)
        output_encoder_1 = encoder_1

        encoder_2 = self.down1_2(output_encoder_1)
        illumination_2 = self.illum_down1_2(illumination_1)
        observability_2 = self.obs_down1_2(observability_1)
        for block in self.encoder_level2:
            encoder_2 = block(encoder_2, illumination_2, observability_2)
        output_encoder_2 = encoder_2

        encoder_3 = self.down2_3(output_encoder_2)
        illumination_3 = self.illum_down2_3(illumination_2)
        observability_3 = self.obs_down2_3(observability_2)
        for block in self.encoder_level3:
            encoder_3 = block(encoder_3, illumination_3, observability_3)
        output_encoder_3 = encoder_3

        encoder_4 = self.down3_4(output_encoder_3)
        illumination_4 = self.illum_down3_4(illumination_3)
        observability_4 = self.obs_down3_4(observability_3)
        for block in self.latent:
            encoder_4 = block(encoder_4, illumination_4, observability_4)

        if self.disentangle is not None:
            disentangle_output = self.disentangle(encoder_4)
            latent = disentangle_output["output"]
        else:
            latent = encoder_4
            disentangle_output = {
                "content": None,
                "degradation": None,
                "cf_features": [],
                "cf_content_features": [],
                "intervention_vectors": None,
            }

        for block in self.decoder_latent:
            latent = block(latent, illumination_4, observability_4)
        decoder_3 = self.up4_3(latent)
        decoder_3 = self.reduce_chan_level3(
            torch.cat([decoder_3, output_encoder_3], dim=1)
        )
        for block in self.decoder_level3:
            decoder_3 = block(decoder_3, illumination_3, observability_3)

        decoder_2 = self.up3_2(decoder_3)
        decoder_2 = self.reduce_chan_level2(
            torch.cat([decoder_2, output_encoder_2], dim=1)
        )
        for block in self.decoder_level2:
            decoder_2 = block(decoder_2, illumination_2, observability_2)

        decoder_1 = self.up2_1(decoder_2)
        decoder_1 = torch.cat([decoder_1, output_encoder_1], dim=1)
        for block in self.decoder_level1:
            decoder_1 = block(decoder_1, illumination_1, observability_1)
        for block in self.refinement:
            decoder_1 = block(decoder_1, illumination_1, observability_1)

        output = (self.output(decoder_1) + inp_img).clamp(0, 1)
        if not self.training:
            return output

        cycle_reconstruction: Optional[torch.Tensor]
        if self.camt is not None:
            cycle_reconstruction = self.camt.inverse(manifold, descriptor)
        else:
            cycle_reconstruction = None
        return {
            "output": output,
            "cycle_recon": cycle_reconstruction,
            "observability": observability_1,
            "content_features": disentangle_output["content"],
            "degradation_features": disentangle_output["degradation"],
            "cf_features": disentangle_output["cf_features"],
            "cf_content_features": disentangle_output["cf_content_features"],
            "intervention_vectors": disentangle_output["intervention_vectors"],
            "descriptor": descriptor,
        }
