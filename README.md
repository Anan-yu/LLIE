# SAIGFormer and CAME-SAIGFormer

This repository retains the original **SAIGFormer** baseline and includes the
experimental **CAME-SAIGFormer** extension for low-light image enhancement.
CAME-SAIGFormer is research code under active validation; no unmeasured
PSNR/SSIM values or state-of-the-art claims are made for the extension.

## CAME-SAIGFormer

CAME-SAIGFormer preserves the SAIGFormer encoder-decoder structure and studies
four complementary components:

- **Camera-Adaptive Manifold Transform (CAMT):** maps RGB into an invertible
  luminance/chroma representation. A camera/degradation descriptor conditions a
  monotonic luminance spline and a chroma-plane rotation.
- **Manifold-Adaptive Illumination:** estimates non-uniform illumination in the
  adaptive manifold, conditioned on the descriptor.
- **Observability-Conditioned Attention:** uses full-image spatial statistics to
  compute cross-channel correlations and retains non-uniform illumination
  information through spatial illumination gates. A recoverability map routes
  local and global feature paths.
- **Counterfactual Degradation Intervention:** intervenes in a learned
  degradation subspace and constrains re-encoded content to remain stable. This
  is a latent degradation intervention, not an explicit simulation of the
  physical camera exposure process.

The auxiliary objectives include CAMT cycle consistency, content invariance,
content/degradation disentanglement, intervention-direction diversity,
observability smoothness, and reference-ambiguity-aware exposure distribution
(RAED). Standard L1/SSIM reconstruction remains configured separately through
`pixel_opt`.

## Ablations

Architecture switches are under `network_g`:

```yaml
use_camt: true
use_manifold_illumination: true
use_observability: true
use_counterfactual: true
use_selective_skip_fusion: false
```

Auxiliary loss switches are under `train.came_loss_opt`:

```yaml
use_raed: true
use_cycle: true
use_disentangle: true
use_intervention_diversity: true
```

All switches default to `true`. Disable one switch at a time and keep the data
split, seed, schedule, checkpoint policy, and metric implementation fixed for a
controlled ablation.

## Environment Setup

```bash
conda create --name saigformer python=3.10
conda activate saigformer
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib scikit-learn scikit-image opencv-python natsort h5py tqdm tensorboard
pip install torchmetrics thop lmdb numpy pyyaml requests scipy yapf typing triton lpips einops
python setup.py develop --no_cuda_ext
```

## Data and Output Paths

Before training or evaluation, replace the placeholders in
`Options/CAME_SAIGFormer_lolv1.yml`:

```yaml
dataroot_lq: /path/to/LOL-v1/our485/low
dataroot_gt: /path/to/LOL-v1/our485/high
```

The example output root is repository-relative (`./outputs`) and can be changed
to a writable experiment directory.

## Training

Train CAME-SAIGFormer on LOL-v1:

```bash
python basicsr/train.py --opt Options/CAME_SAIGFormer_lolv1.yml
```

The committed 200K-iteration schedule is an experiment configuration, not
evidence of convergence. Monitor validation metrics and visual failure cases,
and compare against the baseline using identical settings.

### LOL-v1 OCSF stage-2 fine-tuning

`Options/CAME_SAIGFormer_lolv1_ocsf_finetune.yml` starts from the reproduced
24.71 checkpoint and enables Observability-Calibrated Selective Skip Fusion
(OCSF). Each skip correction is zero initialized, so loading the old checkpoint
with `strict_load_g: false` initially preserves the pretrained network output.
The 30K schedule uses a single low-learning-rate cosine cycle, EMA validation,
a reconstruction-dominant loss, and cosine decay of CAME auxiliary losses.

Place the reproduced checkpoint at the path configured by
`pretrain_network_g`, then run:

```bash
python basicsr/train.py \
  --opt Options/CAME_SAIGFormer_lolv1_ocsf_finetune.yml
```

Best checkpoints produced with EMA contain both `params` and `params_ema`.
Use `params_ema` for evaluation because validation and best-model selection use
the EMA network.

The original SAIGFormer configurations remain available, for example:

```bash
python basicsr/train.py --opt Options/SAIGFormer_lolv1.yml
```

## Testing

Use a CAME-SAIGFormer checkpoint with the matching CAME configuration:

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/CAME_SAIGFormer_lolv1.yml \
  --weights /path/to/came_checkpoint.pth \
  --dataset LOL_v1
```

The test script requires CUDA in its current form. It does **not** apply
ground-truth mean correction by default. The legacy `--GT_mean` flag remains an
optional compatibility setting; any formal comparison must explicitly report
whether it was used.

Original SAIGFormer examples:

```bash
# LOL_v1
python Enhancement/test_from_dataset.py --opt Options/SAIGFormer_lolv1.yml --weights pretrained_weights/lolv1_pretrained_weight.pth --dataset LOL_v1

# LOL_v2_real
python Enhancement/test_from_dataset.py --opt Options/SAIGFormer_lolv2_real.yml --weights pretrained_weights/lolv2_real_pretrained_weight.pth --dataset LOL_v2_real

# LOL_v2_syn
python Enhancement/test_from_dataset.py --opt Options/SAIGFormer_lolv2_syn.yml --weights pretrained_weights/lolv2_syn_pretrained_weight.pth --dataset LOL_v2_syn

# SID
python Enhancement/test_from_dataset.py --opt Options/SAIGFormer_SID.yml --weights pretrained_weights/sid_pretrained_weight.pth --dataset SID

# SMID
python Enhancement/test_from_dataset.py --opt Options/SAIGFormer_SMID.yml --weights pretrained_weights/smid_pretrained_weight.pth --dataset SMID

# LOL_Blur
python Enhancement/test_from_dataset.py --opt Options/SAIGFormer_lol_blur.yml --weights pretrained_weights/lol_blur_pretrained_weight.pth --dataset LOL_Blur
```

## Original SAIGFormer

![SAIGFormer framework](./figure/architecture.png)

### Abstract

Recent Transformer-based low-light enhancement methods have made promising
progress in recovering global illumination. However, they still struggle with
non-uniform lighting scenarios, such as backlit and shadow, appearing as
over-exposure or inadequate brightness restoration. To address this challenge,
the original work presents a Spatially-Adaptive Illumination-Guided Transformer
(SAIGFormer) framework that enables accurate illumination restoration.
Specifically, it proposes a dynamic integral image representation to model
spatially varying illumination and constructs a Spatially-Adaptive Integral
Illumination Estimator (SAI²E). It also introduces an Illumination-Guided
Multi-head Self-Attention mechanism that uses illumination to calibrate
lightness-relevant features. The original SAIGFormer paper reports experiments
on five standard low-light datasets and the LOL-Blur cross-domain benchmark,
including non-uniform illumination and cross-dataset generalization results.
Those baseline-paper claims are distinct from the experimental CAME-SAIGFormer
extension documented above.

### Results and Pretrained Weights

The original visualization results can be downloaded from
[Baidu Pan](https://pan.baidu.com/s/1qpIyvWSQG77wCJsqpkPJWQ). The original
pretrained weights can be downloaded from
[Baidu Pan](https://pan.baidu.com/s/1LyUHijJ5oXQhcyiKYWiNZQ). The extraction code
is `0909`.

## Acknowledgments

Thanks to the [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer.git)
repository. The original SAIGFormer source attribution, pretrained assets, and
paper-oriented material are retained above.
