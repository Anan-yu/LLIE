# SAIGFormer and CAME-SAIGFormer

This repository retains the original **SAIGFormer** baseline and includes the
experimental **CAME-SAIGFormer** extension for low-light image enhancement.
CAME-SAIGFormer is research code under active validation; no unmeasured
PSNR/SSIM values or state-of-the-art claims are made for the extension.

## CAME-SAIGFormer

CAME-SAIGFormer preserves the SAIGFormer encoder-decoder structure and studies
five complementary components:

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
- **Reliability-Calibrated Selective Fusion (RCSF):** calibrates the
  observability map against paired restoration difficulty and uses a bounded,
  group-wise convex blend to replace unreliable shallow skip content with
  decoder-conditioned features. The default Level-1/2 modules add 14,786
  parameters (about 0.107% over CAME-SAIGFormer).

The auxiliary objectives include CAMT cycle consistency, content invariance,
content/degradation disentanglement, intervention-direction diversity,
observability smoothness, paired observability calibration, and
reference-ambiguity-aware exposure distribution (RAED). Standard L1/SSIM
reconstruction remains configured separately through `pixel_opt`.

## Ablations

Architecture switches are under `network_g`:

```yaml
use_camt: true
use_manifold_illumination: true
use_observability: true
use_counterfactual: true
use_selective_skip_fusion: false
use_reliability_calibrated_skip_fusion: false
reliability_fusion_levels: [1, 2]
```

Auxiliary loss switches are under `train.came_loss_opt`:

```yaml
use_raed: true
use_cycle: true
use_disentangle: true
use_intervention_diversity: true
use_observability_calibration: false
```

The original CAME component switches default to `true`; the new RCSF and
observability-calibration switches are opt-in and default to `false` for legacy
checkpoint compatibility. Disable one switch at a time and keep the data split,
seed, schedule, checkpoint policy, and metric implementation fixed for a
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

### LOL-v1 RCSF controlled training

The paper-oriented RCSF experiment now uses the reproduced 24.71 training
protocol: fixed `128 x 128` patches, batch size 8, 200K iterations, two cosine
cycles (`60K + 140K`), and raw-network validation without EMA. The RCSF fusion
and reliability-calibration changes remain enabled; only the training protocol
is aligned so comparisons isolate the method rather than the schedule. From the
repository root, start the full method with the short launcher:

```bash
python train.py lolv1
```

This command validates the four paired LOLv1 directories, selects GPU 0 unless
`CUDA_VISIBLE_DEVICES` is already set, and uses
`Options/CAME_SAIGFormer_lolv1_rcsf.yml`. Its experiment name ends in
`_2471_protocol`, which deliberately creates a fresh output directory instead of
resuming the older 120K RCSF run. Use `--gpu 1` to select another GPU or
`--dry-run` to verify the setup without starting training. The original explicit
`python basicsr/train.py --opt ...` entry point remains available.

The committed controlled ablations are:

| Configuration | RCSF | Reliability calibration |
| --- | --- | --- |
| `CAME_SAIGFormer_lolv1_rcsf_ablation_baseline.yml` | off | off |
| `CAME_SAIGFormer_lolv1_rcsf_ablation_fusion_only.yml` | on | off |
| `CAME_SAIGFormer_lolv1_rcsf.yml` | on | on |

Do not mix the legacy OCSF loss recipe into this comparison. After stage 1,
replace `REPLACE_WITH_YOUR_BEST` in
`CAME_SAIGFormer_lolv1_rcsf_psnr_finetune.yml` with the best stage-1 checkpoint
filename and run the isolated 20K RGB-PSNR refinement:

```bash
python basicsr/train.py \
  --opt Options/CAME_SAIGFormer_lolv1_rcsf_psnr_finetune.yml
```

The stage-2 configuration loads the raw `params` checkpoint, uses a low `1e-5`
learning rate, and decays research auxiliary objectives to zero. Its PSNR loss is intentionally
excluded from the stage-1 architecture ablation.

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

If the reproduced 24.71 checkpoint is unavailable, train OCSF from random
initialization with the dedicated 160K schedule:

```bash
python basicsr/train.py \
  --opt Options/CAME_SAIGFormer_lolv1_ocsf_scratch.yml
```

The scratch schedule uses an 80K primary cycle followed by an 80K refinement
cycle whose restart learning rate is reduced to 25%. It must not be replaced by
the 30K fine-tuning configuration, whose learning rate is too small for random
initialization.

The original SAIGFormer configurations remain available, for example:

```bash
python basicsr/train.py --opt Options/SAIGFormer_lolv1.yml
```

## Testing

Use a CAME-SAIGFormer checkpoint with the matching CAME configuration:

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/CAME_SAIGFormer_lolv1_rcsf.yml \
  --weights /path/to/came_checkpoint.pth \
  --dataset LOL_v1 \
  --param_key auto
```

`--param_key auto` prefers `params_ema` when it is available, matching the
network used for validation and best-checkpoint selection. Use
`--param_key params` only for an explicit raw-versus-EMA ablation.

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
