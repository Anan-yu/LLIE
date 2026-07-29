from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from basicsr.models.came_model import CAMEModel
from basicsr.models.archs.CAME_SAIGFormer_arch import CAME_SAIGFormer
from basicsr.models.archs.came_modules import CAMT
from basicsr.models.losses.came_losses import CAMELoss
from basicsr.utils.options import parse


def _small_network_options(**overrides):
    options = {
        "type": "CAME_SAIGFormer",
        "embed_dim": 8,
        "encoder_num_blocks": (1, 1, 1, 1),
        "decoder_num_blocks": (1, 1, 1, 1),
        "heads": (1, 1, 2, 4),
        "train_patch": 32,
        "descriptor_dim": 16,
        "num_interventions": 2,
    }
    options.update(overrides)
    return options


def _small_network(**overrides):
    options = _small_network_options(**overrides)
    options.pop("type")
    return CAME_SAIGFormer(**options)


def _assert_finite_tree(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for nested in value.values():
            _assert_finite_tree(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite_tree(nested)


def _training_options(optimizer_options, use_amp=False, num_gpu=0):
    return {
        "name": "came_smoke",
        "model_type": "CAMEModel",
        "is_train": True,
        "num_gpu": num_gpu,
        "dist": False,
        "use_amp": use_amp,
        "network_g": _small_network_options(),
        "path": {
            "pretrain_network_g": None,
            "strict_load_g": True,
            "param_key": "params",
        },
        "train": {
            "ema_decay": 0,
            "total_iter": 2,
            "grad_clip_norm": 1.0,
            "pixel_opt": [
                {"type": "L1Loss", "loss_weight": 1.0, "reduction": "mean"}
            ],
            "came_loss_opt": {
                "content_inv_weight": 0.05,
                "cycle_weight": 0.02,
                "raed_weight": 0.03,
                "obs_smooth_weight": 0.01,
                "disentangle_weight": 0.01,
                "intervention_diversity_weight": 0.005,
                "warmup_iter": 1,
            },
            "optim_g": optimizer_options,
            "scheduler": {
                "type": "CosineAnnealingRestartCyclicLR",
                "periods": [2],
                "restart_weights": [1],
                "eta_mins": [1e-6],
            },
        },
    }


def test_camt_cycle_is_invertible():
    torch.manual_seed(1)
    camt = CAMT(descriptor_dim=16, num_bins=8)
    input_image = torch.rand(2, 3, 32, 32)
    transformed = camt(input_image)
    reconstructed = camt.inverse(
        transformed["manifold"], transformed["descriptor"]
    )
    cycle_error = F.l1_loss(reconstructed, input_image)

    assert reconstructed.shape == input_image.shape
    assert torch.isfinite(reconstructed).all()
    assert cycle_error.item() < 1e-4
    assert "color_basis" in dict(camt.color_rotation.named_buffers())
    assert not camt.color_rotation.color_basis.requires_grad


def test_full_network_forward_loss_backward_and_inference():
    torch.manual_seed(2)
    model = _small_network()
    input_image = torch.rand(1, 3, 32, 32)
    ground_truth = torch.rand_like(input_image)

    model.train()
    model_output = model(input_image)
    required_keys = {
        "output",
        "cycle_recon",
        "observability",
        "content_features",
        "degradation_features",
        "cf_features",
        "cf_content_features",
        "descriptor",
    }
    assert required_keys.issubset(model_output)
    assert model_output["output"].shape == input_image.shape
    assert len(model_output["cf_features"]) == 2
    assert len(model_output["cf_content_features"]) == 2
    assert model_output["observability"].amin().item() >= 0
    assert model_output["observability"].amax().item() <= 1
    _assert_finite_tree(model_output)

    came_loss = CAMELoss()
    losses = came_loss(model_output, ground_truth, input_image)
    expected_loss_keys = {
        "l_content_inv",
        "l_cycle",
        "l_raed",
        "l_obs_smooth",
        "l_disentangle",
        "l_intervention_diversity",
        "l_came_total",
    }
    assert set(losses) == expected_loss_keys
    total_loss = F.l1_loss(model_output["output"], ground_truth)
    total_loss = total_loss + losses["l_came_total"]
    assert torch.isfinite(total_loss)
    total_loss.backward()

    assert model.disentangle.intervention_vectors.grad is not None
    assert torch.isfinite(model.disentangle.intervention_vectors.grad).all()
    finite_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert finite_gradients
    assert all(torch.isfinite(gradient).all() for gradient in finite_gradients)

    model.eval()
    with torch.no_grad():
        inference_output = model(input_image)
    assert torch.is_tensor(inference_output)
    assert inference_output.shape == input_image.shape
    assert torch.isfinite(inference_output).all()


def test_selective_skip_fusion_preserves_pretrained_initial_function():
    torch.manual_seed(3)
    baseline = _small_network(use_selective_skip_fusion=False)
    enhanced = _small_network(use_selective_skip_fusion=True)
    incompatible = enhanced.load_state_dict(
        baseline.state_dict(),
        strict=False,
    )
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("skip_fusion_level")
        for key in incompatible.missing_keys
    )

    input_image = torch.rand(1, 3, 32, 32)
    baseline.eval()
    enhanced.eval()
    with torch.no_grad():
        baseline_output = baseline(input_image)
        enhanced_output = enhanced(input_image)
    torch.testing.assert_close(enhanced_output, baseline_output)

    enhanced.train()
    output = enhanced(input_image)["output"]
    output.mean().backward()
    correction_gradients = [
        enhanced.skip_fusion_level1.correction.weight.grad,
        enhanced.skip_fusion_level2.correction.weight.grad,
        enhanced.skip_fusion_level3.correction.weight.grad,
    ]
    assert all(gradient is not None for gradient in correction_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in correction_gradients)


@pytest.mark.parametrize(
    "switches",
    [
        {
            "use_camt": False,
            "use_observability": False,
            "use_counterfactual": False,
        },
        {"use_manifold_illumination": False},
    ],
)
def test_architecture_ablation_switches(switches):
    model = _small_network(**switches)
    model.train()
    input_image = torch.rand(1, 3, 32, 32)
    output = model(input_image)
    assert output["output"].shape == input_image.shape
    _assert_finite_tree(output)
    if not switches.get("use_camt", True):
        assert output["cycle_recon"] is None
        assert torch.count_nonzero(output["descriptor"]) == 0
    if not switches.get("use_observability", True):
        assert torch.all(output["observability"] == 0.5)
    if not switches.get("use_counterfactual", True):
        assert output["cf_features"] == []
        assert output["cf_content_features"] == []


@pytest.mark.parametrize(
    "optimizer_options",
    [
        {"type": "Adam", "lr": 2e-4, "betas": (0.9, 0.999)},
        {
            "type": "SAM",
            "rho": 0.05,
            "base_optimizer": {
                "type": "Adam",
                "lr": 2e-4,
                "betas": (0.9, 0.999),
            },
        },
    ],
)
def test_training_model_optimizer_scheduler_and_config_integrity(
    optimizer_options,
):
    options = _training_options(optimizer_options)
    original_optimizer = deepcopy(options["train"]["optim_g"])
    original_scheduler = deepcopy(options["train"]["scheduler"])
    model = CAMEModel(options)
    model.feed_train_data(
        {"lq": torch.rand(1, 3, 32, 32), "gt": torch.rand(1, 3, 32, 32)}
    )
    model.optimize_parameters(current_iter=1)
    model.update_learning_rate(current_iter=2)

    assert options["train"]["optim_g"] == original_optimizer
    assert options["train"]["scheduler"] == original_scheduler
    assert all(torch.isfinite(torch.tensor(value)) for value in model.log_dict.values())
    expected_auxiliary = {
        "l_content_inv",
        "l_cycle",
        "l_raed",
        "l_obs_smooth",
        "l_disentangle",
        "l_intervention_diversity",
        "l_came_total",
        "l_total",
    }
    assert expected_auxiliary.issubset(model.log_dict)

    bare_model = model.get_bare_model(model.net_g)
    assert bare_model.training
    model.nonpad_test()
    assert bare_model.training
    assert torch.isfinite(model.output).all()

    state_dict = deepcopy(bare_model.state_dict())
    reloaded = _small_network()
    reloaded.load_state_dict(state_dict, strict=True)


def test_auxiliary_loss_cosine_decay_schedule():
    options = _training_options(
        {"type": "Adam", "lr": 2e-4, "betas": (0.9, 0.999)}
    )
    options["train"]["came_loss_opt"].update(
        {
            "warmup_iter": 2,
            "decay_start_iter": 10,
            "decay_end_iter": 20,
            "min_auxiliary_scale": 0.1,
        }
    )
    model = CAMEModel(options)

    assert model._get_auxiliary_scale(1) == pytest.approx(0.5)
    assert model._get_auxiliary_scale(10) == pytest.approx(1.0)
    assert model._get_auxiliary_scale(15) == pytest.approx(0.55)
    assert model._get_auxiliary_scale(20) == pytest.approx(0.1)
    assert model._get_auxiliary_scale(30) == pytest.approx(0.1)


def test_validation_metrics_are_weighted_by_image_count():
    options = _training_options(
        {"type": "Adam", "lr": 2e-4, "betas": (0.9, 0.999)}
    )
    options["val"] = {
        "window_size": 0,
        "metrics": {
            "psnr": {
                "type": "calculate_psnr",
                "crop_border": 0,
                "test_y_channel": False,
            }
        },
    }
    model = CAMEModel(options)

    def passthrough_test(img=None):
        model.output = model.lq.clone()

    model.nonpad_test = passthrough_test

    class ValidationDataset:
        opt = {"name": "weighted_validation"}

    class ValidationLoader:
        dataset = ValidationDataset()

        def __iter__(self):
            yield {
                "lq": torch.zeros(4, 3, 8, 8),
                "gt": torch.full((4, 3, 8, 8), 0.1),
            }
            yield {
                "lq": torch.zeros(1, 3, 8, 8),
                "gt": torch.full((1, 3, 8, 8), 10 ** (-0.5)),
            }

    metric = model.nondist_validation(
        ValidationLoader(), 1, None, False, True, False
    )
    assert metric == pytest.approx(18.0, abs=1e-5)


def test_committed_config_parses_and_schedule_is_consistent():
    options = parse("Options/CAME_SAIGFormer_lolv1.yml", is_train=True)
    scheduler = options["train"]["scheduler"]
    assert sum(scheduler["periods"]) == options["train"]["total_iter"]
    assert max(scheduler["eta_mins"]) < options["train"]["optim_g"]["lr"]
    assert "rec_weight" not in options["train"]["came_loss_opt"]
    assert "ssim_weight" not in options["train"]["came_loss_opt"]


def test_ocsf_finetune_config_is_checkpoint_compatible():
    options = parse(
        "Options/CAME_SAIGFormer_lolv1_ocsf_finetune.yml",
        is_train=True,
    )
    assert options["network_g"]["use_selective_skip_fusion"]
    assert options["path"]["strict_load_g"] is False
    assert options["train"]["ema_decay"] == pytest.approx(0.999)
    assert sum(options["train"]["scheduler"]["periods"]) == 30000
    assert options["train"]["came_loss_opt"]["decay_end_iter"] == 10000
    assert options["datasets"]["val"]["val_batch_size"] == 1


def test_checkpoint_and_training_state_roundtrip(tmp_path):
    options = _training_options(
        {"type": "Adam", "lr": 2e-4, "betas": (0.9, 0.999)}
    )
    models_path = tmp_path / "models"
    states_path = tmp_path / "training_states"
    models_path.mkdir()
    states_path.mkdir()
    options["path"].update(
        {
            "models": str(models_path),
            "training_states": str(states_path),
            "experiments_root": str(tmp_path),
        }
    )
    model = CAMEModel(options)
    model.feed_train_data(
        {"lq": torch.rand(1, 3, 32, 32), "gt": torch.rand(1, 3, 32, 32)}
    )
    model.optimize_parameters(current_iter=1)
    model.save(
        epoch=0,
        current_iter=1,
        best_metric={"psnr": 0.0, "iter": 0},
    )

    model_checkpoint = models_path / "net_g_1.pth"
    training_state_path = states_path / "1.state"
    assert model_checkpoint.is_file()
    assert training_state_path.is_file()

    restored = CAMEModel(deepcopy(options))
    restored.load_network(restored.net_g, str(model_checkpoint), strict=True)
    training_state = torch.load(training_state_path, weights_only=False)
    restored.resume_training(training_state)


def test_best_checkpoint_contains_evaluated_ema_weights(tmp_path):
    options = _training_options(
        {"type": "Adam", "lr": 2e-4, "betas": (0.9, 0.999)}
    )
    options["train"]["ema_decay"] = 0.9
    options["path"]["experiments_root"] = str(tmp_path)
    model = CAMEModel(options)
    model.save_best({"psnr": 25.0, "iter": 10})

    checkpoint = torch.load(
        tmp_path / "best_psnr_25.00_10.pth",
        weights_only=False,
    )
    assert set(checkpoint) == {"params", "params_ema"}
    assert checkpoint["params_ema"].keys() == checkpoint["params"].keys()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_forward_backward():
    options = _training_options(
        {"type": "Adam", "lr": 2e-4, "betas": (0.9, 0.999)},
        use_amp=True,
        num_gpu=1,
    )
    model = CAMEModel(options)
    bare_model = model.get_bare_model(model.net_g)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in bare_model.named_parameters()
    }
    model.feed_train_data(
        {"lq": torch.rand(1, 3, 32, 32), "gt": torch.rand(1, 3, 32, 32)}
    )
    model.optimize_parameters(current_iter=1)
    assert model.use_amp
    assert all(torch.isfinite(torch.tensor(value)) for value in model.log_dict.values())
    assert any(
        not torch.equal(parameters_before[name], parameter.detach())
        for name, parameter in bare_model.named_parameters()
    )
