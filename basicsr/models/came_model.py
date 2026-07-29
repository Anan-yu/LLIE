"""
CAME-SAIGFormer Training Model
================================
Handles the dict output from CAME_SAIGFormer during training,
computing all CAME-specific losses (content invariance, cycle consistency,
RAED, observability smoothness) alongside standard reconstruction losses.
"""

import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
import glob
import os

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.models.losses.came_losses import CAMELoss

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import torch.nn.functional as F
from functools import partial
import basicsr.models.optimizer as optimizer

try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_AWARE = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_AWARE = False


class CAMEModel(BaseModel):
    """Training model for CAME-SAIGFormer.
    
    Extends the standard ImageCleanModel to handle:
    - Dict output from CAME_SAIGFormer (output, cf_outputs, camt_info, observability)
    - CAME-specific losses (content invariance, cycle, RAED, observability smoothness)
    - Standard reconstruction losses (L1, SSIM)
    """
    
    def __init__(self, opt):
        super(CAMEModel, self).__init__(opt)
        
        requested_amp = opt.get('use_amp', False)
        self.use_amp = requested_amp and self.device.type == 'cuda'
        amp_init_scale = float(opt.get('amp_init_scale', 1024.0))
        try:
            self.amp_scaler = GradScaler(
                'cuda', init_scale=amp_init_scale, enabled=self.use_amp)
        except TypeError:
            self.amp_scaler = GradScaler(
                init_scale=amp_init_scale, enabled=self.use_amp)
        logger = get_root_logger()
        if requested_amp and not self.use_amp:
            logger.warning('AMP was requested but CUDA is unavailable; using full precision.')
        logger.info(f'Automatic Mixed Precision enabled: {self.use_amp}')
        
        # Define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        
        # Load pretrained
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                            self.opt['path'].get('strict_load_g', True),
                            param_key=self.opt['path'].get('param_key', 'params'))
        
        if self.is_train:
            self.init_training_settings()

    def _autocast_context(self):
        if AMP_DEVICE_AWARE:
            return autocast(device_type=self.device.type, enabled=self.use_amp)
        return autocast(enabled=self.use_amp)
    
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # EMA
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use EMA with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
        
        # Standard pixel losses (L1, SSIM)
        self.cri_pix = []
        if train_opt.get('pixel_opt'):
            losses = train_opt['pixel_opt']
            for loss_conf in losses:
                loss_conf = deepcopy(loss_conf)
                pixel_type = loss_conf.pop('type')
                cri_pix_cls = getattr(loss_module, pixel_type)
                self.cri_pix.append(cri_pix_cls(**loss_conf).to(self.device))
        
        # CAME-specific losses
        came_loss_opt = train_opt.get('came_loss_opt', {})
        self.came_loss = CAMELoss(
            content_inv_weight=came_loss_opt.get('content_inv_weight', 0.05),
            cycle_weight=came_loss_opt.get('cycle_weight', 0.02),
            raed_weight=came_loss_opt.get('raed_weight', 0.03),
            obs_smooth_weight=came_loss_opt.get('obs_smooth_weight', 0.01),
            disentangle_weight=came_loss_opt.get('disentangle_weight', 0.01),
            intervention_diversity_weight=came_loss_opt.get(
                'intervention_diversity_weight', 0.005),
            use_raed=came_loss_opt.get('use_raed', True),
            use_cycle=came_loss_opt.get('use_cycle', True),
            use_disentangle=came_loss_opt.get('use_disentangle', True),
            use_intervention_diversity=came_loss_opt.get(
                'use_intervention_diversity', True),
        ).to(self.device)
        
        # Progressive loss scheduling: warmup CAME losses after N iterations
        self.came_loss_warmup_iter = came_loss_opt.get('warmup_iter', 5000)
        
        self.grad_clip_norm = float(train_opt.get('grad_clip_norm', 1.0))
        self.sam_optim = False
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        
        optim_config = deepcopy(train_opt['optim_g'])
        optim_type = optim_config.pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **optim_config)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **optim_config)
        elif optim_type == 'SAM':
            base_config = deepcopy(optim_config.pop('base_optimizer'))
            base_type = base_config.pop('type')
            if base_type == 'Adam':
                base_optimizer = torch.optim.Adam(optim_params, **base_config)
            elif base_type == 'AdamW':
                base_optimizer = torch.optim.AdamW(optim_params, **base_config)
            else:
                raise NotImplementedError(
                    f'SAM base optimizer {base_type} not supported.')
            self.optimizer_g = optimizer.SAM(
                optim_params, base_optimizer, **optim_config)
            self.sam_optim = True
        else:
            raise NotImplementedError(f'Optimizer {optim_type} not supported.')
        self.optimizers.append(self.optimizer_g)
    
    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def _compute_training_loss(self, current_iter):
        model_output = self.net_g(self.lq)
        if not isinstance(model_output, dict):
            model_output = {'output': model_output}
        output = model_output['output'].clamp(0, 1)
        model_output['output'] = output

        pixel_loss = output.new_zeros(())
        for loss_function in self.cri_pix:
            pixel_loss = pixel_loss + loss_function(output, self.gt)

        came_losses = self.came_loss(model_output, self.gt, self.lq)
        if self.came_loss_warmup_iter > 0:
            auxiliary_scale = min(
                1.0, max(0.0, current_iter / self.came_loss_warmup_iter))
        else:
            auxiliary_scale = 1.0
        scaled_came_losses = {
            key: value * auxiliary_scale
            for key, value in came_losses.items()
            if key != 'l_came_total'
        }
        scaled_came_losses['l_came_total'] = torch.stack(
            list(scaled_came_losses.values())).sum()
        total_loss = pixel_loss + scaled_came_losses['l_came_total']

        loss_dict = OrderedDict(l_pix=pixel_loss)
        loss_dict.update(scaled_came_losses)
        loss_dict['l_total'] = total_loss
        for name, value in loss_dict.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f'Non-finite training loss detected in {name}.')
        return output, total_loss, loss_dict

    def _clip_gradients(self):
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(), self.grad_clip_norm)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with self._autocast_context():
            output, total_loss, loss_dict = self._compute_training_loss(
                current_iter)
        self.output = output

        if self.sam_optim:
            # SAM needs two backward passes. Autocast remains supported, while
            # gradients stay unscaled so both perturbation steps share a scale.
            total_loss.backward()
            self._clip_gradients()
            self.optimizer_g.first_step(zero_grad=True)
            with self._autocast_context():
                _, second_loss, _ = self._compute_training_loss(current_iter)
            second_loss.backward()
            self._clip_gradients()
            self.optimizer_g.second_step(zero_grad=True)
        else:
            self.amp_scaler.scale(total_loss).backward()
            self.amp_scaler.unscale_(self.optimizer_g)
            self._clip_gradients()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = torch.clamp(
            self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale], 0, 1)
    
    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, dict):
                pred = pred['output']
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = torch.clamp(pred, 0, 1)
        else:
            raw_model = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
            was_training = raw_model.training
            raw_model.eval()
            try:
                with torch.no_grad():
                    pred = raw_model(img)
            finally:
                raw_model.train(was_training)
            if isinstance(pred, dict):
                pred = pred['output']
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = torch.clamp(pred, 0, 1)
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ.get('LOCAL_RANK', '0') == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.
    
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt['val']['metrics'].keys()
            }
        
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test
        
        cnt = 0
        for val_data in dataloader:
            self.feed_data(val_data)
            test()
            
            out_dict = OrderedDict()
            out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(out_dict['result'], out_dict['gt'], **opt_)
                cnt += 1
        
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric
    
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    
    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g',
                            current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)
    
    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(exp_root, save_filename)
        
        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g
            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(param_key, list) else [param_key]
            
            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict
            torch.save(save_dict, save_path)
