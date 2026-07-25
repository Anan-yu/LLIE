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
from os import path as osp
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
    load_amp = True
except:
    load_amp = False


class CAMEModel(BaseModel):
    """Training model for CAME-SAIGFormer.
    
    Extends the standard ImageCleanModel to handle:
    - Dict output from CAME_SAIGFormer (output, cf_outputs, camt_info, observability)
    - CAME-specific losses (content invariance, cycle, RAED, observability smoothness)
    - Standard reconstruction losses (L1, SSIM)
    """
    
    def __init__(self, opt):
        super(CAMEModel, self).__init__(opt)
        
        # Mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')
        
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
            rec_weight=came_loss_opt.get('rec_weight', 1.0),
            ssim_weight=came_loss_opt.get('ssim_weight', 1.0),
            content_inv_weight=came_loss_opt.get('content_inv_weight', 0.1),
            cycle_weight=came_loss_opt.get('cycle_weight', 0.05),
            raed_weight=came_loss_opt.get('raed_weight', 0.05),
            obs_smooth_weight=came_loss_opt.get('obs_smooth_weight', 0.01),
        ).to(self.device)
        
        # Progressive loss scheduling: warmup CAME losses after N iterations
        self.came_loss_warmup_iter = came_loss_opt.get('warmup_iter', 5000)
        
        # Optimizer and scheduler
        self.setup_optimizers()
        self.setup_schedulers()
        
        # SAM optimizer support
        optim_type = train_opt['optim_g'].pop('type', None)
        self.sam_optim = False
        if optim_type is not None and optim_type == 'SAM':
            self.optimizer_g = optimizer.SAM(
                self.optimizer_g.param_groups, self.optimizer_g, **train_opt['optim_g'])
            self.optimizers.pop(-1)
            self.optimizers.append(self.optimizer_g)
            self.sam_optim = True
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
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
    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        with autocast(device_type='cuda', enabled=self.use_amp):
            # Forward pass - returns dict during training
            model_output = self.net_g(self.lq)
            
            # Handle both dict (training) and tensor (inference fallback)
            if isinstance(model_output, dict):
                output = model_output['output']
            else:
                output = model_output
                model_output = {'output': output}
            
            output = torch.clamp(output, 0, 1)
            model_output['output'] = output
            self.output = output
            
            loss_dict = OrderedDict()
            
            # Standard pixel losses
            l_pix = 0.
            for loss_fn in self.cri_pix:
                l_pix += loss_fn(output, self.gt)
            loss_dict['l_pix'] = l_pix
            
            # CAME-specific losses (with warmup)
            if current_iter >= self.came_loss_warmup_iter:
                came_losses = self.came_loss(model_output, self.gt, self.lq)
                for k, v in came_losses.items():
                    if k != 'l_rec':  # Avoid double-counting reconstruction
                        loss_dict[k] = v
                l_total = l_pix + came_losses['l_total'] - came_losses['l_rec']
            else:
                l_total = l_pix
        
        self.amp_scaler.scale(l_total).backward()
        self.amp_scaler.unscale_(self.optimizer_g)
        
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        if self.sam_optim:
            def closure(losses, model, lq, gt, amp_scaler):
                out = model(lq)
                if isinstance(out, dict):
                    out = out['output']
                pred = torch.clamp(out, 0, 1)
                l = sum(loss(pred, gt) for loss in losses)
                amp_scaler.scale(l).backward()
                return l
            self.amp_scaler.step(self.optimizer_g, closure, self.cri_pix,
                               self.net_g, self.lq, self.gt, self.amp_scaler)
        else:
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
            raw_model.eval()
            with torch.no_grad():
                pred = raw_model(img)
            if isinstance(pred, dict):
                pred = pred['output']
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = torch.clamp(pred, 0, 1)
            self.net_g.train()
    
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
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
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
