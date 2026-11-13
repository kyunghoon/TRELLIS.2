import os
from typing import *
import torch


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _cfg_can_batch(self, x_t, cond, neg_cond, kwargs) -> bool:
        """
        Whether cond/neg_cond can be evaluated in a single batched forward
        pass. Measured on MPS: no speedup for any model type (kernel
        launch overhead dominates), and batch-2 GEMM tiling introduces
        bf16-level rounding differences (~1e-2 relative, amplified by the
        guidance factor), so it is OFF by default. Opt in with
        TRELLIS_BATCHED_CFG=1 (dense only) or =2 (dense + sparse).
        """
        mode = os.environ.get('TRELLIS_BATCHED_CFG', '0')
        if mode == '0':
            return False
        is_sparse = not torch.is_tensor(x_t)
        if is_sparse and mode != '2':
            return False
        # x_t: dense tensor or SparseTensor-like, always batch size 1 here
        if torch.is_tensor(x_t):
            if x_t.shape[0] != 1:
                return False
        elif hasattr(x_t, 'feats') and hasattr(x_t, 'coords'):
            if x_t.shape[0] != 1:
                return False
        else:
            return False
        # cond: dense tensor or list of dense tensors
        if torch.is_tensor(cond):
            if not (torch.is_tensor(neg_cond) and cond.shape == neg_cond.shape):
                return False
        elif isinstance(cond, list):
            if not (isinstance(neg_cond, list) and len(cond) == len(neg_cond)):
                return False
        else:
            return False
        # concat_cond (tex SLat path) must be a batch-1 SparseTensor-like
        cc = kwargs.get('concat_cond')
        if cc is not None and not (hasattr(cc, 'feats') and hasattr(cc, 'coords') and cc.shape[0] == 1):
            return False
        return True

    def _cfg_batched_inputs(self, x_t, cond, neg_cond, kwargs):
        """Duplicate x_t (and concat_cond) across the batch and stack cond."""
        from ...modules.sparse import sparse_cat
        if torch.is_tensor(x_t):
            x = torch.cat([x_t, x_t], dim=0)
        else:
            x = sparse_cat([x_t, x_t], dim=0)
        if torch.is_tensor(cond):
            c = torch.cat([cond, neg_cond], dim=0)
        else:
            c = list(cond) + list(neg_cond)
        kw = dict(kwargs)
        cc = kw.pop('concat_cond', None)
        if cc is not None:
            kw['concat_cond'] = sparse_cat([cc, cc], dim=0)
        return x, c, kw

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            if self._cfg_can_batch(x_t, cond, neg_cond, kwargs):
                x, c, kw = self._cfg_batched_inputs(x_t, cond, neg_cond, kwargs)
                pred = super()._inference_model(model, x, t, c, **kw)
                if torch.is_tensor(x_t):
                    pred_pos, pred_neg = pred.chunk(2, dim=0)
                else:
                    from ...modules.sparse import sparse_unbind
                    pred_pos, pred_neg = sparse_unbind(pred, dim=0)
            else:
                pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
                pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            
            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
                
            return pred
