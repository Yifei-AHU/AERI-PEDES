from model import objectives
from .clip_model import ResidualAttentionBlock, ResidualCrossAttentionBlock, Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'fta' in args.loss_names:  
            self.num_query = 4
            self.query = nn.Parameter(torch.randn(self.num_query, self.embed_dim))
            
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            self.mlp_logsigma2 = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim*2),
                nn.ReLU(),
                nn.Linear(self.embed_dim*2, self.embed_dim)
            )

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x, None)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def compute_fuzzy_membership(self, A, B):  # compute_fuzzy_membership v3

        A = A.half() # [B，K，D]
        B = B.half() # [B, D]

        Qn_norm  = F.normalize(A.half(), dim=-1)            # [B, K, D]
        Tn_norm  = F.normalize(B.half(), dim=-1) # [B, D]

        log_sigma2 = self.mlp_logsigma2(Tn_norm) 
        sigma2 = torch.exp(log_sigma2)
        sigma2 = torch.clamp(sigma2, min=1e-6)

        Qn_exp = Qn_norm.unsqueeze(2)  
        Tn_exp = Tn_norm.unsqueeze(0).unsqueeze(0)  
        r_dim = Qn_exp * Tn_exp 

        sigma2_exp = sigma2.unsqueeze(0).unsqueeze(0)
        mu_dim = torch.exp(- ((1.0 - r_dim) ** 2) / (2 * sigma2_exp ** 2))  # [B, K, B, D]
        mu_mean = mu_dim.mean(dim=-1)  # [Bq, K, Bt]
        membership = mu_mean.transpose(1, 2).contiguous()  # [B, B, K]

        return membership
    
    def entropy_maximize_loss_from_mu(self, member, dim=-1, eps=1e-8):
        # mu: [Bq, Bt, K]
        member = member.clamp(min=0.0)
        p = member / (member.sum(dim=dim, keepdim=True) + eps)  # p is prob over Bt
        entropy = - (p * torch.log(p + eps)).sum(dim=dim)  # shape [Bq, K]
        loss = - entropy.mean()  # minimize loss => maximize mean entropy
        return loss

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        ground_images = batch['ground_imgs']
        # ground_images = None
        caption_ids = batch['caption_ids']
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            image_feats, ground_image_feats, text_feats = self.base_model(images, ground_images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        if ground_image_feats is not None:
            g_i_feats = ground_image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale

        if 'cda' in self.current_task:
            ret.update({'cda_loss': objectives.compute_selective_align_loss(i_feats, g_i_feats, t_feats, batch['pids'], logit_scale)})

        if 'fta' in self.current_task:
            B = text_feats.shape[0]
            query_expand = self.query.unsqueeze(0).expand(B, -1, -1)

            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                Q_v = self.cross_former(query_expand.half(), image_feats, image_feats) 
                Q_t = self.cross_former(query_expand.half(), text_feats, text_feats)  # 

            # v2 cross-modal membership
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                mu_t2v = self.compute_fuzzy_membership(Q_t, t_feats)
                mu_v2t = self.compute_fuzzy_membership(Q_v, i_feats)

            Q_t = F.normalize(Q_t, dim=-1) # [b,4,512]
            Q_v = F.normalize(Q_v, dim=-1) # [b,4,512]

            t2v_simi = torch.einsum('bkd,Bkd->bBk', Q_t, Q_v)
            v2t_simi = torch.einsum('bkd,Bkd->bBk', Q_v, Q_t)
            mu_and = mu_t2v * mu_v2t
            S_t2v = (t2v_simi * mu_and).mean(dim=-1)
            S_v2t = (v2t_simi * mu_and).mean(dim=-1) 

            ret.update({'fta_loss':0.5*objectives.compute_fa_loss(S_t2v, S_v2t, batch['pids'], logit_scale)})

        return ret


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
