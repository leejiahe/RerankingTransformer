import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from .transformer import TransformerEncoder, TransformerEncoderLayer
import torch


class MatchERT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_global: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        normalize_before: bool,
    ):
        super(MatchERT, self).__init__()
        assert d_model % 2 == 0
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pos_encoder = PositionEmbeddingSine(d_model // 2, normalize=True, scale=2.0)
        self.seg_encoder = nn.Embedding(4, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_global: torch.Tensor,
        src_local: torch.Tensor,
        tgt_global: torch.Tensor,
        tgt_local: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        ##########################################################
        # Global features are not used in the final model
        # Keep the API here for future study
        ##########################################################
        # src_global = self.remap(src_global)
        # tgt_global = self.remap(tgt_global)    
        # if normalize:
        #     src_global = F.normalize(src_global, p=2, dim=-1)
        #     tgt_global = F.normalize(tgt_global, p=2, dim=-1)

        bsize, fsize, h, w = src_local.size()
        pos_embed  = self.pos_encoder(src_local.new_ones((1, h, w))).expand(bsize, fsize, h, w)
        cls_embed  = self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long)).permute(0, 2, 1)
        sep_embed  = self.seg_encoder(src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)
        src_local  = src_local.flatten(2)    + self.seg_encoder(2 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1) + pos_embed.flatten(2)
        tgt_local  = tgt_local.flatten(2)    + self.seg_encoder(3 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1) + pos_embed.flatten(2)
        # src_global = src_global.unsqueeze(1) + self.seg_encoder(4 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)
        # tgt_global = tgt_global.unsqueeze(1) + self.seg_encoder(5 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)
        
        # global features were not used in the final model
        input_feats = torch.cat([cls_embed, src_local, sep_embed, tgt_local], -1).permute(2, 0, 1)
        logits = self.encoder(input_feats)[0]
        return self.classifier(logits).view(-1)