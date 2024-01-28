from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.classification import BinaryAccuracy
from wandb.sdk.wandb_run import Run

from .matcher import MatchERT
from src.utils.utils import is_master_process
from src.sota.delg.delg import DELG

class BaseModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_global_features: int,
        num_local_features: int,
        dropout: float = 0.0,
        detach: bool = False,
        norm_layer: Optional[str] = None,
        normalize: bool = False,
        set_bn_eval: bool = False,
        remap: bool = False,
        normalize_weight: bool = False,
        ert_seq_len: int = 102,
        ert_dim_feedforward: int = 1024,
        ert_nhead: int = 4,
        ert_num_encoder_layers: int = 6,
        ert_dropout: float = 0.0,
        ert_activation: str = "relu",
        ert_normalize_before: bool = False,
        delg_pretrained: str = None,
        **kwargs
    ) -> None:
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.num_global_features = num_global_features
        self.num_local_features = num_local_features
        self.detach = detach
        self.normalize = normalize
        self.set_bn_eval = set_bn_eval
        self.normalize_weight = normalize_weight

        self.norm_layer = nn.Identity()
        if norm_layer == "layer":
            self.norm_layer = nn.LayerNorm(self.backbone_features, elementwise_affine=False)
        if norm_layer == "batch":
            self.norm_layer = nn.BatchNorm1d(self.backbone_features, affine=False)

        self.remap = nn.Identity()
        if remap or num_global_features != self.backbone_features:
            self.remap = nn.Linear(self.backbone_features, num_global_features)
            nn.init.zeros_(self.remap.bias)
        self.remap_local = nn.Identity()
        if num_local_features != self.backbone_features:
            self.remap_local = nn.Conv2d(
                self.backbone_features, num_local_features, kernel_size=1, stride=1, padding=0
            )
            nn.init.zeros_(self.remap_local.bias)

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(num_global_features, num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.matcher = MatchERT(
            num_classes=num_classes,
            d_global=num_global_features,
            d_model=num_local_features,
            nhead=ert_nhead,
            num_encoder_layers=ert_num_encoder_layers,
            dim_feedforward=ert_dim_feedforward,
            dropout=ert_dropout,
            activation=ert_activation,
            normalize_before=ert_normalize_before,
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.acc = BinaryAccuracy()
        
        self.delg = DELG()
        
        if delg_pretrained:
            self.delg.load_state_dict(torch.load(delg_pretrained))    
        

    def _forward(
        self,
        images,
        pairwise_matching: bool = False,
        src_global:Tensor=None,
        src_local:Tensor=None,
        tgt_global:Tensor=None,
        tgt_local:Tensor=None,
    )->Tuple[Tensor, Tensor, Tensor]:
        # forward step of original implementation
        if pairwise_matching:
            logits = self.matcher(
                src_global=src_global, src_local=src_local, tgt_global=tgt_global, tgt_local=tgt_local
            )
            return logits, (src_global, src_local), (tgt_global, tgt_local)

        g, l = self.feature_extractor(images)
        g = self.norm_layer(g)
        g = self.remap(g)
        l = self.remap_local(l)

        if self.normalize:
            g = nn.functional.normalize(g, p=2, dim=-1)

        classification_features = self.dropout(g.detach() if self.detach else g)
        classifier_weight = self.classifier.weight
        if self.normalize_weight:
            classifier_weight = F.normalize(classifier_weight)
        logits = F.linear(classification_features, classifier_weight, self.classifier.bias)

        return logits, g, l

    def forward(self, img1:Tensor, img2:Tensor)->Tensor:
        _, local_descriptors1, _, _, _ = self.delg(img1)
        _, local_descriptors2, _, _, _ = self.delg(img2)
        logits = self.matcher(src_global=None, src_local=local_descriptors1, tgt_global=None, tgt_local=local_descriptors2)
        return logits

    def calculate_matching_loss(self, logits:Tensor, labels:Tensor, run:Run)->Tensor:
        loss = self.criterion(logits, labels)
        
        if is_master_process():
            run.log({'train/loss': loss.mean().item(),
                     'train/acc': self.acc(logits, labels).item()})
        
        return loss

    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.set_bn_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self
