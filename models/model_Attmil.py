# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 08:49:22 2021

@author: nghaffarilal
"""

from fastai.vision.all import *
import torch
import torch.nn as nn
import os
from typing import Tuple, Any
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def to_fixed_size_bag(bag: torch.Tensor, bag_size: int = 512) -> Tuple[torch.Tensor, int]:
    
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]
    
    # zero-pad if we don't have enough samples
    zero_padded = torch.cat((bag_samples,
                              torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1])))
    return zero_padded, min(bag_size, len(bag))

##############################################################################
    
class MILBagTransform(Transform):
        
    def __init__(self, valid_files: Iterable[os.PathLike], max_bag_size: int = 512) -> None:
        
        self.max_bag_size = max_bag_size
        self.valid = {f: self._draw(f) for f in valid_files}
        
    def encodes(self, f: os.PathLike):# -> Tuple[torch.Tensor, int]:
        return self.valid.get(f, self._draw(f))
    
    def _draw(self, f: os.PathLike) -> Tuple[torch.Tensor, int]:
        return to_fixed_size_bag(torch.load(f), bag_size=self.max_bag_size)

##############################################################################
        
def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    
    """A network calculating an embedding's importance weight.
    
    Taken from arXiv:1802.04712
    """
    n_latent = n_latent or (n_in + 1) // 2
    
    return nn.Sequential(
        nn.Linear(n_in, n_latent),
        nn.Tanh(),
        nn.Linear(n_latent, 1))


##############################################################################
    
class GatedAttention(nn.Module):
    
    """A network calculating an embedding's importance weight.
    
    Taken from arXiv:1802.04712
    """
    
    def __init__(self, n_in: int, n_latent: Optional[int] = None) -> None:
        super().__init__()
        n_latent = n_latent or (n_in + 1) // 2
        
        self.fc1 = nn.Linear(n_in, n_latent)
        self.gate = nn.Linear(n_in, n_latent)
        self.fc2 = nn.Linear(n_latent, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(h)) * torch.sigmoid(self.gate(h)))

##############################################################################

class MILModel(nn.Module):
        
    def __init__(        
        self, n_feats: int, n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        with_attention_scores: bool = False,
    ) -> None:
        """
        
        Args:
            n_feats:  The nuber of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
            with_attention_scores:  Also return attention scores on :func:`forward`. #TODO doesn't really work
        """
        super().__init__()
        
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats, 512), nn.ReLU())
        self.attention = attention or GatedAttention(512)
        self.head = head or nn.Linear(512, n_out)

        self.with_attention_scores = with_attention_scores
        
                
    def forward(self, bags_and_lens):
        bags, lens = bags_and_lens
        bags = bags.to(device)
        lens = lens.to(device)
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]
        
        embeddings = self.encoder(bags)
        
        masked_attention_scores = self._masked_attention_scores(embeddings, lens)
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)
        
        scores = self.head(weighted_embedding_sums)
        
        return (scores, masked_attention_scores) if self.with_attention_scores else scores
    
    def _masked_attention_scores(self, embeddings, lens):
        """Calculates attention scores for all bags.
        
        Returns:
            A tensor containing
              *  The attention score of instance i of bag j if i < len[j]
              *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)
        
        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = (torch.arange(bag_size)
                .repeat(bs, 1)
                .to(attention_scores.device))
        
        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)
        
        masked_attention = torch.where(
                attention_mask,
                attention_scores,
                torch.full_like(attention_scores, -1e10))
        return torch.softmax(masked_attention, dim=1)