# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from ToMe.merge import bipartite_soft_matching, merge_source, merge_wavg, bipartite_soft_matching_dim0
from ToMe.utils import parse_r
from models.transformer import TransformerEncoderLayer, TransformerDecoderLayer, Transformer, TransformerEncoder

class ToMeTransformerEncoder(TransformerEncoder):

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output, pos, src_key_padding_mask = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        # modify: cut src_key_padding_mask again
        r = src_key_padding_mask.shape[1] - output.shape[0]
        src_key_padding_mask = src_key_padding_mask[:, r:] ## 因為src_key_padding_mask會比output少減一次r

        if self.norm is not None:
            output = self.norm(output)

        return output, pos, src_key_padding_mask


class ToMeTransformerEncoderLayer(TransformerEncoderLayer):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        # cut mask
        r = src_key_padding_mask.shape[1] - q.shape[0]
        src_key_padding_mask = src_key_padding_mask[:, r:]

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # apply tome option 1
        r = self._tome_info["e_r"].pop(0)        
        if r > 0:
            merge, merge_ref = bipartite_soft_matching_dim0(
                src2,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, src, self._tome_info["source"]
                )
            
            cur_size = self._tome_info["e_size"]
            # merge token
            src, self._tome_info["e_size"] = merge_wavg(merge, src, cur_size)
            # merge pos embedding
            pos, _ = merge_wavg(merge_ref, pos, cur_size, self._tome_info["e_size"], 'pos')


        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, pos, src_key_padding_mask
    
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        src, pos, src_key_padding_mask = self.forward_post(src, src_mask, src_key_padding_mask, pos)
        
        return src, pos, src_key_padding_mask


class ToMeTransformerDecoderLayer(TransformerDecoderLayer):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        r = memory_key_padding_mask.shape[1] - memory.shape[0]

        # cut memory_key_padding_mask
        memory_key_padding_mask = memory_key_padding_mask[:, r:]
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # apply tome option 2: merge object queries
        dr = self._tome_info["d_r"].pop(0)
        
        if dr > 0:
            merge, merge_ref = bipartite_soft_matching_dim0(
                tgt2,
                dr,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, tgt, self._tome_info["source"]
                )

            cur_size = self._tome_info["d_size"]
            tgt, self._tome_info["d_size"] = merge_wavg(merge, tgt, cur_size)
            query_pos, _ = merge_wavg(merge_ref, query_pos, cur_size, self._tome_info["d_size"], 'pos')

        # apply tome option 3: merge memory (encoder output)
        mr = self._tome_info["m_r"].pop(0)
        
        if mr > 0:
            merge, merge_ref = bipartite_soft_matching_dim0(
                memory,
                mr,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, memory, self._tome_info["source"]
                )

            cur_size = self._tome_info["e_size"]
            memory, self._tome_info["e_size"] = merge_wavg(merge, memory, cur_size)
            pos, _ = merge_wavg(merge_ref, pos, cur_size, self._tome_info["e_size"], 'pos')

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, query_pos, memory, pos


def make_tome_class(transformer_class):
    class ToMeDeformableTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["e_r"] = parse_r(self.transformer.num_encoder_layers, self.er) # ex: 3個blocks, 則_tome_info["r"] = [r, r, r]
            self._tome_info["d_r"] = parse_r(self.transformer.num_decoder_layers, self.dr)
            self._tome_info["m_r"] = parse_r(self.transformer.num_decoder_layers, self.mr)

            self._tome_info["e_size"] = None
            self._tome_info["d_size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeDeformableTransformer

# apply_patch(DeformableTransformer)
def apply_patch(
    model: Transformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeTransformer
    model.er = 0
    model.dr = 0
    model.mr = 0
    model._tome_info = {
        "e_r": model.er,
        "d_r": model.dr,
        "m_r": model.mr,
        "e_size": None,
        "d_size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }


    for module in model.modules():
        if isinstance(module, TransformerEncoderLayer):
            module.__class__ = ToMeTransformerEncoderLayer
            module._tome_info = model._tome_info
        if isinstance(module, TransformerEncoder):
            module.__class__ = ToMeTransformerEncoder
            module._tome_info = model._tome_info
        elif isinstance(module, TransformerDecoderLayer):
            module.__class__ = ToMeTransformerDecoderLayer
            module._tome_info = model._tome_info

    