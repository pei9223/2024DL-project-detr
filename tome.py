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
# from models.ops.modules import MSDeformAttn
# from models.ops.functions import MSDeformAttnFunction
from models.transformer import TransformerEncoderLayer, TransformerDecoderLayer, Transformer, TransformerEncoder

class ToMeTransformerEncoder(TransformerEncoder):

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output, pos, src_key_padding_mask = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos) ##
        
        r = src_key_padding_mask.shape[1] - output.shape[0] ##
        # print('r: ', r)
        src_key_padding_mask = src_key_padding_mask[:, r:] ## 因為src_key_padding_mask會比output少減一次r

        if self.norm is not None:
            output = self.norm(output)

        return output, pos, src_key_padding_mask ##


class ToMeTransformerEncoderLayer(TransformerEncoderLayer): # DeformableTransformerEncoderLayer
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

        # reshape mask
        # print('------------')
        # print('src_key_padding_mask: ', src_key_padding_mask.shape)
        # print('q: ', q.shape)
        r = src_key_padding_mask.shape[1] - q.shape[0]
        # print('r: ', r)
        src_key_padding_mask = src_key_padding_mask[:, r:]
        # print('merge src_key_padding_mask: ', src_key_padding_mask.shape)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        r = self._tome_info["e_r"].pop(0)
        # num = spatial_shapes[0][0] * spatial_shapes[0][1]
        # np.savetxt('mask.txt', np.array(src_key_padding_mask[0].cpu()))
        # print('ori mask:', src_key_padding_mask.shape)
        # print('mask:', src_key_padding_mask[..., None].shape)
        # print('mask count 0: ', torch.unique(src_key_padding_mask[0], return_counts=True))
        
        if r > 0:
            # Apply ToMe here 使用key找到要merge哪些token
            # print('ori src:', src.shape)
            # src2 = src2.transpose(0, 1)
            merge, merge_ref = bipartite_soft_matching_dim0(
                src2,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            # src2 = src2.transpose(0, 1)
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, src, self._tome_info["source"]
                )
            
            # print('before x: ', x.shape, 'pos: ', pos.shape, 'reference_points:', reference_points.shape)
            # print('merge merge ref:', merge, merge_ref)
            cur_size = self._tome_info["e_size"]

            # src = src.transpose(0, 1)
            src, self._tome_info["e_size"] = merge_wavg(merge, src, cur_size) # 將token進行merge
            
            # src = src.transpose(0, 1)
            # print('merge src:', src.shape)
            # print('ori pos:', pos.shape)
            # pos = pos.transpose(0, 1)
            pos, _ = merge_wavg(merge_ref, pos, cur_size, self._tome_info["e_size"], 'pos') # merge pos
            # pos = pos.transpose(0, 1)
            # print('merge pos:', pos.shape)
            # N, n_q, lvl, coor = reference_points.shape
            # reference_points = reference_points.reshape((N, n_q, -1))
            
            # reference_points, _ = merge_wavg(merge_ref, reference_points, cur_size, self._tome_info["size"], 'ref') # merge pos
            # reference_points = reference_points.reshape((N, -1, lvl, coor))

        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # print('forward post')

        return src, pos, src_key_padding_mask ##
    
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # print('new forward')
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        src, pos, src_key_padding_mask = self.forward_post(src, src_mask, src_key_padding_mask, pos) ##
        
        return src, pos, src_key_padding_mask ##


class ToMeTransformerDecoderLayer(TransformerDecoderLayer): # DeformableTransformerDecoderLayer
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
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        r = self._tome_info["d_r"].pop(0)
        # num = spatial_shapes[0][0] * spatial_shapes[0][1]
        # np.savetxt('mask.txt', np.array(src_key_padding_mask[0].cpu()))
        # print('ori mask:', src_key_padding_mask.shape)
        # print('mask:', src_key_padding_mask[..., None].shape)
        # print('mask count 0: ', torch.unique(src_key_padding_mask[0], return_counts=True))
        
        if r > 0:
            merge, merge_ref = bipartite_soft_matching_dim0(
                tgt2,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, tgt, self._tome_info["source"]
                )

            # print('tgt:', tgt.shape)
            # print('query_pos:', query_pos.shape)
            cur_size = self._tome_info["d_size"]
            tgt, self._tome_info["d_size"] = merge_wavg(merge, tgt, cur_size) # 將token進行merge
            query_pos, _ = merge_wavg(merge_ref, query_pos, cur_size, self._tome_info["d_size"], 'pos') # merge pos
            # print('merge tgt:', tgt.shape)
            # print('merge query_pos:', query_pos.shape)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, query_pos


# class ToMeMSDeformAttn(MSDeformAttn):
#     """
#     Modifications:
#      - Apply proportional attention
#      - Return the mean of k over heads from attention
#     """

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, size=None):
#         """
#         :param query                       (N, Length_{query}, C)
#         :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
#                                         or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
#         :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
#         :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#         :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
#         :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

#         :return output                     (N, Length_{query}, C)
#         """
#         N, Len_q, _ = query.shape
#         N, Len_in, _ = input_flatten.shape
#         # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in    note: comment

#         value = self.value_proj(input_flatten)
#         # print('ori mask:', input_padding_mask.shape)
#         # print('mask:', input_padding_mask[..., None].shape)
#         # print('mask count 0: ', torch.unique(input_padding_mask[0], return_counts=True))
#         # print('mask count 1: ', torch.unique(input_padding_mask[1], return_counts=True))

#         # np.savetxt('mask.txt', np.array(input_padding_mask[0].cpu()))
#         # with open('mask.txt', 'a') as f:
#         #     f.write(str(np.array(input_padding_mask[0].cpu())))
#         #     f.write('\n\n')

#         # print('value:', value.shape)

#         # input_padding_mask: 
#         #     True: pixel exist
#         #     False: padding to same token length, no image patch
#         if input_padding_mask is not None:
#             r = input_padding_mask.shape[1] - value.shape[1]
#             input_padding_mask = input_padding_mask[:, r:]
#             value = value.masked_fill(input_padding_mask[..., None], float(0))           # note: comment

#         # print('value: ', value.shape)

#         value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
#         attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        
#         # Apply proportional attention
#         if size is not None:
#             # print('attention_weights:', attention_weights.shape)
#             # print('size:', size.shape)
#             # print('size log:', size.log()[:, :, :, None].shape)
#             # attention_weights: [2, 22112, 8, 16]  N, Len_q, self.n_heads, self.n_levels*self.n_points (4*4)
#             attention_weights = attention_weights + size.log()[:, :, :, None]       # size: [2, 22112, 1] -> [2, 22112, 1, 1]
#         attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
#         # N, Len_q, n_heads, n_levels, n_points, 2
#         # print('ori reference_points:', reference_points.shape)
#         # print('reference_points:', reference_points[:, :, None, :, None, :].shape)
        
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
#             # print('sampling_offsets / offset_normalizer[None, None, None, :, None, :]', (sampling_offsets / offset_normalizer[None, None, None, :, None, :]).shape)
#             # print('--------------------------')
#             sampling_locations = reference_points[:, :, None, :, None, :] \
#                                  + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = reference_points[:, :, None, :, None, :2] \
#                                  + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
#         else:
#             raise ValueError(
#                 'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
#         output = MSDeformAttnFunction.apply(
#             value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
#         output = self.output_proj(output)

#         # return output, output.mean(1) 
#         return output


def make_tome_class(transformer_class):
    class ToMeDeformableTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            # num_layers = self.transformer.num_encoder_layers + self.transformer.num_decoder_layers
            self._tome_info["e_r"] = parse_r(self.transformer.num_encoder_layers, self.er) # ex: 3個blocks, 則_tome_info["r"] = [r, r, r]
            self._tome_info["d_r"] = parse_r(self.transformer.num_decoder_layers, self.dr)
            # 在class DeformableTransformer(nn.Module):加上self.blocks = num_encoder_layers + num_decoder_layers

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
    model._tome_info = {
        "e_r": model.er,
        "d_r": model.dr,
        "e_size": None,
        "d_size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    # if hasattr(model, "dist_token") and model.dist_token is not None: # 是否有dist_token這個屬性
    #     model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, TransformerEncoderLayer):
            module.__class__ = ToMeTransformerEncoderLayer
            module._tome_info = model._tome_info
        if isinstance(module, TransformerEncoder):
            module.__class__ = ToMeTransformerEncoder
            module._tome_info = model._tome_info
            # for submodule_name, submodule in module.named_modules():
            #     if isinstance(submodule, MSDeformAttn):
            #         # Replace the submodule with ToMeMSDeformAttn
            #         submodule.__class__ = ToMeMSDeformAttn
                    # setattr(module, submodule_name, ToMeMSDeformAttn())

        elif isinstance(module, TransformerDecoderLayer):
            module.__class__ = ToMeTransformerDecoderLayer
            module._tome_info = model._tome_info
        # elif isinstance(module, MSDeformAttn):
        #     module.__class__ = ToMeMSDeformAttn
    