# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Decoder for SeqTrack, modified from DETR transformer class.
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_position_embeddings, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_embeds = self.word_embeddings(x)
        embeddings = input_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class SeqTrackDecoderXL(nn.Module):

    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, attn_type=0,
                 return_intermediate_dec=False, bins=1000, num_frames=9,
                 tgt_len=5, mem_len=0, ext_len=0):
        super().__init__()
        self.bins = bins
        self.num_frames = num_frames
        self.n_layer = num_decoder_layers
        self.num_coordinates = 4 # [x,y,w,h]
        max_position_embeddings = (self.num_coordinates+1) * num_frames
        self.embedding = DecoderEmbeddings(bins+2, d_model, max_position_embeddings, dropout)
        
        # Absolute positional embeddings
        if attn_type == 0: 
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        # Relative positional embeddings
        if attn_type == 1:
            decoder_layer = RelPartialLearnableDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.body = TransformerDecoderXL(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, attn_type=attn_type)

        self._reset_parameters()

        self.d_model = d_model
        self.n_head = nhead
        self.d_head = d_model // nhead
        
        self.tgt_len = tgt_len # target length
        self.mem_len = mem_len # memory length
        self.ext_len = ext_len # extended length
        self.max_klen = tgt_len + mem_len + ext_len # maximum key length

        self.attn_type = attn_type # attention type
        self._create_params() # create parameters for bias

    def _create_params(self):
        if self.attn_type == 1: # partial learnable attention (relative)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        print("Length of hids: ", len(hids))
        print("Length of mems: ", len(mems))

        # mems is not None
        # assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems


    def _forward(self, src, seq, pos_embed, mems=None):
        # Chnage the layers to the new decoder layers
        bsz, qlen = seq.size() # batch size, query length

        tgt = self.embedding(seq).permute(1, 0, 2)  # word embedding
        print("SeqDec forward Tgt shape: ", tgt.shape)

        mlen = mems[0].size(0) if mems is not None else 0   # memory length
        klen = mlen + qlen # key length
        dec_attn_mask = torch.triu(
                tgt.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]
      
        # TODO; add positional encodings
        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bsz, 1)

        memory = src

        hs = self.body(tgt=tgt, memory=memory, r_w_bias=self.r_w_bias, r_r_bias=self.r_r_bias, pos=pos_embed, query_pos=query_embed[:len(tgt)],
                       tgt_mask=dec_attn_mask, memory_mask=None, mems=mems)
        new_mems = self._update_mems(hs, mems, mlen, qlen)

        return hs, new_mems    

    # src: encoder output, pos_embed: positional embedding, seq: target sequence
    def forward(self, src, pos_embed, seq, *mems):
        if not mems: mems = self.init_mems()

        # flatten NxCxHxW to HWxNxC
        n, bs, c = src.shape
        new_mems = None
    
        # TODO: add relative positional encoding
        if self.attn_type == 1:
            hs, new_mems = self._forward(src, seq, pos_embed, mems=mems) # hidden is the output of the decoder, new_mems is the updated memory
        else:    
            tgt = self.embedding(seq).permute(1, 0, 2) 
            query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
            query_embed = query_embed.repeat(1, bs, 1)

            memory = src

            tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device) #generate the causal mask

            # hs is the output of the decoder
            hs = self.body(tgt, memory, pos=pos_embed, query_pos=query_embed[:len(tgt)],
                        tgt_mask=tgt_mask, memory_mask=None) # pass the target, memory, and positional embedding to the decoder

        print("SeqDec forward hs: ", hs)
        print("SeqDec forward new_mems: ", new_mems)

        if new_mems is not None:
            return hs, new_mems
        return hs.transpose(1, 2)


    def inference(self, src, pos_embed, seq, vocab_embed,
                  window, seq_format):
        # flatten NxCxHxW to HWxNxC
        n, bs, c = src.shape
        memory = src
        confidence_list = []
        box_pos = [0, 1, 2, 3] # the position of bounding box
        center_pos = [0, 1]  # the position of x_center and y_center
        if seq_format == 'whxy':
            center_pos = [2, 3]

        for i in range(self.num_coordinates): # only cycle 4 times, because we do not need to predict the end token during inference
            tgt = self.embedding(seq).permute(1, 0, 2)
            query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
            query_embed = query_embed.repeat(1, bs, 1)
            tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device)

            #TODO; add relative positional encoding
            hs = self.body(tgt, memory, pos=pos_embed[:len(memory)], query_pos=query_embed[:len(tgt)],
                              tgt_mask=tgt_mask, memory_mask=None)

            # embedding --> likelihood
            out = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
            out = out.softmax(-1)
           
            if i in box_pos:
                out = out[:, :self.bins] # only include the coordinate values' confidence

            if ((i in center_pos) and (window!=None)):
                out = out * window # window penalty
     
            confidence, token_generated = out.topk(dim=-1, k=1)
            seq = torch.cat([seq, token_generated], dim=-1)
            confidence_list.append(confidence)

        out_dict = {}
        out_dict['pred_boxes'] = seq[:, -self.num_coordinates:] # Discard the START token, only get the bounding box
        out_dict['confidence'] = torch.cat(confidence_list, dim=-1)[:, :]

        return out_dict


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """

    #each token only can see tokens before them
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerDecoderXL(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, attn_type=0):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.attn_type = attn_type

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, 
                r_w_bias: Optional[Tensor] = None,
                r_r_bias: Optional[Tensor] = None,
                mems = None):
        output = tgt

        intermediate = []

        if self.attn_type == 0:
            for layer in self.layers:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)

                if self.return_intermediate:
                    intermediate.append(self.norm(output))
        elif self.attn_type == 1:
            index = 0 # index for the memory
            for layer in self.layers:
                mem_i = mems[index] if mems is not None else None
                output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos, 
                            r_w_bias=r_w_bias[index], r_r_bias=r_r_bias[index], mems=mem_i)
                print("Decoder XL output shape: ", output.shape, "for layer ", index)
                index += 1
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt2, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * self.d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * self.d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (self.d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    # w: query, r: relative positional encoding, r_w_bias: global content bias, r_r_bias: global position bias
    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        # This is the same across all batches
        r_head_k = r_head_k.view(rlen, bsz, self.n_head, self.d_head)                # qlen x n_head x d_head
        r_head_k = r_head_k[:, 0, :, :]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = RelPartialLearnableMultiHeadAttn(nhead, d_model, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # tgt: value, memory: encoder value, mems: decoder memory, r_w_bias: global content bias, r_r_bias: global position bias
    def forward(self, tgt, memory, r_w_bias, r_r_bias, tgt_mask: Optional[Tensor] = None, mems=None, 
                     memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None, partial_query: Optional[Tensor] = None):
        print("I am the new decoder layer!!!!!!!!!!!!!!")
        print("Tgt shape: ", tgt.shape)
        # TODO: replace this with relative embeddings
        tgt2 = self.self_attn(tgt, query_pos, r_w_bias, r_r_bias, attn_mask=tgt_mask, mems=mems)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_decoderXL(cfg):
    return SeqTrackDecoderXL(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=cfg.MODEL.DECODER.NHEADS,
        dim_feedforward=cfg.MODEL.DECODER.DIM_FEEDFORWARD,
        num_decoder_layers=cfg.MODEL.DECODER.DEC_LAYERS,
        normalize_before=cfg.MODEL.DECODER.PRE_NORM,
        return_intermediate_dec=False,
        bins=cfg.MODEL.BINS,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        attn_type=cfg.MODEL.DECODER.ATTN_TYPE,
        tgt_len=5,
        mem_len=cfg.MODEL.DECODER.MEM_LEN
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")