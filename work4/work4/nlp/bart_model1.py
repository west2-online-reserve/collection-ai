
import os 
import math
import random
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer,BertModel
token = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    #表示保留概率最高的前 k 个 tokens
    top_k = min(top_k, logits.size(-1))  # 确保 top_k 的值不大于 logits 的长度
    if top_k > 0:
        #如果 top_k 大于 0，函数会找出概率低于前 k 个最高概率的最后一个 token 的所有 tokens，并将这些 tokens 的 logits 值设置为 filter_value
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        #如果 top_p 大于 0.0，函数首先按降序排序 logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        #计算排序后的 logits 的累积概率
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)


        sorted_indices_to_remove = cumulative_probs > top_p
        # 找出累积概率超过 top_p 的 tokens，并将其右移一个位置，确保也包括第一个超过阈值的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits



bert_pretrain = BertModel.from_pretrained('bert-base-chinese')
bert_pretrain=bert_pretrain.to("cuda:0")

class BartConfig():
    def __init__(
        self,
        vocab_size=token.vocab_size,
        max_position_embeddings=512,
        encoder_layers=6,
        encoder_ffn_dim=3072,
        encoder_attention_heads=12,
        decoder_layers=3,
        decoder_ffn_dim=3072,
        decoder_attention_heads=12,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=768,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        gradient_checkpointing=False,
        force_bos_token_to_be_generated=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
        is_encoder_decoder=True,
        decoder_start_token_id=102,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated  # only relevant for CNN
        self.output_attentions = False


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    # 用来存储向右移动一位后的token序列。
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # input_ids中的每一行所有token向右移动一位
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将每一行的第一个token设置为decoder_start_token_id
    shifted_input_ids[:, 0] = decoder_start_token_id
    # 这行代码会查找shifted_input_ids中所有值为-100的元素，并将这些元素的值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def _make_causal_mask(input_ids_shape, dtype, past_key_values_length):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    # mask_cond + 1 将每个索引加1，并将其形状变为列向量。当mask_cond中的值小于mask_cond + 1的
    # 任意值时（即每个位置i小于j），掩码对应位置被填充为0，表示这些位置可以被当前位置看到
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
def _expand_mask(mask, dtype, tgt_len= None):
    # 用于扩展和调整给定的掩码，使其适用于处理序列到序列模型中的注意力机制
    #这使得掩码具有形状 [bsz, 1, 1, src_len]
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    #将反转掩码转换成布尔型，用于指示哪些位置需要填充。
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    #通过学习每一个位置的嵌入来捕捉序列中位置的信息，适用于处理变长的输入序列。
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        self.offset = 2
        ## 调用父类的构造函数，初始化一个嵌入层,总的嵌入数量需要加上偏移量
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape, past_key_values_length):
        bsz, seq_len = input_ids_shape[:2]
        # 创建一个从 past_key_values_length 开始，长度为 seq_len 的位置索引数组
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout= 0.0,
        is_decoder= False,
        bias= True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states= None,
        past_key_value= None,
        attention_mask= None):

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # 交叉注意力且有历史键值，直接使用历史键值
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
            # 交叉注意力，计算新的键值对
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # 自注意力且有历史键值，合并历史和当前计算的键值对
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
         key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
         value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:

            past_key_value = (key_states, value_states)
        # 调整查询、键、值的形状以适应多头注意力
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        # 计算注意力权重
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # 添加注意力掩码并调整形状
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # 应用Softmax标准化注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        # 计算最终的注意力输出
        attn_output = torch.bmm(attn_probs, value_states)
        # 调整输出形状并通过最后的投影层
        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)
        # 返回注意力输出、权重和可能更新的历史键值对
        return attn_output, attn_weights_reshaped, past_key_value



class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ):

        residual = hidden_states

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)


        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,

            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)


        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        return outputs


class BartDecoder(nn.Module):


    def __init__(self, config: BartConfig, embed_tokens= None):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):

        device = input_ids.device

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids)

        combined_attention_mask = _make_causal_mask(
            input_shape, inputs_embeds.dtype, past_key_values_length=0
        )
        combined_attention_mask = combined_attention_mask.to(device)

            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #函数生成因果掩码，这种掩码使得在每一步生成的时候，模型只能看到前面的输出
        combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, 0)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, decoder_layer in enumerate(self.layers):

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            hidden_states = layer_outputs[0]
        return hidden_states



class BartModel(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.config = config
        # self.encoder = BartEncoder(config, self.shared)
        self.encoder=bert_pretrain
        self.decoder = BartDecoder(config, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )


        if attention_mask is None:
            # build encoder mask 
            attention_mask = (input_ids > 0).float()
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        if decoder_attention_mask is None :
            decoder_attention_mask = (decoder_input_ids > 0).float()
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        return decoder_outputs, encoder_outputs

class BartForConditionalGeneration(nn.Module):

    def __init__(self, config: BartConfig):
        super().__init__()
        self.model = BartModel(config)
        self.config = config
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.tokenizer = token
        self.bos_id = token.convert_tokens_to_ids("[CLS]")
        self.eos_id = token.convert_tokens_to_ids("[SEP]")
        self.unk_id = token.convert_tokens_to_ids("[UNK]")
        self.device="cuda:0"

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        labels=None,
    ):

        return_dict = True
        if labels is not None:
            if decoder_input_ids is None:   
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        decoder_out, encoder_out = self.model(
            input_ids,
            decoder_input_ids=decoder_input_ids,
        )
        lm_logits = self.lm_head(decoder_out)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = (loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1)))

        output = (lm_logits,)
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # return masked_lm_loss,lm_logits

    def sample_generate_encoder_decoder(self, text, input_max_length=256, out_max_length=100, top_k=30, top_p=0.0, add_eos=True):

        token_out = self.tokenizer.encode(text, max_length=input_max_length)
        if len(token_out) == 2:
            token_ids = token_out[0]
        else:
            token_ids = token_out
        # 如果 add_eos 参数为False，移除token ID列表的最后一个元素，通常是结束符
        if not add_eos:
            token_ids = token_ids[:-1]
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []

        input_decoder_ids = torch.tensor(self.bos_id, device=self.device, dtype=torch.long).view(1, -1)
        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.model(input_ids=token_ids, decoder_input_ids=input_decoder_ids)[0]
                scores = self.lm_head(scores)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                #将未知token（UNK）的得分设置为负无穷，避免选择它
                logit_score[self.unk_id] = -float('Inf')
                #应用Top-K和Top-P过滤
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                #选择下一个token
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if self.eos_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                input_decoder_ids = torch.cat((input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(output_ids,skip_special_tokens=True)


    