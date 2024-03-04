
import copy
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size, # 词表大小
                 hidden_size=768, # encoder的隐藏的神经元数
                 num_hidden_layers=8,# encoder中的隐藏层数
                 num_attention_heads=12,#multi-head attention 的head数
                 intermediate_size=3072,#encoder里隐层神经元数
                 hidden_act="gelu",#隐藏层激活函数gelu,论文中gelu效果比relu更好
                 hidden_dropout_prob=0.1,#隐层dropout概率
                 attention_probs_dropout_prob=0.1,#注意力里的dropout概率
                 max_position_embeddings=512,#最大序列长度
                 type_vocab_size=2,#token_type_ids的词典大小
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):

        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # word embeding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 位置embedding, 其中max_position_embeddings就是512
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # type_vocab_size=2, 0代表当前句子, 1代表下一句, 用作NSP任务,这里还是不需要

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)#第一维度的size就是句子的长度，在训练集已经统一定义长度了
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # 按照句子长度从0到seq_length给出位置下标
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            #没下标就把下标全设为0,不进行NSP任务

        words_embeddings = self.word_embeddings(input_ids)#大小为[batch_size * seq_length * hidden_size(768)]
        position_embeddings = self.position_embeddings(position_ids)#大小为[batch_size * seq_length * hidden_size(768)]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)#大小为[batch_size * seq_length * hidden_size(768)]

        embeddings = words_embeddings + position_embeddings + token_type_embeddings#embeddings相加
        embeddings = self.LayerNorm(embeddings)#丢到BertLayerNorm层归一化
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        #隐藏层除以多头注意力的头数要求整数
        self.num_attention_heads = config.num_attention_heads
        # multi-head 头数8
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 一个head的大小：768/8=96
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #这个就是hidden_size大小

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # query矩阵, hidden_size * all_head_size (768*768) 大小的矩阵
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # key矩阵, hidden_size * all_head_size (768*768) 大小的矩阵
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # value矩阵, hidden_size * all_head_size (768*768) 大小的矩阵
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 输入x: [batch_size * seq_len * 768]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 大小为[batch_size, seq_len, 12, 64]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # 输出大小为[batch_size, 8, seq_len, 96], 8代表头数，96代表每个头的维度

    def forward(self, hidden_states, attention_mask):
        # hidden_states:  输入大小为[batch_size * seq_len * 768]
        # attention_mask: 输入大小为[batch_size * 1 * 1 * 768]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # q,k,v,输出为 batch_size * seq_len * 768

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # 转为multi-head-attention, 大小为batch_size * 12 * seq_len * 64

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 输出为 batch_size * 8 * seq_len * seq_len
        # self-attention 机制是 q, k 矩阵乘归一化得出系数后, 乘到 value 上
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#归一化
        attention_scores = attention_scores + attention_mask
        # [batch_size * 8 * seq_len * seq_len] 与 [batch_size*1*1*seq_len] 之和

        #softmax转换成概率
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        #调用self_attention
        self.output = BertSelfOutput(config)
        #调用self_attention的输出层

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        # self-attention, 大小为[batch_size * seq_len * 768]
        self.intermediate = BertIntermediate(config)
        # self-attention 中间层 [batch_size * seq_len * 更大的dim]
        self.output = BertOutput(config) #输出层

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)#调用BertLayer
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []#用于保留每层BertLayer()的输出
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)#把每一层的结果保存下来
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)#只保留最后一层的结果
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertModel(nn.Module):
    """
     input_ids: 句子中的每个字查vocab.txt所得到的下标，举个例子：tensor[101,2456,1074,......,102]
     token_type_ids: 有地方称作 segment_ids, 用于Next Sentence Predict任务
     attention_mask: 1代表有用信息，0代表padding这种无用信息
     position_ids: 代表位置下标
    """
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)#对输入的句子做初步的embedding
        self.encoder = BertEncoder(config)#bertEncoder,这里用的是8层的
        self.pooler = BertPooler(config)#Bert pool,得到的输出可以用来下游 NSP 任务,这次的掩码任务其实不需要这个

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            #参数没给attention_mask,全部设为1
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            #没给token_type_ids,每个字下标设为0

        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMaskedLM(nn.Module):

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__()
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.output = nn.Linear(256, 3)
        self.config=config
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        # if masked_lm_labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        #     return masked_lm_loss
        # else:
        return prediction_scores
