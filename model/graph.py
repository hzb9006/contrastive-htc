import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os
import numpy as np

from torch_geometric.nn import GCNConv, GATConv

# GRAPH = 'GCN'
GRAPH = "GRAPHORMER"


# GRAPH = 'GAT'


class SelfAttention(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.self = BartAttention(config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states,
                attention_mask=None, output_attentions=False, extra_attn=None):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions,
            extra_attn=extra_attn,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights += extra_attn

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GraphLayer(nn.Module):
    def __init__(self, config, last=False):
        super(GraphLayer, self).__init__()
        self.config = config

        class _Actfn(nn.Module):
            def __init__(self):
                super(_Actfn, self).__init__()
                if isinstance(config.hidden_act, str):
                    self.intermediate_act_fn = ACT2FN[config.hidden_act]
                else:
                    self.intermediate_act_fn = config.hidden_act

            def forward(self, x):
                return self.intermediate_act_fn(x)

        if GRAPH == 'GRAPHORMER':
            self.hir_attn = SelfAttention(config) # 使用了多头注意力
        elif GRAPH == 'GCN':
            self.hir_attn = GCNConv(config.hidden_size, config.hidden_size)
        elif GRAPH == 'GAT':
            self.hir_attn = GATConv(config.hidden_size, config.hidden_size, 1)

        self.last = last
        if last:
            self.cross_attn = BartAttention(config.hidden_size, 8, 0.1, True)
            self.cross_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.output_layer = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),
                                          _Actfn(),
                                          nn.Linear(config.intermediate_size, config.hidden_size),
                                          )
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, label_emb, extra_attn, self_attn_mask, inputs_embeds, cross_attn_mask):
        if GRAPH == 'GRAPHORMER':
            label_emb = self.hir_attn(label_emb,
                                      attention_mask=self_attn_mask, extra_attn=extra_attn)[0]
            # label_emb = self.output_layer_norm(self.dropout(self.output_layer(label_emb)) + label_emb)
        elif GRAPH == 'GCN' or GRAPH == 'GAT':
            label_emb = self.hir_attn(label_emb.squeeze(0), edge_index=extra_attn)
        if self.last:
            label_emb = label_emb.expand(inputs_embeds.size(0), -1, -1)
            label_emb = self.cross_attn(inputs_embeds, label_emb,
                                        attention_mask=cross_attn_mask.unsqueeze(1), output_attentions=True,
                                        only_attn=True)
            return label_emb

        label_emb = self.output_layer_norm(self.dropout(self.output_layer(label_emb)) + label_emb)
        if self.last:
            label_emb = self.dropout(self.classifier(label_emb))
        return label_emb


class GraphEncoder(nn.Module):
    def __init__(self, config, graph=False, layer=1, data_path=None, threshold=0.01, tau=1):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.tau = tau
        self.label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
        self.label_name = []
        for i in range(len(self.label_dict)):
            self.label_name.append(self.label_dict[i])
        self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
        self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        self.hir_layers = nn.ModuleList([GraphLayer(config, last=i == layer - 1) for i in range(layer)])

        self.label_num = len(self.label_name)

        self.graph = graph
        self.threshold = threshold

        if graph:
            label_hier = torch.load(os.path.join(data_path, 'slot.pt'))
            path_dict = {} # 键值对，键是子节点，值是父节点，代表子节点属于哪个父节点
            num_class = 0 # 获取类别数
            for s in label_hier: # 获取根节点
                for v in label_hier[s]: # 获取子节点
                    path_dict[v] = s
                    if num_class < v:
                        num_class = v
            if GRAPH == 'GRAPHORMER':
                num_class += 1
                for i in range(num_class):# path_dict增加，父节点的父节点是他自己
                    if i not in path_dict:
                        path_dict[i] = i
                self.inverse_label_list = {}

                def get_root(path_dict, n):# 获取根节点
                    ret = []
                    while path_dict[n] != n:# 不是根节点
                        ret.append(n)
                        n = path_dict[n] # 把n的父节点赋值给n，这样下次循环时会跳出循环
                    ret.append(n)
                    return ret

                for i in range(num_class):
                    self.inverse_label_list.update({i: get_root(path_dict, i) + [-1]}) # 把[父节点，子节点] 颠倒成[子节点，父节点]，根节点的父节点为-1，如12:[12,0,-1]
                label_range = torch.arange(len(self.inverse_label_list))
                self.label_id = label_range
                node_list = {}

                def get_distance(node1, node2): # node1从distance_mat中按行取值，node2从hier_mat_t中按行取值
                    p = 0
                    q = 0
                    node_list[(node1, node2)] = a = []
                    node1 = self.inverse_label_list[node1]
                    node2 = self.inverse_label_list[node2] # 获取节点及其父节点
                    while p < len(node1) and q < len(node2):
                        if node1[p] > node2[q]:
                            a.append(node1[p])
                            p += 1

                        elif node1[p] < node2[q]:
                            a.append(node2[q])
                            q += 1

                        else:
                            break
                    return p + q # 1. 节点到他本身的距离为0 ； 2.

                self.distance_mat = self.label_id.reshape(1, -1).repeat(self.label_id.size(0), 1) # (141,)--->(1,141)-->(141,141)
                hier_mat_t = self.label_id.reshape(-1, 1).repeat(1, self.label_id.size(0)) # 第一行为0，第二行为1，一共141行
                self.distance_mat.map_(hier_mat_t, get_distance) # map_(hier_mat_t, get_distance)表示对hier_mat_t中的每个元素都应用get_distance函数,其中每个元素按行读取,节点之间的距离是两个节点到他们最近的共同节点的距离
                self.distance_mat = self.distance_mat.view(1, -1) # (141,141)-->(1,19881),每一行都进行拼接，表示标签到标签之间的距离，0-0，1-0，2-0，3-0
                self.edge_mat = torch.zeros(len(self.inverse_label_list), len(self.inverse_label_list), 15,
                                            dtype=torch.long) #(141,141,15)
                for i in range(len(self.inverse_label_list)):
                    for j in range(len(self.inverse_label_list)):
                        edge_list = node_list[(i, j)] # 获取节点到节点之间的通路
                        self.edge_mat[i, j, :len(edge_list)] = torch.tensor(edge_list) + 1 # 因为root节点为-1，所以+1，经过root节点，就是经过0节点
                self.edge_mat = self.edge_mat.view(-1, self.edge_mat.size(-1)) # 保存的是节点到节点之间边的信息，经过的值要-1，如0-0，1-0，2-0，3-0的边

                self.id_embedding = nn.Embedding(len(self.inverse_label_list) + 1, config.hidden_size,
                                                 len(self.inverse_label_list)) #id_embedding:（142，768，padding_idx=141）
                self.distance_embedding = nn.Embedding(20, 1, 0) #(20,1,padding_idx=0)
                self.edge_embedding = nn.Embedding(len(self.inverse_label_list) + 1, 1, 0) #（142,1,padding_idx=0)
                self.label_id = nn.Parameter(self.label_id, requires_grad=False)
                self.edge_mat = nn.Parameter(self.edge_mat, requires_grad=False)
                self.distance_mat = nn.Parameter(self.distance_mat, requires_grad=False)
            self.edge_list = [[v, i] for v, i in path_dict.items()] #[子节点，父节点] 141
            self.edge_list += [[i, v] for v, i in path_dict.items()] # [子节点，父节点] 141+ [父节点，子节点] 141
            self.edge_list = nn.Parameter(torch.tensor(self.edge_list).transpose(0, 1), requires_grad=False)

    def forward(self, inputs_embeds, attention_mask, labels, embeddings):
        label_mask = self.label_name != self.tokenizer.pad_token_id
        # full name，label_emb对应论文里面的name_emb,把每个标签的词使用bert进行emb后相加求平均作为该标签的emb
        label_emb = embeddings(self.label_name)
        label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1) # label_emb * label_mask.unsqueeze(-1)-->(141,10,768)把填充的emb置为0，对于3维的，dim=1是按照x轴进行累加，对于二维的，dim=1是y轴，所以 label_mask.sum(dim=1)可以获取每个标签的有效长度,此处把每个标签对应的有效的词的embedding做了累加求平均
        label_emb = label_emb.unsqueeze(0) # （141，768）-->(1,141,768)

        label_attn_mask = torch.ones(1, label_emb.size(1), device=label_emb.device) # label_attn_mask-->(1,141)

        extra_attn = None
        # cross_attn_mask是（3，512，141），矩阵中如果是pad，则为0，如果是真实的输入，是1 todo：表示交叉attention？
        self_attn_mask = (label_attn_mask * 1.).t().mm(label_attn_mask * 1.).unsqueeze(0).unsqueeze(0) # (label_attn_mask * 1.).t()-->(141,1)*(1,141)=(141,141)-->(1,1,141,141)
        cross_attn_mask = (attention_mask * 1.).unsqueeze(-1).bmm(
            (label_attn_mask.unsqueeze(0) * 1.).repeat(attention_mask.size(0), 1, 1)) #  (attention_mask * 1.) 则true为1，false为0，(attention_mask * 1.).unsqueeze(-1)：(3,512)-->(3,512,1)  ,(label_attn_mask.unsqueeze(0) * 1.).repeat(attention_mask.size(0), 1, 1) -->(3,1,141),其中每个batch是一样的
        expand_size = label_emb.size(-2) // self.label_name.size(0)
        if self.graph:
            if GRAPH == 'GRAPHORMER': # 加入id_emb
                label_emb += self.id_embedding(self.label_id[:, None].expand(-1, expand_size)).view(1, -1,
                                                                                                    self.config.hidden_size) # 这是论文里的label_embedding,把141个标签随机初始化
                extra_attn = self.distance_embedding(self.distance_mat) + self.edge_embedding(self.edge_mat).sum(
                    dim=1) / (
                                     self.distance_mat.view(-1, 1) + 1e-8) # distance_embedding是随机初始化的（20，1）
                extra_attn = extra_attn.view(self.label_num, 1, self.label_num, 1).expand(-1, expand_size, -1,
                                                                                          expand_size)
                extra_attn = extra_attn.reshape(self.label_num * expand_size, -1)
            elif GRAPH == 'GCN' or GRAPH == 'GAT':
                extra_attn = self.edge_list
        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb, extra_attn, self_attn_mask, inputs_embeds, cross_attn_mask)

        token_probs = label_emb.mean(dim=1).view(attention_mask.size(0), attention_mask.size(1),
                                                 self.label_name.size(0),
                                                 )

        # sum
        contrast_mask = (F.gumbel_softmax(token_probs, hard=False, dim=-1, tau=self.tau) * labels.unsqueeze(1)).sum(
            -1)

        temp = self.threshold
        _mask = contrast_mask > temp
        contrast_mask = contrast_mask + (1 - contrast_mask).detach()
        contrast_mask = contrast_mask * _mask

        return contrast_mask
