from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SentenceMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)
        self.out_dense = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # (batch_size, d_model) → (batch_size, num_heads, depth)
        x = x.view(-1, self.num_heads, self.depth)
        return x.transpose(0, 1)  # (num_heads, batch_size, depth)

    def scaled_dot_product(self, query, key, value):
        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (dk**0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def forward(self, query, key, value):
        # Input: (batch_size, d_model)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        scaled_attention, _ = self.scaled_dot_product(query, key, value)

        # (num_heads, batch_size, depth) → (batch_size, num_heads, depth)
        scaled_attention = scaled_attention.transpose(0, 1)
        concat_attention = scaled_attention.reshape(-1, self.d_model)

        output = self.out_dense(concat_attention)
        return output


class SentenceClusteringLayer(nn.Module):
    def __init__(self, dff, d_model, num_heads, dropout):
        super(SentenceClusteringLayer, self).__init__()
        self.attention = SentenceMultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch_size, d_model)
        attn_output = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        return out2


class CustomBGEM3FlagModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.d_model = 1024
        self.dff = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.hidden_dropout_prob

        self.bge_m3 = AutoModel.from_pretrained("BAAI/bge-m3")

        self.clustering_layer = SentenceClusteringLayer(
            dff=self.dff,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        output = self.bge_m3(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output.last_hidden_state  # [batch_size, seq_len, d_model]

        sentence_embeddings = sequence_output[:, 0, :]  # [CLS] token
        out = self.clustering_layer(sentence_embeddings)  # [batch_size, d_model]

        return out


## save ##

config = PretrainedConfig.from_pretrained("BAAI/bge-m3")  # 그대로 사용
model = CustomBGEM3FlagModel(config)

path = "../model"

model.save_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
tokenizer.save_pretrained(path)
