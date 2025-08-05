import os
import logging
from typing import Dict, List, Union, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel, EmbedderOutput


# Custom Sub-layer
class SentenceMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Args :
          d_model : hidden_size
          num_heads : num_attention_heads
        """
        super().__init__()
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


# Custom Embedding layer
class SentenceAttentionLayer(nn.Module):
    """
    Args :
      dff : feedforward_size
      d_model : model hidden_size
      num_heads : number of attention heads
      dropout : dropout rate

    Returns:
      sentence_attention_embedding : (batch_size, d_model)
    """

    def __init__(self, dff, d_model, num_heads, dropout):
        super().__init__()
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


logger = logging.getLogger(__name__)


class BGEM3SAModel(AbsEmbedderModel):  # class AbsEmbedderModel(ABC, nn.Module):
    """Embedder class for M3 model.

    Args:
        base_model (dict[str, Any]): The base model to train on.
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        negatives_cross_device (bool, optional): If True, will compute cross devices negative loss. Defaults to ``False``.
        temperature (float, optional): Temperature to control the scale of scores. Defaults to ``1.0``.
        sub_batch_size (int, optional): Sub-batch size during encoding. If negative, will not split to sub-batch.
            Defaults to ``-1``.
        kd_loss_type (str, optional): Type of knowledge distillation loss. Defaults to ``'m3_kd_loss'``.
        sentence_pooling_method (str, optional): Pooling method to get sentence embedding. Defaults to ``'cls'``.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to ``False``.
        unified_finetuning (bool, optional): If True, will finetune colbert vector and sparce embedding. Defaults to ``True``.
        use_self_distill (bool, optional): If True, will do self distillation. Defaults to ``False``.
        self_distill_start_step (int, optional): Step num to start self distillation. Defaults to ``-1``.
    """

    def __init__(
        self,
        base_model: Dict[str, Any],
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1,
        sub_batch_size: int = -1,
        kd_loss_type: str = "m3_kd_loss",
        sentence_pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        unified_finetuning: bool = True,
        use_self_distill: bool = False,
        self_distill_start_step: int = -1,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

        self.unified_finetuning = unified_finetuning
        if not self.unified_finetuning:
            self.model = base_model["model"]
            self.colbert_linear = None
            self.sparse_linear = None
            #
            self.sentence_attention_layer = None
        else:
            self.model = base_model["model"]
            self.colbert_linear = base_model["colbert_linear"]
            self.sparse_linear = base_model["sparse_linear"]
            #
            self.sentence_attention_layer = base_model["sentence_attention_layer"]

        self.config = self.model.config

        self.vocab_size = self.model.config.vocab_size
        self.use_self_distill = use_self_distill
        self.self_distill_start_step = self_distill_start_step
        self.step = 0

    def _dense_embedding(self, last_hidden_state, attention_mask):
        """Use the pooling method to get the dense embedding.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state. (batch_size, sequence_length, hidden_size)
            attention_mask (torch.Tensor): Mask out padding tokens during pooling. (batch_size, seq_len) 0->padding, 1->valid token

        Raises:
            NotImplementedError: Specified pooling method not implemented.

        Returns:
            torch.Tensor: The dense embeddings.
        """
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )  # (batch_size, seq_len, hidden_size) -sum->  (batch_size, hidden_size)
            d = attention_mask.sum(
                dim=1, keepdim=True
            ).float()  # (batch_size, seq_len) -sum->  (batch_size)
            return s / d
        elif self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(
                f"pooling method {self.sentence_pooling_method} not implemented"
            )

    def _sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        """Compute and return the sparse embedding.

        Args:
            hidden_state (torch.Tensor): The model output's last hidden state.
            input_ids (_type_): Ids from input features. (batch_size, sequence_len)
            return_embedding (bool, optional): If True, return the computed embedding, otherwise just return the token weights.
                Defaults to ``True``.

        Returns:
            torch.Tensor: The sparse embedding or just the token weights.
        """
        token_weights = torch.relu(
            self.sparse_linear(hidden_state)
        )  # (batch_size, seq_len, hidden_size) @ (hidden_size, 1) -> (batch_size, seq_len, 1)
        if not return_embedding:
            return token_weights

        if self.training:
            sparse_embedding = torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                self.vocab_size,  # seq_len
                dtype=token_weights.dtype,
                device=token_weights.device,
            )
            sparse_embedding = torch.scatter(
                sparse_embedding,  # (batch_size, seq_len, seq_len)
                dim=-1,
                index=input_ids.unsqueeze(-1),  # (batch_size, seq_len, 1)
                src=token_weights,  # (batch_szie, seq_len, 1)
            )
            #  If a term t appears multiple times in the query, we only retain its max weight.
            sparse_embedding = torch.max(sparse_embedding, dim=1).values
        else:
            # Optimize suggestion from issue #1364: https://github.com/FlagOpen/FlagEmbedding/issues/1364
            # Disable when self.training = True, otherwise will cause:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # (batch_size, vocab_size)
            sparse_embedding = torch.zeros(
                input_ids.size(0),
                self.vocab_size,
                dtype=token_weights.dtype,
                device=token_weights.device,
            )

            sparse_embedding = sparse_embedding.scatter_reduce(
                dim=-1,
                index=input_ids,
                src=token_weights.squeeze(-1),
                reduce="amax",
            )

        unused_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]
        sparse_embedding[:, unused_tokens] *= 0.0
        return sparse_embedding

    def _colbert_embedding(self, last_hidden_state, mask):
        """Get the colbert vectors.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.

        Returns:
            torch.Tensor: The colbert vectors.
        """
        colbert_vecs = self.colbert_linear(
            last_hidden_state[:, 1:]
        )  # (batch_size, seq_len-1, hidden_size) @ (hidden_size, colbert_dim)
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    #
    def _sentence_attention_embedding(self, last_hidden_state, mask):
        dense_embedding = self._dense_embedding(
            last_hidden_state, mask
        )  # (batch_size, hidden_size)
        sentence_attention_embedding = self.sentence_attention_layer(dense_embedding)

        return sentence_attention_embedding

    def compute_score(
        self,
        q_reps,
        p_reps,
        q_mask: torch.Tensor,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.3,
        colbert_weight: float = 1.0,
        #
        # sentence_attention_weight: float = 0.3,
        sentence_attention_weight: float = 1.0,
    ):
        """_summary_

        Args:
            q_reps (_type_): Query representations.
            p_reps (_type_): Passage representations.
            q_mask (torch.Tensor): _description_
            dense_weight (float, optional): _description_. Defaults to 1.0.
            sparse_weight (float, optional): _description_. Defaults to 0.3.
            colbert_weight (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        dense_score = self.compute_dense_score(q_reps, p_reps)
        sparse_score = self.compute_sparse_score(q_reps, p_reps)
        colbert_score = self.compute_colbert_score(q_reps, p_reps, q_mask=q_mask)
        sentence_attention_score = self.compute_sentence_attention_score(q_reps, p_reps)
        return (
            dense_score * dense_weight
            + sparse_score * sparse_weight
            + colbert_score * colbert_weight
            #
            + sentence_attention_score * sentence_attention_weight
        )

    def compute_dense_score(self, q_reps, p_reps):
        """Compute the dense score.

        Args:
            q_reps (torch.Tensor): Query representations. (q, n)
            p_reps (torch.Tensor): Passage representations. (p, n)

        Returns:
            torch.Tensor: The computed dense scores, adjusted by temperature.
        """
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def compute_sparse_score(self, q_reps, p_reps):
        """Compute the sparse score.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed sparse scores, adjusted by temperature.
        """
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def compute_colbert_score(self, q_reps, p_reps, q_mask: torch.Tensor = None):
        """Compute the colbert score.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed colber scores, adjusted by temperature.
        """
        token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)
        scores, _ = token_scores.max(-1)  # qip
        scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
        scores = scores / self.temperature
        return scores

    #
    def compute_sentence_attention_score(self, q_reps, p_reps):
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def ensemble_score(
        self,
        q_reps,
        p_reps,
        dense_scores=None,
        sparse_scores=None,
        colbert_scores=None,
        sentence_attention_scores=None,
    ):
        """Compute the ensemble score of the three methods.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.
            dense_scores (torch.Tensor, optional): The dense scores. Defaults to ``None``.
            sparse_scores (torch.Tensor, optional): The sparse scores. Defaults to ``None``.
            colbert_scores (torch.Tensor, optional): The colbert scores. Defaults to ``None``.

        Raises:
            ValueError: dense_scores, sparse_scores, colbert_scores must be provided

        Returns:
            _type_: The ensemble score of the three methods.
        """
        if (
            dense_scores is None
            or sparse_scores is None
            or colbert_scores is None
            or sentence_attention_scores is None
        ):
            raise ValueError(
                "dense_scores, sparse_scores, colbert_scores, sentence_attention_scores must be provided!"
            )
        return (
            dense_scores
            + 0.3 * sparse_scores
            + colbert_scores
            #
            # + 0.3 * sentence_attention_scores
            + sentence_attention_scores
        )

    def _encode(self, features):
        """Helper function to encode using input features.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: Dense embedding.
            torch.Tensor: Sparce embedding.
            torch.Tensor: Colbert vector.
            +
            torch.Tensor: Sentence attention embedding
        """
        dense_vecs, sparse_vecs, colbert_vecs, sentence_attention_vecs = (
            None,
            None,
            None,
            None,
        )
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
        dense_vecs = self._dense_embedding(
            last_hidden_state, features["attention_mask"]
        )
        if self.unified_finetuning:
            sparse_vecs = self._sparse_embedding(
                last_hidden_state, features["input_ids"]
            )
            colbert_vecs = self._colbert_embedding(
                last_hidden_state, features["attention_mask"]
            )
            #
            sentence_attention_vecs = self._sentence_attention_embedding(
                last_hidden_state, features["attention_mask"]
            )
        if self.normalize_embeddings:
            dense_vecs = F.normalize(dense_vecs, dim=-1)
            if self.unified_finetuning:
                colbert_vecs = F.normalize(colbert_vecs, dim=-1)
                #
                sentence_attention_vecs = F.normalize(sentence_attention_vecs, dim=-1)
        return dense_vecs, sparse_vecs, colbert_vecs, sentence_attention_vecs

    def encode(self, features):
        """Encode and get the embedding.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: Dense embeddings.
            torch.Tensor: Sparce embeddings.
            torch.Tensor: Colbert vectors.
            +
            torch.Tensor: Sentence attention embedding
        """
        if features is None:
            return None

        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size != -1:
                (
                    all_dense_vecs,
                    all_sparse_vecs,
                    all_colbert_vecs,
                    all_sentence_attention_vecs,
                ) = ([], [], [], [])
                for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
                    end_inx = min(
                        i + self.sub_batch_size, len(features["attention_mask"])
                    )
                    sub_features = {}
                    for k, v in features.items():
                        sub_features[k] = v[i:end_inx]

                    dense_vecs, sparse_vecs, colbert_vecs, sentence_attention_vecs = (
                        self._encode(sub_features)
                    )
                    all_dense_vecs.append(dense_vecs)
                    all_sparse_vecs.append(sparse_vecs)
                    all_colbert_vecs.append(colbert_vecs)
                    #
                    all_sentence_attention_vecs.append(sentence_attention_vecs)

                dense_vecs = torch.cat(all_dense_vecs, 0)
                if self.unified_finetuning:
                    sparse_vecs = torch.cat(all_sparse_vecs, 0)
                    colbert_vecs = torch.cat(all_colbert_vecs, 0)
                    #
                    sentence_attention_vecs = torch.cat(all_sentence_attention_vecs, 0)
            else:
                dense_vecs, sparse_vecs, colbert_vecs, sentence_attention_vecs = (
                    self._encode(features)
                )
        else:
            (
                all_dense_vecs,
                all_sparse_vecs,
                all_colbert_vecs,
                all_sentence_attention_vecs,
            ) = ([], [], [], [])
            for sub_features in features:
                dense_vecs, sparse_vecs, colbert_vecs, sentence_attention_vecs = (
                    self._encode(sub_features)
                )
                all_dense_vecs.append(dense_vecs)
                all_sparse_vecs.append(sparse_vecs)
                all_colbert_vecs.append(colbert_vecs)
                #
                all_sentence_attention_vecs.append(sentence_attention_vecs)

            dense_vecs = torch.cat(all_dense_vecs, 0)
            if self.unified_finetuning:
                sparse_vecs = torch.cat(all_sparse_vecs, 0)
                colbert_vecs = torch.cat(all_colbert_vecs, 0)
                sentence_attention_vecs = torch.cat(all_sentence_attention_vecs, 0)

        if self.unified_finetuning:
            return (
                dense_vecs.contiguous(),
                sparse_vecs.contiguous(),
                colbert_vecs.contiguous(),
                #
                sentence_attention_vecs.contiguous(),
            )
        else:
            return dense_vecs.contiguous(), None, None, None

    def _compute_similarity(self, q_reps, p_reps):
        """Computes the similarity between query and passage representations using inner product.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed similarity matrix.
        """
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def _get_queries_attention_mask(
        self, queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
    ):
        """padding attention mask for colbert

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]]): Input queries.

        Returns:
            torch.Tensor: The query attention mask.
        """
        if not isinstance(queries, list):
            q_mask = queries["attention_mask"]
        else:
            q_mask_list = [sub_features["attention_mask"] for sub_features in queries]
            _length = max([mask.shape[1] for mask in q_mask_list])
            if self.tokenizer.padding_side == "right":
                q_mask = torch.cat(
                    [
                        F.pad(mask, (0, _length - mask.shape[1]), value=0)
                        for mask in q_mask_list
                    ],
                    dim=0,
                )
            else:
                q_mask = torch.cat(
                    [
                        F.pad(mask, (_length - mask.shape[1], 0), value=0)
                        for mask in q_mask_list
                    ],
                    dim=0,
                )
        return q_mask

    def forward(
        self,
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        """The computation performed at every call.

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input queries. Defaults to ``None``.
            passages (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input passages. Defaults to ``None``.
            teacher_scores (Union[None, List[float]], optional): Teacher scores for distillation. Defaults to ``None``.
            no_in_batch_neg_flag (bool, optional): If True, use no in-batch negatives and no cross-device negatives. Defaults to ``False``.

        Returns:
            EmbedderOutput: Output of the forward call of model.
        """
        q_dense_vecs, q_sparse_vecs, q_colbert_vecs, q_sentence_attention_vecs = (
            self.encode(queries)
        )  # (batch_size, dim)
        p_dense_vecs, p_sparse_vecs, p_colbert_vecs, p_sentence_attention_vecs = (
            self.encode(passages)
        )  # (batch_size * group_size, dim)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(
                    teacher_scores, device=q_dense_vecs.device
                )
                teacher_scores = teacher_scores.view(
                    q_dense_vecs.size(0), -1
                ).detach()  # (batch_size, group_size)
                teacher_targets = F.softmax(
                    teacher_scores, dim=-1
                )  # (batch_size, group_size)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            else:
                if self.negatives_cross_device:
                    compute_loss_func = self._compute_cross_device_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

            # dense loss
            dense_scores, loss = compute_loss_func(
                q_dense_vecs,
                p_dense_vecs,
                teacher_targets=teacher_targets,
                compute_score_func=self.compute_dense_score,
            )

            if self.unified_finetuning:
                # disable cross device negatives for unified finetuning
                if no_in_batch_neg_flag:
                    compute_loss_func = self._compute_no_in_batch_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

                # sparse loss
                sparse_scores, sparse_loss = compute_loss_func(
                    q_sparse_vecs,
                    p_sparse_vecs,
                    teacher_targets=teacher_targets,
                    compute_score_func=self.compute_sparse_score,
                )

                # colbert loss
                colbert_scores, colbert_loss = compute_loss_func(
                    q_colbert_vecs,
                    p_colbert_vecs,
                    teacher_targets=teacher_targets,
                    compute_score_func=self.compute_colbert_score,
                    q_mask=self._get_queries_attention_mask(queries),
                )

                #
                # sentence attention loss
                sentence_attention_scores, sentence_attention_loss = compute_loss_func(
                    q_sentence_attention_vecs,
                    p_sentence_attention_vecs,
                    teacher_targets=teacher_targets,
                    compute_score_func=self.compute_sentence_attention_score,
                )

                # get dense scores of current process
                if not no_in_batch_neg_flag and self.negatives_cross_device:
                    dense_scores = dense_scores[
                        q_dense_vecs.size(0)
                        * self.process_rank : q_dense_vecs.size(0)
                        * (self.process_rank + 1),
                        p_dense_vecs.size(0)
                        * self.process_rank : p_dense_vecs.size(0)
                        * (self.process_rank + 1),
                    ]  # (batch_size, batch_size * group_size)
                elif no_in_batch_neg_flag:
                    # get local p_dense_vecs: fix a bug described in
                    # https://github.com/FlagOpen/FlagEmbedding/issues/1410
                    group_size = p_dense_vecs.size(0) // q_dense_vecs.size(0)
                    indices = (
                        torch.arange(
                            0, q_dense_vecs.size(0), device=q_dense_vecs.device
                        )
                        * group_size
                    )
                    p_dense_vecs = p_dense_vecs[indices, :]

                # ensemble loss
                ensemble_scores, ensemble_loss = compute_loss_func(
                    q_dense_vecs,
                    p_dense_vecs,
                    teacher_targets=teacher_targets,
                    compute_score_func=self.ensemble_score,
                    dense_scores=dense_scores,
                    sparse_scores=sparse_scores,
                    colbert_scores=colbert_scores,
                    #
                    sentence_attention_scores=sentence_attention_scores,
                )

                # loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4
                loss = (
                    loss
                    + ensemble_loss
                    + 0.1 * sparse_loss
                    + colbert_loss
                    # + 0.1 * sentence_attention_loss
                    + sentence_attention_loss
                ) / 5

                if self.use_self_distill and self.step > self.self_distill_start_step:
                    self_teacher_targets = torch.softmax(
                        ensemble_scores.detach(), dim=-1
                    )

                    dense_self_distill_loss = self.distill_loss(
                        "kl_div", self_teacher_targets, dense_scores
                    )
                    sparse_self_distill_loss = self.distill_loss(
                        "kl_div", self_teacher_targets, sparse_scores
                    )
                    colbert_self_distill_loss = self.distill_loss(
                        "kl_div", self_teacher_targets, colbert_scores
                    )
                    #
                    sentence_self_distill_loss = self.distill_loss(
                        "kl_div", self_teacher_targets, sentence_attention_scores
                    )

                    # loss += (
                    #     dense_self_distill_loss
                    #     + 0.1 * sparse_self_distill_loss
                    #     + colbert_self_distill_loss
                    # ) / 3
                    loss += (
                        dense_self_distill_loss
                        + 0.1 * sparse_self_distill_loss
                        + colbert_self_distill_loss
                        #
                        # + 0.1 * sentence_self_distill_loss
                        + sentence_self_distill_loss
                    ) / 4

                    loss = loss / 2
            self.step += 1
        else:
            loss = None

        return EmbedderOutput(
            loss=loss,
        )

    def compute_loss(self, scores, target):
        """Compute the loss using cross entropy.

        Args:
            scores (torch.Tensor): Computed score.
            target (torch.Tensor): The target value.

        Returns:
            torch.Tensor: The computed cross entropy loss.
        """
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model.
        """
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        """
        Enables the gradients for the input embeddings.
        """
        self.model.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        """Save the model to the directory.

        Args:
            output_dir (str): Directory for saving the model.
        """

        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu() for k, v in state_dict.items()}
            )
            return state_dict

        self.model.save_pretrained(
            output_dir, state_dict=_trans_state_dict(self.model.state_dict())
        )

        if self.unified_finetuning:
            torch.save(
                _trans_state_dict(self.colbert_linear.state_dict()),
                os.path.join(output_dir, "colbert_linear.pt"),
            )
            torch.save(
                _trans_state_dict(self.sparse_linear.state_dict()),
                os.path.join(output_dir, "sparse_linear.pt"),
            )
            #
            torch.save(
                _trans_state_dict(self.sentence_attention_layer.state_dict()),
                os.path.join(output_dir, "sentence_attention_layer.pt"),
            )


class BGEM3SAModelForInference(BGEM3SAModel):
    """
    Inference class of M3 model.
    """

    def forward(
        self,
        text_input: Dict[str, Tensor] = None,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        return_sparse_embedding: bool = False,
        #
        return_sentence_attention_vecs: bool = False,
    ):
        """Encode the text input using the selected way.

        Args:
            text_input (Dict[str, Tensor], optional): Text inputs. Defaults to ``None``.
            return_dense (bool, optional): If True, return the dense embedding. Defaults to ``True``.
            return_sparse (bool, optional): If True, return the sparse embedding. Defaults to ``False``.
            return_colbert_vecs (bool, optional): If True, return the colbert vectors. Defaults to ``False``.
            return_sparse_embedding (bool, optional): Parameter for :meth:`_sparse_embedding()`. If True, will return sparse embedding.
                Otherwise, return the token weights. Defaults to ``False``.

        Returns:
            dict: A dictionary containing the three types of embeddings.
        """
        assert (
            return_dense
            or return_sparse
            or return_colbert_vecs
            or return_sentence_attention_vecs
        ), "Must choose one or more from `return_colbert_vecs`, `return_sparse`, `return_dense`, `return_sentence_attention_embedding` to set `True`!"

        # this is for sparse embedding computation: using optimization suggestion from
        # issue #1364: https://github.com/FlagOpen/FlagEmbedding/issues/1364
        self.training = False

        last_hidden_state = self.model(**text_input, return_dict=True).last_hidden_state

        output = {}
        if return_dense:
            dense_vecs = self._dense_embedding(
                last_hidden_state, text_input["attention_mask"]
            )
            output["dense_vecs"] = dense_vecs
        if return_sparse:
            sparse_vecs = self._sparse_embedding(
                last_hidden_state,
                text_input["input_ids"],
                return_embedding=return_sparse_embedding,
            )
            output["sparse_vecs"] = sparse_vecs
        if return_colbert_vecs:
            colbert_vecs = self._colbert_embedding(
                last_hidden_state, text_input["attention_mask"]
            )
            output["colbert_vecs"] = colbert_vecs
        #
        if return_sentence_attention_vecs:
            sentence_attention_vecs = self._sentence_attention_embedding(
                last_hidden_state, text_input["attention_mask"]
            )
            output["sentence_attention_vecs"] = sentence_attention_vecs

        if self.normalize_embeddings:
            if "dense_vecs" in output:
                output["dense_vecs"] = F.normalize(output["dense_vecs"], dim=-1)
            if "colbert_vecs" in output:
                output["colbert_vecs"] = F.normalize(output["colbert_vecs"], dim=-1)
            #
            if "sentence_attention_vecs" in output:
                output["sentence_attention_vecs"] = F.normalize(
                    output["sentence_attention_vecs"], dim=-1
                )

        return output
