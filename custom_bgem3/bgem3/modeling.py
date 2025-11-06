

import os
from typing import Dict, List, Union, Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from FlagEmbedding.finetune.embedder.encoder_only.m3 import EncoderOnlyEmbedderM3Model
from FlagEmbedding.abc.finetune.embedder import EmbedderOutput

class BGEM3(EncoderOnlyEmbedderM3Model):
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
        self_distill_start_step: int = -1
    ):
        super().__init__(
            base_model = base_model,
            tokenizer = tokenizer,
            negatives_cross_device = negatives_cross_device,
            temperature = temperature,
            sub_batch_size = sub_batch_size,
            kd_loss_type = kd_loss_type,
            sentence_pooling_method = sentence_pooling_method,
            normalize_embeddings = normalize_embeddings,
            unified_finetuning = unified_finetuning,
            use_self_distill = use_self_distill,
            self_distill_start_step = self_distill_start_step,
        )

        # a list of of tensor names to ignore when saving the model
        # (useful for keys that aren't trained, but which are deterministic)
        # reference : https://huggingface.co/transformers/v4.3.3/_modules/transformers/modeling_utils.html PreTrainedModel
        self._keys_to_ignore_on_save = None

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

        def _trans_state_dict_append_prefix(state_dict, prefix:str):
            # Add prefix "model" to state_dict keys for consistency between saving and loading.
            state_dict = type(state_dict)(
                {
                    (f"{prefix}.{k}" if not k.startswith(f"{prefix}.") else k): v.clone().cpu()
                    for k, v in state_dict.items()
                }
            )
            return state_dict

        self.model.save_pretrained(
            output_dir, state_dict=_trans_state_dict_append_prefix(self.model.state_dict(), "model")
        )

        if self.unified_finetuning:
            torch.save(
                _trans_state_dict_append_prefix(self.colbert_linear.state_dict(), "colbert_linear"),
                os.path.join(output_dir, "colbert_linear.pt"),
            )
            torch.save(
                _trans_state_dict_append_prefix(self.sparse_linear.state_dict(), "sparse_linear"),
                os.path.join(output_dir, "sparse_linear.pt"),
            )

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
        q_dense_vecs, q_sparse_vecs, q_colbert_vecs = self.encode(queries)  # (batch_size, dim)
        p_dense_vecs, p_sparse_vecs, p_colbert_vecs = self.encode(passages) # (batch_size * group_size, dim)

        # To include eval_loss in evaluation metrics
        if self.training or self.model_eval:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_dense_vecs.device)
                teacher_scores = teacher_scores.view(q_dense_vecs.size(0), -1).detach()   # (batch_size, group_size)
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # (batch_size, group_size)
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
                q_dense_vecs, p_dense_vecs, teacher_targets=teacher_targets,
                compute_score_func=self.compute_dense_score
            )

            if self.unified_finetuning:
                # disable cross device negatives for unified finetuning
                if no_in_batch_neg_flag:
                    compute_loss_func = self._compute_no_in_batch_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

                # sparse loss
                sparse_scores, sparse_loss = compute_loss_func(
                    q_sparse_vecs, p_sparse_vecs, teacher_targets=teacher_targets,
                    compute_score_func=self.compute_sparse_score
                )

                # colbert loss
                colbert_scores, colbert_loss = compute_loss_func(
                    q_colbert_vecs, p_colbert_vecs, teacher_targets=teacher_targets,
                    compute_score_func=self.compute_colbert_score,
                    q_mask=self._get_queries_attention_mask(queries)
                )

                # get dense scores of current process
                if not no_in_batch_neg_flag and self.negatives_cross_device:
                    dense_scores = dense_scores[
                        q_dense_vecs.size(0)*self.process_rank : q_dense_vecs.size(0)*(self.process_rank+1),
                        p_dense_vecs.size(0)*self.process_rank : p_dense_vecs.size(0)*(self.process_rank+1)
                    ]   # (batch_size, batch_size * group_size)
                elif no_in_batch_neg_flag:
                    # get local p_dense_vecs: fix a bug described in 
                    # https://github.com/FlagOpen/FlagEmbedding/issues/1410
                    group_size = p_dense_vecs.size(0) // q_dense_vecs.size(0)
                    indices = torch.arange(0, q_dense_vecs.size(0), device=q_dense_vecs.device) * group_size
                    p_dense_vecs = p_dense_vecs[indices, :]

                # ensemble loss
                ensemble_scores, ensemble_loss = compute_loss_func(
                    q_dense_vecs, p_dense_vecs, teacher_targets=teacher_targets,
                    compute_score_func=self.ensemble_score,
                    dense_scores=dense_scores,
                    sparse_scores=sparse_scores,
                    colbert_scores=colbert_scores
                )

                loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4

                if self.use_self_distill and self.step > self.self_distill_start_step:
                    self_teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)

                    dense_self_distill_loss = self.distill_loss("kl_div", self_teacher_targets, dense_scores)
                    sparse_self_distill_loss = self.distill_loss("kl_div", self_teacher_targets, sparse_scores)
                    colbert_self_distill_loss = self.distill_loss("kl_div", self_teacher_targets, colbert_scores)

                    loss += (dense_self_distill_loss + 0.1 * sparse_self_distill_loss + colbert_self_distill_loss) / 3
                    loss = loss / 2
            self.step += 1
        else:
            loss = None

        return EmbedderOutput(
            loss=loss,
        )