"""
modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/trainer.py
"""

import os
import torch
from torch import nn
import logging
from typing import Optional, Any, Union

# from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import (
    nested_detach,
    # smp_forward_only,
    # smp_nested_concat,
)
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderM3Trainer(AbsEmbedderTrainer):
    """
    Trainer class for M3.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save the model to directory.

        Args:
            output_dir (Optional[str], optional): Output directory to save the model. Defaults to ``None``.

        Raises:
            NotImplementedError
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, "save"):
            raise NotImplementedError(
                f"MODEL {self.model.__class__.__name__} "
                f"does not support save interface"
            )
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        # if self.is_world_process_zero():
        #     save_ckpt_for_sentence_transformers(output_dir,
        #                                         pooling_mode=self.args.sentence_pooling_method,
        #                                         normlized=self.args.normlized)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        Args:
            model (AbsEmbedderModel): The model being trained.
            inputs (dict): A dictionary of input tensors to be passed to the model.
            return_outputs (bool, optional): If ``True``, returns both the loss and the model's outputs. Otherwise,
                returns only the loss.

        Returns:
            Union[torch.Tensor, tuple(torch.Tensor, EmbedderOutput)]: The computed loss. If ``return_outputs`` is ``True``,
                also returns the model's outputs in a tuple ``(loss, outputs)``.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config,
                    "keys_to_ignore_at_inference",
                    ["past_key_values"],
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.detach().mean()

                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                    )
                else:
                    logits = outputs[1:]
            else:
                # loss = None
                loss = self.compute_loss(model, inputs)
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
