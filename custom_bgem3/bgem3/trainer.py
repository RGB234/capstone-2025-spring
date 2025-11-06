"""
modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/trainer.py
"""

import os
import torch
from torch import nn
import logging
from typing import Optional, Any, Union
from torch.utils.data import DataLoader, Dataset
from functools import partial
from collections.abc import Callable

# from transformers.utils import is_sagemaker_mp_enabled
from transformers.utils import (
    is_datasets_available,
)
from transformers.trainer_utils import(
    seed_worker,
)
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

if is_datasets_available():
    import datasets

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

    # FlagEmbedding.abc.finetune.embedder.AbsEmbedderTrainer
    def compute_loss(self, model, inputs, return_outputs=False):
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

        """
        inputs:
            queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
            passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
            teacher_scores: Union[None, List[float]] = None,
            no_in_batch_neg_flag: bool = False,
        """
        outputs = model(**inputs) # model.forward(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    # override
    # transformers.trainer
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
        model.model_eval = True
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        model.model_eval = False
        logits = loss # Make `compute_metrics` receive `eval_loss` instead of `eval_pred`, which is currently `None`.

        """
            def evaluation_loop(
                self,
                dataloader: DataLoader,
                description: str,
                prediction_loss_only: Optional[bool] = None,
                ignore_keys: Optional[list[str]] = None,
                metric_key_prefix: str = "eval",
            ) -> EvalLoopOutput:

            ...

            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            ...

            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)

            ...

            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        """
        return (loss, logits, torch.zeros(1)) # loss, logits, labels(dummy)
    
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        return self._get_dataloader(
            dataset=eval_dataset,
            description="Evaluation",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            dataloader_key=dataloader_key,
        )
    
    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

