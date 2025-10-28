"""
modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/runner.py
"""

import os
from pathlib import Path

import torch
import logging
from typing import Tuple
from transformers import (
    set_seed,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from huggingface_hub import snapshot_download

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderRunner,
    AbsEmbedderModel,
    AbsEmbedderDataArguments,
    EmbedderTrainerCallbackForDataRefresh,
)

from dataclasses import asdict, fields

# from .modeling import EncoderOnlyEmbedderM3Model
from .modeling import (
    BGEM3SaqModel,
    SentenceAttentionModule,
)
from .trainer import EncoderOnlyEmbedderM3Trainer
from .dataset import EmbedderEvalDataset
from .arguments import (
    EncoderOnlyEmbedderM3ModelArguments,
    EncoderOnlyEmbedderM3TrainingArguments,
    #
    EncoderOnlyEmbedderM3DataArguments,
)


###
from transformers import TrainerCallback, EarlyStoppingCallback, EvalPrediction

logger = logging.getLogger(__name__)


#
class WeightChangeTrackerCallback(TrainerCallback):
    def __init__(self, layer_name="sentence_attention_module.layers.5.norm2.weight"):
        self.layer_name = layer_name
        self.prev_weights = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        with torch.no_grad():
            weight = dict(model.named_parameters())[self.layer_name].clone().cpu()
            if self.prev_weights is not None:
                delta = torch.norm(weight - self.prev_weights).item()
                print(
                    f"[Step {state.global_step}] {self.layer_name} changed by {delta:.6f}"
                )

            self.prev_weights = weight


class EncoderOnlyEmbedderM3Runner(AbsEmbedderRunner):
    """
    M3 model runner for finetuning.

    Args:
        model_args (EncoderOnlyEmbedderM3ModelArguments): Model arguments
        data_args (AbsEmbedderDataArguments): Data arguments.
        training_args (EncoderOnlyEmbedderM3TrainingArguments): Training arguments.
    """

    def __init__(
        self,
        model_args: EncoderOnlyEmbedderM3ModelArguments,
        data_args: EncoderOnlyEmbedderM3DataArguments,
        training_args: EncoderOnlyEmbedderM3TrainingArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

        # Set seed
        set_seed(training_args.seed)

        self.tokenizer, self.model = self.load_tokenizer_and_model()
        self.train_dataset = self.load_train_dataset()
        self.data_collator = self.load_data_collator()
        self.trainer = self.load_trainer()

        self.model_args: EncoderOnlyEmbedderM3ModelArguments
        # self.data_args: AbsEmbedderDataArguments
        self.data_args: EncoderOnlyEmbedderM3DataArguments
        self.training_args: EncoderOnlyEmbedderM3TrainingArguments

    @staticmethod
    def get_model(
        model_name_or_path: str,
        trust_remote_code: bool = False,
        colbert_dim: int = -1,
        cache_dir: str = None,
    ):
        """Get the model.

        Args:
            model_name_or_path (str):  If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
                load a model from HuggingFace Hub with the name.
            trust_remote_code (bool, optional): trust_remote_code to use when loading models from HF. Defaults to ``False``.
            colbert_dim (int, optional): Colbert dim to set. Defaults to ``-1``.
            cache_dir (str, optional): HF cache dir to store the model. Defaults to ``None``.

        Returns:
            dict: A dictionary containing the model, colbert linear and sparse linear.
        """
        cache_folder = (
            os.getenv("HF_HUB_CACHE", None) if cache_dir is None else cache_dir
        )
        if not os.path.exists(model_name_or_path):
            model_name_or_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_folder,
                ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
            )

        model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
        )
        colbert_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=model.config.hidden_size if colbert_dim <= 0 else colbert_dim,
        )
        sparse_linear = torch.nn.Linear(
            in_features=model.config.hidden_size, out_features=1
        )
        #
        sentence_attention_module = SentenceAttentionModule(
            dff=model.config.intermediate_size,
            d_model=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
            dropout=model.config.hidden_dropout_prob,
        )

        colbert_model_path = os.path.join(model_name_or_path, "colbert_linear.pt")
        sparse_model_path = os.path.join(model_name_or_path, "sparse_linear.pt")
        colbert_linear: torch.nn.Linear
        #
        sentence_attention_model_path = os.path.join(
            model_name_or_path, "sentence_attention_module.pt"
        )
        if os.path.exists(colbert_model_path) and os.path.exists(sparse_model_path):
            logger.info(
                "loading existing colbert_linear and sparse_linear---------!!!!!!!!!"
            )
            colbert_state_dict = torch.load(
                colbert_model_path, map_location="cpu", weights_only=True
            )
            sparse_state_dict = torch.load(
                sparse_model_path, map_location="cpu", weights_only=True
            )
            colbert_linear.load_state_dict(colbert_state_dict)
            sparse_linear.load_state_dict(sparse_state_dict)
        else:
            logger.info(
                "The parameters of colbert_linear and sparse linear is new initialize. Make sure the model is loaded for training, not inferencing"
            )

        #
        if os.path.exists(sentence_attention_model_path):
            logger.info("loading existing sentence_attention_module---------")
            sentence_attention_state_dict = torch.load(
                sentence_attention_model_path, map_location="cpu", weights_only=True
            )
            sentence_attention_module.load_state_dict(sentence_attention_state_dict)
        else:
            logger.info(
                "The parameters of sentence_attention_module is new initialize. Make sure the model is loaded for training, not inferencing"
            )

        return {
            "model": model,
            "colbert_linear": colbert_linear,
            "sparse_linear": sparse_linear,
            #
            "sentence_attention_module": sentence_attention_module,
        }

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        num_labels = 1
        config = AutoConfig.from_pretrained(
            (
                self.model_args.config_name
                if self.model_args.config_name
                else self.model_args.model_name_or_path
            ),
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info("Config: %s", config)

        # model = EncoderOnlyEmbedderM3Model(
        model = BGEM3SaqModel(
            self.get_model(
                self.model_args.model_name_or_path,
                self.model_args.trust_remote_code,
                self.model_args.colbert_dim,
            ),
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            unified_finetuning=self.training_args.unified_finetuning,
            use_self_distill=self.training_args.use_self_distill,
            self_distill_start_step=self.training_args.self_distill_start_step,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False

        if self.training_args.fix_encoder:
            for k, v in model.named_parameters():
                # if "colbert_linear" in k or 1"sparse_linear" in k :
                if (
                    "colbert_linear" in k
                    or "sparse_linear" in k
                    or "sentence_attention_module" in k
                ):
                    logging.info(f"train the parameters for {k}")
                    v.requires_grad = True
                else:
                    v.requires_grad = False

        return tokenizer, model

    def load_trainer(self) -> EncoderOnlyEmbedderM3Trainer:
        """Load the M3 trainer.

        Returns:
            EncoderOnlyEmbedderM3Trainer: M3 Trainer instance.
        """
        trainer = EncoderOnlyEmbedderM3Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(
                EmbedderTrainerCallbackForDataRefresh(self.train_dataset)
            )
        #
        # trainer.add_callback(WeightChangeTrackerCallback())
        return trainer

    def run(self):
        """
        Executes the training process.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # logger.info("##### Model stete dict  #####\n")
        # logger.info(self.model.state_dict)

        # Training
        self.trainer.train(
            resume_from_checkpoint=self.training_args.resume_from_checkpoint
        )

        # logger.info("##### Trainer.stete.best_model_checkpoint #####")
        # logger.info(self.trainer.state.best_model_checkpoint)

        # logger.info("##### Trainer.stete #####")
        # logger.info(self.trainer.state)

        self.trainer.save_model()
