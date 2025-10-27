"""
modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/arguments.py
"""

from dataclasses import dataclass, field
from typing import Union

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderTrainingArguments,
    AbsEmbedderModelArguments,
    AbsEmbedderDataArguments,
)


@dataclass
class EncoderOnlyEmbedderM3ModelArguments(AbsEmbedderModelArguments):
    """
    Model argument class for M3.
    """

    colbert_dim: int = field(default=-1, metadata={"help": "Dim of colbert linear"})


@dataclass
class EncoderOnlyEmbedderM3TrainingArguments(AbsEmbedderTrainingArguments):
    """
    Training argument class for M3.
    """

    unified_finetuning: bool = field(
        default=False, metadata={"help": "use unify fine-tuning"}
    )
    use_self_distill: bool = field(
        default=False,
        metadata={"help": "use self-distill when using unify fine-tuning"},
    )
    fix_encoder: bool = field(
        default=False, metadata={"help": "Freeze the parameters of encoder"}
    )
    self_distill_start_step: int = field(
        default=-1, metadata={"help": "Num of step when using self-distill"}
    )

    # For EarlyStoppingCallback
    # save_strategy: str = field(
    #     default="steps",
    #     metadata={
    #         "help": "The checkpoint save strategy to adopt during training.",
    #     },
    #     kw_only=["no", "epoch", "steps", "best"],
    # )
    # eval_strategy: str = field(
    #     default="steps",
    #     metadata={"help": "The evaluation strategy to adopt during training."},
    #     kw_only=["No", "steps", "epoch"],
    # )
    # load_best_model_at_end: bool = field(
    #     default=True,
    #     metadata={
    #         "help": """Whether or not to load the best model found during training at the end of training. When this option is enabled, the best checkpoint will always be saved.
    #         When set to True, the parameters save_strategy needs to be the same as eval_strategy, and in the case it is “steps”, save_steps must be a round multiple of eval_steps."""
    #     },
    # )

    # eval_steps: int = field(
    #     default=100,
    #     metadata={
    #         "help": """Number of update steps between two evaluations if eval_strategy="steps". Will default to the same value as logging_steps if not set.
    #         Note that if the TrainingArguments argument save_steps differs from eval_steps, the early stopping will not occur until the next save step.
    #         """
    #     },
    # )
    # metric_for_best_model: str = field(
    #     default="eval_loss",
    #     metadata={
    #         "help": """Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models. Must be the name of a metric returned by the evaluation with or without the prefix \"eval_\".
    #         When set to None, `eval_loss` is used as the default metric."""
    #     },
    # )


@dataclass
class EncoderOnlyEmbedderM3DataArguments(AbsEmbedderDataArguments):
    eval_data: str = field(
        default=None,
        metadata={
            "help": "eval_dataset argument of transformers.Trainer. One or more paths to training data. `query: str`, `pos: List[str]`, `neg: List[str]` are required in the training data.",
            "nargs": "+",
        },
    )
