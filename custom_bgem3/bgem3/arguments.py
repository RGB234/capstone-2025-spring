"""
modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/arguments.py
"""

from dataclasses import dataclass, field
from typing import Union, Optional, List

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
    # Early stopping
    load_best_model_at_end: bool = field(
        default=True,
    )
    evaluation_strategy: str = field(
        default="epoch",
    )
    save_strategy: str = field(
        default="epoch"
    )
    metric_for_best_model: str = field(
        default="loss"
    )
    greater_is_better: bool = field(
        default=True,
    )


@dataclass
class EncoderOnlyEmbedderM3DataArguments(AbsEmbedderDataArguments):
    eval_data: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "eval_dataset argument of transformers.Trainer. One or more paths to validating data.",
            # "help": "eval_dataset argument of transformers.Trainer. One or more paths to training data. `query: str`, `pos: List[str]`, `neg: List[str]` are required in the training data.",
            "nargs": "+",
        },
    )
