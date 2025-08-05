"""
modified from
https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/__init__.py
"""

# from FlagEmbedding.abc.finetune.embedder import (
#     AbsEmbedderDataArguments as EncoderOnlyEmbedderM3DataArguments,
# )

from .arguments import (
    EncoderOnlyEmbedderM3ModelArguments,
    EncoderOnlyEmbedderM3TrainingArguments,
    #
    EncoderOnlyEmbedderM3DataArguments,
)

# from .modeling import EncoderOnlyEmbedderM3Model, EncoderOnlyEmbedderM3ModelForInference
from .modeling import BGEM3SaeModel, BGEM3SaeModelForInference
from .trainer import EncoderOnlyEmbedderM3Trainer
from .runner import EncoderOnlyEmbedderM3Runner


__all__ = [
    "EncoderOnlyEmbedderM3ModelArguments",
    "EncoderOnlyEmbedderM3DataArguments",
    "EncoderOnlyEmbedderM3TrainingArguments",
    "BGEM3SaeModel",
    "BGEM3SaeModelForInference",
    "EncoderOnlyEmbedderM3Trainer",
    "EncoderOnlyEmbedderM3Runner",
]
