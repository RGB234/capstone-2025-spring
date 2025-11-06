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

from FlagEmbedding.finetune.embedder.encoder_only.m3 import EncoderOnlyEmbedderM3ModelForInference
from .modeling import BGEM3
from .trainer import EncoderOnlyEmbedderM3Trainer
from .runner import EncoderOnlyEmbedderM3Runner


__all__ = [
    "EncoderOnlyEmbedderM3ModelArguments",
    "EncoderOnlyEmbedderM3DataArguments",
    "EncoderOnlyEmbedderM3TrainingArguments",
    "BGEM3",
    "EncoderOnlyEmbedderM3ModelForInference",
    "EncoderOnlyEmbedderM3Trainer",
    "EncoderOnlyEmbedderM3Runner",
]
