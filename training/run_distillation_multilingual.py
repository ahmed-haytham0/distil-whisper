#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# Modified for multilingual support with per-sample language handling.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multilingual Whisper Distillation Training Script.
Supports per-sample language token handling for multilingual datasets.
"""

import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    Audio,
)
from huggingface_hub import create_repo, get_full_repo_name, upload_folder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    get_scheduler,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed.
check_min_version("4.34.0.dev0")
require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = get_logger(__name__)


# ==========================================
# LANGUAGE MAPPING
# ==========================================
LANGUAGE_MAP = {
    'arabic_only': 'ar',
    'english_only': 'en',
    'no_language': 'ar',
    'ar-en': 'ar',
    'arabic': 'ar',
    'english': 'en',
    'ar': 'ar',
    'en': 'en',
}


def get_whisper_lang_code(label: str) -> str:
    """Map dataset labels to Whisper's language codes."""
    if label is None:
        return 'ar'
    return LANGUAGE_MAP.get(str(label).lower().strip(), 'ar')


# ==========================================
# ARGUMENT CLASSES
# ==========================================
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained student Whisper model or model identifier from huggingface.co/models"}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained teacher model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": (
                "Which attention implementation to use. Can be one of:\n"
                "1. `eager` or `None`: default Transformers attention implementation.\n"
                "2. `sdpa`: Flash Attention through PyTorch SDPA.\n"
                "3. `flash_attention_2`: Flash Attention 2 through the Flash Attention package."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to local training dataset directory (parquet format)."},
    )
    eval_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to local evaluation dataset directory (parquet format)."},
    )
    train_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the training dataset to use (via the datasets library)."},
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the training dataset."},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."},
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the evaluation dataset."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing if using non-streaming mode."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Number of examples per batch provided to the `prepare_dataset` function."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="whisper_transcript",
        metadata={"help": "The name of the dataset column containing the text data (pseudo-labels or ground truth)."},
    )
    eval_text_column_name: str = field(
        default="normalized_text",
        metadata={"help": "The name of the dataset column containing the text data in evaluation set."},
    )
    language_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the language label."},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    min_duration_in_seconds: float = field(
        default=0.5,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},
    )
    max_label_length: int = field(
        default=448,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "If set will pad the target sequence to a multiple of the provided value."},
    )
    train_split_name: str = field(
        default="train",
        metadata={"help": "The name of the training data set split to use."},
    )
    eval_split_name: str = field(
        default="train",
        metadata={"help": "The name of the evaluation data set split to use."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use Datasets' streaming mode to load and pre-process the data."},
    )
    wer_threshold: float = field(
        default=None,
        metadata={"help": "Filter training data with WER greater than this threshold."},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={"help": "Whether or not to predict timestamps in the generation step."},
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "The name of the wandb run."},
    )
    wandb_dir: str = field(
        default="./wandb",
        metadata={"help": "The dir where wandb metadata will be stored."},
    )


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    freeze_encoder: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the entire encoder model."},
    )
    freeze_decoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the entire decoder model."},
    )
    freeze_embed_positions: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the decoder embedding positions."},
    )
    temperature: Optional[float] = field(
        default=2.0,
        metadata={"help": "Temperature to anneal the logits when computing the softmax."},
    )
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Weighting assigned to the KL divergence loss."},
    )
    ce_weight: Optional[float] = field(
        default=0.8,
        metadata={"help": "Weighting assigned to the cross-entropy loss."},
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "The data type (dtype) in which to run training. One of `float32`, `float16` or `bfloat16`."},
    )
    save_best_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Number of best models to be saved."},
    )
    push_model_only: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, only push the student model to Hub, not datasets."},
    )


# ==========================================
# DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Preserves language information for per-sample language handling.
    """
    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # dataloader returns a list of features which we convert to a dict
        input_features = {"input_features": [feature["input_features"] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        
        # Keep language info for generation
        if "language" in features[0]:
            batch["language"] = [feature["language"] for feature in features]

        return batch


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    norm_pred_str: List[str],
    norm_label_str: List[str],
    step: int,
    prefix: str = "eval",
    num_lines: int = 50,
):
    """Helper function to log target/predicted transcriptions to weights and biases."""
    if accelerator.is_main_process:
        # Check if wandb tracker is available
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
            prefix_pretty = prefix.replace("/", "-")

            str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
            wandb_tracker.log_table(
                table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
                columns=["Target", "Pred", "Norm Target", "Norm Pred"],
                data=str_data[:num_lines],
                step=step,
            )
        except ValueError:
            # wandb tracker not available, log to console instead
            logger.info(f"Sample predictions at step {step}:")
            for i in range(min(5, len(pred_str))):
                logger.info(f"  Target: {label_str[i][:100]}...")
                logger.info(f"  Pred:   {pred_str[i][:100]}...")


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]
    glob_checkpoints = [path for path in glob_checkpoints if "val-wer" not in path]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    return [checkpoint[1] for checkpoint in checkpoints_sorted]


def sorted_best_checkpoints(output_dir=None, checkpoint_prefix="checkpoint"):
    """Helper function to sort saved best checkpoints."""
    ordering_and_checkpoint_path = []
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]
    
    for path in glob_checkpoints:
        regex_match = re.search(r"val-wer-([0-9]+\.[0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((float(regex_match.groups(1)[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path, reverse=True)
    return [checkpoint[1] for checkpoint in checkpoints_sorted]


def rotate_checkpoints(save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint", sorting_fn=sorted_checkpoints) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    checkpoints_sorted = sorting_fn(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}].")
        shutil.rmtree(checkpoint, ignore_errors=True)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))


def get_parameter_names(model, forbidden_layer_types, forbidden_module=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    result += list(model._parameters.keys())
    return result


# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistillationTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        init_kwargs={
            "wandb": {
                "name": data_args.wandb_name,
                "dir": data_args.wandb_dir
            }
        }
    )

    # 3. Set-up basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    # 4. Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 5. Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            if training_args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(training_args.output_dir).absolute().name,
                    token=training_args.hub_token,
                )
            else:
                repo_name = training_args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=training_args.hub_token)

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 6. Load datasets
    set_seed(training_args.seed)
    raw_datasets = DatasetDict()

    if training_args.do_train:
        if data_args.train_dataset_path is not None:
            logger.info(f"Loading training dataset from: {data_args.train_dataset_path}")
            raw_datasets["train"] = load_dataset(
                "parquet",
                data_dir=data_args.train_dataset_path,
                split=data_args.train_split_name,
                cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
                num_proc=data_args.preprocessing_num_workers,
            )
        elif data_args.train_dataset_name is not None:
            raw_datasets["train"] = load_dataset(
                data_args.train_dataset_name,
                data_args.train_dataset_config_name,
                split=data_args.train_split_name,
                cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
                token=model_args.token,
            )
        else:
            raise ValueError("Either train_dataset_path or train_dataset_name must be provided")
        
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(
                range(min(data_args.max_train_samples, len(raw_datasets["train"])))
            )
        logger.info(f"Training dataset: {len(raw_datasets['train'])} samples")

    if training_args.do_eval:
        if data_args.eval_dataset_path is not None:
            logger.info(f"Loading evaluation dataset from: {data_args.eval_dataset_path}")
            raw_datasets["eval"] = load_dataset(
                "parquet",
                data_dir=data_args.eval_dataset_path,
                split=data_args.eval_split_name,
                cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
                num_proc=data_args.preprocessing_num_workers,
            )
        elif data_args.eval_dataset_name is not None:
            raw_datasets["eval"] = load_dataset(
                data_args.eval_dataset_name,
                data_args.eval_dataset_config_name,
                split=data_args.eval_split_name,
                cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
                token=model_args.token,
            )
        
        if data_args.max_eval_samples is not None and "eval" in raw_datasets:
            raw_datasets["eval"] = raw_datasets["eval"].select(
                range(min(data_args.max_eval_samples, len(raw_datasets["eval"])))
            )
        if "eval" in raw_datasets:
            logger.info(f"Evaluation dataset: {len(raw_datasets['eval'])} samples")

    if not training_args.do_train and not training_args.do_eval:
        raise ValueError("At least one of training or evaluation has to be performed.")

    # 7. Load pretrained model, tokenizer, and feature extractor
    logger.info(f"Loading student model: {model_args.model_name_or_path}")
    logger.info(f"Loading teacher model: {model_args.teacher_model_name_or_path}")
    
    config = WhisperConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    # Override timestamp tokens
    timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
    tokenizer.add_tokens(timestamps)

    # Load teacher model (frozen, for inference only)
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        low_cpu_mem_usage=True,
        torch_dtype=teacher_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Load student model
    student_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=model_args.token,
        low_cpu_mem_usage=True,
        attn_implementation=model_args.attn_implementation,
    )

    if student_model.config.decoder_start_token_id is None or teacher_model.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for both the "
            f"student and teacher model. Got {student_model.config.decoder_start_token_id} for the "
            f"student and {teacher_model.config.decoder_start_token_id} for the teacher."
        )

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()

    def set_trainable_parameters(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad
        module._requires_grad = requires_grad

    # Freeze student encoder
    if training_args.freeze_encoder:
        set_trainable_parameters(student_model.model.encoder, requires_grad=False)
        student_model.model.encoder.gradient_checkpointing = False
        logger.info("Encoder frozen")

    # Freeze student decoder if specified
    if training_args.freeze_decoder:
        set_trainable_parameters(student_model.model.decoder, requires_grad=False)
        student_model.model.decoder.gradient_checkpointing = False
        set_trainable_parameters(student_model.proj_out, requires_grad=True)
        logger.info("Decoder frozen (except proj_out)")

    if training_args.freeze_embed_positions:
        set_trainable_parameters(student_model.model.decoder.embed_positions, requires_grad=False)
        logger.info("Embed positions frozen")

    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad):.3e}"
    )

    # Share encoder if frozen and same architecture
    share_hidden_states = training_args.freeze_encoder and student_model.config.d_model == teacher_model.config.d_model
    if share_hidden_states:
        teacher_model.model.encoder = student_model.model.encoder
        logger.info("Sharing encoder hidden states between student and teacher")

    # Check if multilingual
    is_multilingual = hasattr(teacher_model.generation_config, "is_multilingual") and teacher_model.generation_config.is_multilingual
    logger.info(f"Model is multilingual: {is_multilingual}")

    # Clear forced decoder ids for dynamic language generation
    student_model.generation_config.forced_decoder_ids = None
    student_model.config.forced_decoder_ids = None
    teacher_model.generation_config.forced_decoder_ids = None
    teacher_model.config.forced_decoder_ids = None

    # 8. Create processor and save
    if accelerator.is_main_process:
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
        student_model.generation_config.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()
    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    # 9. Resample speech dataset
    sampling_rate = feature_extractor.sampling_rate
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        Audio(sampling_rate=sampling_rate),
    )

    # 10. Preprocessing
    max_input_length = int(data_args.max_duration_in_seconds * sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * sampling_rate)
    max_label_length = data_args.max_label_length if data_args.max_label_length is not None else student_model.config.max_length

    return_timestamps = data_args.return_timestamps
    timestamp_ids = tokenizer.timestamp_ids()
    timestamp_begin = tokenizer.all_special_ids[-1]
    timestamp_position = 3 if is_multilingual else 1

    decoder_start_token_id = student_model.config.decoder_start_token_id
    decoder_prev_token_id = tokenizer.all_special_ids[-3]

    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    prefetch_factor = training_args.dataloader_prefetch_factor

    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def prepare_train_dataset(batch):
        """Pre-process the training dataset."""
        audio = [sample["array"] for sample in batch["audio"]]
        inputs = feature_extractor(audio, sampling_rate=sampling_rate)
        batch["input_features"] = inputs.input_features
        batch["input_length"] = [len(sample) for sample in audio]

        # Process text targets
        text_column = data_args.text_column_name
        input_str_batched = batch[text_column]
        
        all_token_ids = []
        all_languages = []
        
        for idx, input_str in enumerate(input_str_batched):
            # Get language for this sample
            lang_label = batch.get(data_args.language_column_name, [None] * len(input_str_batched))[idx]
            lang_code = get_whisper_lang_code(lang_label)
            all_languages.append(lang_code)
            
            # Set prefix tokens for this language
            tokenizer.set_prefix_tokens(language=lang_code, task=data_args.task, predict_timestamps=return_timestamps)
            
            # Tokenize
            token_ids = tokenizer(input_str, add_special_tokens=True, max_length=max_label_length, truncation=True).input_ids
            all_token_ids.append(token_ids)

        batch["labels"] = all_token_ids
        batch["language"] = all_languages
        return batch

    def prepare_eval_dataset(batch):
        """Pre-process the evaluation dataset."""
        sample = batch["audio"]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_features"] = inputs.input_features[0]
        batch["input_length"] = len(sample["array"])

        # Get language
        lang_label = batch.get(data_args.language_column_name)
        lang_code = get_whisper_lang_code(lang_label)
        batch["language"] = lang_code
        
        # Set prefix tokens for this language
        tokenizer.set_prefix_tokens(language=lang_code, task=data_args.task, predict_timestamps=return_timestamps)
        
        # Process ground-truth targets for evaluation
        input_str = batch[data_args.eval_text_column_name]
        batch["labels"] = tokenizer(input_str, add_special_tokens=True, max_length=max_label_length, truncation=True).input_ids
        return batch

    # Preprocess datasets
    vectorized_datasets = DatasetDict()
    
    if training_args.do_train:
        raw_datasets_train_features = list(raw_datasets["train"].features.keys())
        with accelerator.main_process_first():
            vectorized_datasets["train"] = raw_datasets["train"].map(
                prepare_train_dataset,
                remove_columns=raw_datasets_train_features,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                num_proc=num_workers,
                desc="Preprocessing train dataset",
            )

    if training_args.do_eval and "eval" in raw_datasets:
        raw_datasets_eval_features = list(raw_datasets["eval"].features.keys())
        with accelerator.main_process_first():
            vectorized_datasets["eval"] = raw_datasets["eval"].map(
                prepare_eval_dataset,
                remove_columns=raw_datasets_eval_features,
                num_proc=num_workers,
                desc="Preprocessing eval dataset",
            )

    # Filter by audio length
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    with accelerator.main_process_first():
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
            num_proc=num_workers,
            desc="Filtering by audio length",
        )

    # Filter by label length
    def is_labels_in_length_range(labels):
        return 0 < len(labels) <= max_label_length

    with accelerator.main_process_first():
        vectorized_datasets = vectorized_datasets.filter(
            is_labels_in_length_range,
            input_columns=["labels"],
            num_proc=num_workers,
            desc="Filtering by label length",
        )

    logger.info(f"Final training samples: {len(vectorized_datasets.get('train', []))}")
    logger.info(f"Final evaluation samples: {len(vectorized_datasets.get('eval', []))}")

    # 11. Define evaluation metrics
    def compute_metrics(preds, labels):
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=return_timestamps)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)
        return {"wer": wer, "wer_ortho": wer_ortho}, pred_str, label_str, norm_pred_str, norm_label_str

    # 12. Define Training Schedule
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    else:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        num_epochs = int(np.ceil(total_train_steps / steps_per_epoch))

    if training_args.eval_steps is None:
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # 13. Define optimizer, LR scheduler, collator
    forbidden_module = [
        module
        for module, flag in [
            (student_model.model.encoder, training_args.freeze_encoder),
            (student_model.model.decoder, training_args.freeze_decoder)
        ]
        if flag
    ] or None

    decay_parameters = get_parameter_names(student_model, [nn.LayerNorm], forbidden_module=forbidden_module)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [param for name, param in student_model.named_parameters() if name in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [param for name, param in student_model.named_parameters() if name not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        decoder_prev_token_id=decoder_prev_token_id,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 14. Define generation arguments
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(student_model.generation_config, "num_beams", 1)
    )

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": return_timestamps,
    }

    # 15. Prepare everything with accelerate
    student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, teacher_model, optimizer, lr_scheduler
    )

    def kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        divergence = divergence.sum() / padding_mask.sum()
        return divergence

    def train_step(batch, temperature=2.0):
        student_model.train()
        teacher_model.eval()

        # Remove language key before forward pass
        batch_for_model = {k: v for k, v in batch.items() if k != 'language'}
        
        student_outputs = student_model(**batch_for_model)
        with torch.no_grad():
            if share_hidden_states:
                encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state.to(dtype=teacher_dtype))
                teacher_outputs = teacher_model(encoder_outputs=encoder_outputs, labels=batch_for_model["labels"])
            else:
                teacher_outputs = teacher_model(**batch_for_model)

        ce_loss = student_outputs.loss
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
        student_distribution = nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1)
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch_for_model["labels"]) * temperature**2

        loss = training_args.ce_weight * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return loss, metrics

    def eval_step(batch):
        student_model.eval()
        teacher_model.eval()

        # Remove language key before forward pass
        batch_for_model = {k: v for k, v in batch.items() if k != 'language'}
        
        with torch.no_grad():
            student_outputs = student_model(**batch_for_model)
            if share_hidden_states:
                encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state.to(dtype=teacher_dtype))
                teacher_outputs = teacher_model(encoder_outputs=encoder_outputs, labels=batch_for_model["labels"])
            else:
                teacher_outputs = teacher_model(**batch_for_model)

        ce_loss = student_outputs.loss
        student_distribution = nn.functional.log_softmax(student_outputs.logits, dim=-1)
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits, dim=-1)
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch_for_model["labels"])

        loss = training_args.ce_weight * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return metrics

    def generate_step(batch):
        student_model.eval()
        # Get language for generation from batch
        languages = batch.get("language", ["ar"])
        
        # For simplicity, use first language in batch (you can extend to per-sample)
        lang = languages[0] if isinstance(languages, list) else languages
        
        gen_kwargs_with_lang = gen_kwargs.copy()
        gen_kwargs_with_lang["language"] = lang
        gen_kwargs_with_lang["task"] = data_args.task
        
        output_ids = accelerator.unwrap_model(student_model).generate(
            batch["input_features"], 
            **gen_kwargs_with_lang
        )
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    # ======================== Training ================================
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info(f"  Num epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0
    best_val_wer = np.inf

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info(f"  Continuing training from checkpoint, epoch {epochs_trained}, step {cur_step}")
        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)

        resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
    else:
        resume_step = None

    for epoch in range(epochs_trained, num_epochs):
        vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=dataloader_num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)

        if resume_step is not None:
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(batch, temperature=training_args.temperature)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # Save checkpoint
                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    accelerator.save_state(output_dir=intermediate_dir)
                    feature_extractor.save_pretrained(intermediate_dir)
                    tokenizer.save_pretrained(intermediate_dir)
                    config.save_pretrained(intermediate_dir)
                    student_model.generation_config.save_pretrained(intermediate_dir)

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir)

                        # Only push model to hub, not datasets
                        if training_args.push_to_hub and training_args.push_model_only:
                            upload_folder(
                                folder_path=intermediate_dir,
                                repo_id=repo_name,
                                repo_type="model",
                                commit_message=f"Saving train state of step {cur_step}",
                            )

                # Evaluation
                if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    student_model.eval()

                    eval_metrics = []
                    eval_preds = []
                    eval_labels = []
                    eval_start = time.time()

                    validation_dataloader = DataLoader(
                        vectorized_datasets["eval"],
                        collate_fn=data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=dataloader_num_workers,
                        prefetch_factor=prefetch_factor,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    for batch in tqdm(
                        validation_dataloader,
                        desc="Evaluating...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        eval_metric = eval_step(batch)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metrics.append(eval_metric)

                        if training_args.predict_with_generate:
                            generated_ids = generate_step(batch)
                            generated_ids, labels = accelerator.gather_for_metrics(
                                (generated_ids, batch["labels"])
                            )
                            eval_preds.extend(generated_ids)
                            eval_labels.extend(labels)

                    eval_time = time.time() - eval_start
                    eval_metrics = {
                        key: torch.mean(torch.stack([d[key] for d in eval_metrics])) for key in eval_metrics[0]
                    }

                    wer_desc = ""
                    if training_args.predict_with_generate:
                        wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(
                            eval_preds, eval_labels
                        )
                        eval_metrics.update(wer_metric)
                        wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
                        log_pred(
                            accelerator,
                            pred_str,
                            label_str,
                            norm_pred_str,
                            norm_label_str,
                            step=cur_step,
                            prefix="eval",
                        )

                    steps_trained_progress_bar.write(
                        f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                        f" {wer_desc})"
                    )

                    log_metric(
                        accelerator,
                        metrics=eval_metrics,
                        train_time=eval_time,
                        step=cur_step,
                        epoch=epoch,
                        prefix="eval",
                    )

                    train_start = time.time()

                    # Save best checkpoint
                    val_wer = wer_metric.get("wer", np.inf)
                    if val_wer < best_val_wer:
                        intermediate_dir = os.path.join(
                            training_args.output_dir, 
                            f"checkpoint-{cur_step}-epoch-{epoch}-val-wer-{val_wer:.3f}"
                        )
                        logger.info(f"Saving new best model, validation WER: {val_wer:.3f}")
                        accelerator.save_state(output_dir=intermediate_dir)
                        feature_extractor.save_pretrained(intermediate_dir)
                        tokenizer.save_pretrained(intermediate_dir)
                        config.save_pretrained(intermediate_dir)
                        student_model.generation_config.save_pretrained(intermediate_dir)

                        accelerator.wait_for_everyone()

                        if accelerator.is_main_process:
                            rotate_checkpoints(
                                training_args.save_best_total_limit, 
                                output_dir=training_args.output_dir, 
                                sorting_fn=sorted_best_checkpoints
                            )
                            accelerator.unwrap_model(student_model).save_pretrained(training_args.output_dir)

                            if training_args.push_to_hub and training_args.push_model_only:
                                upload_folder(
                                    folder_path=training_args.output_dir,
                                    repo_id=repo_name,
                                    repo_type="model",
                                    commit_message=f"Saving best state, step {cur_step}, val wer {val_wer:.3f}",
                                )

                        best_val_wer = val_wer

                # Break condition
                if cur_step == total_train_steps:
                    final_weights_dir = os.path.join(training_args.output_dir, "end-of-training-weights")

                    feature_extractor.save_pretrained(final_weights_dir)
                    tokenizer.save_pretrained(final_weights_dir)
                    config.save_pretrained(final_weights_dir)
                    student_model.generation_config.save_pretrained(final_weights_dir)

                    student_model = accelerator.unwrap_model(student_model)
                    student_model.save_pretrained(final_weights_dir)

                    if training_args.push_to_hub and training_args.push_model_only:
                        upload_folder(
                            folder_path=training_args.output_dir,
                            repo_id=repo_name,
                            repo_type="model",
                            commit_message=f"Saving final weights of step {cur_step}",
                        )

                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
