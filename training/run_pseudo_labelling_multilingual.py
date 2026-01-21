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
Multilingual Pseudo-labelling audio data using Whisper model.
Supports per-sample language token handling for multilingual datasets.
"""

import csv
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    Audio,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from soundfile import LibsndfileError
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
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
    Arguments pertaining to which model/config/tokenizer we are going to use.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
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
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "processor name or path if not the same as model_name"},
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
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "The data type (dtype) in which to load the model weights. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
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
    Arguments pertaining to what data we are going to input our model for pseudo-labelling.
    """
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to local parquet/arrow dataset directory."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library). Use dataset_path for local."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: str = field(
        default="train",
        metadata={"help": "The name of the data set split to use."},
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
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=500,
        metadata={"help": "The batch size to use for the dataset pre-processing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="normalized_text",
        metadata={"help": "The name of the dataset column containing the text data."},
    )
    language_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the language label."},
    )
    id_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the id data. Auto-generated if not provided."},
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
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples to this value if set."},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={"help": "Whether to return the timestamps with the text."},
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    wandb_project: str = field(
        default="distil-whisper-pseudo",
        metadata={"help": "The name of the wandb project."},
    )


# ==========================================
# DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: Any
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # dataloader returns a list of features which we convert to a dict
        model_input_name = self.processor.model_input_names[0]
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        
        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )
        
        # Keep language info for generation
        if "language" in features[0]:
            batch["language"] = [feature["language"] for feature in features]
        
        # Keep labels for WER computation
        if "labels" in features[0]:
            label_features = {"input_ids": [feature["labels"] for feature in features]}
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                max_length=self.max_target_length,
                padding=self.target_padding,
                return_tensors="pt",
            )
            # replace padding with -100 to ignore correctly when computing the loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
        
        return batch


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    prefix: str = "eval",
):
    """Helper function to log all evaluation metrics."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    accelerator.log(log_metrics)


# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup dtype
    if model_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif model_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    # 3. Initialize the accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers(project_name=data_args.wandb_project)

    # 4. Set-up basic logging
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
    
    logger.info("Evaluation parameters %s", training_args)

    # 5. Load dataset
    logger.info("Loading dataset...")
    
    if data_args.dataset_path is not None:
        # Load local parquet dataset
        raw_datasets = load_dataset(
            "parquet",
            data_dir=data_args.dataset_path,
            split=data_args.dataset_split_name,
            cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
        )
    elif data_args.dataset_name is not None:
        # Load from Hub
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.dataset_split_name,
            cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
            token=model_args.token,
        )
    else:
        raise ValueError("Either dataset_path or dataset_name must be provided")
    
    # Apply max_samples limit
    if data_args.max_samples is not None:
        raw_datasets = raw_datasets.select(range(min(data_args.max_samples, len(raw_datasets))))
    
    logger.info(f"Dataset loaded: {len(raw_datasets)} samples")
    logger.info(f"Dataset columns: {raw_datasets.column_names}")

    # Validate columns
    if data_args.audio_column_name not in raw_datasets.column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset. "
            f"Available columns: {', '.join(raw_datasets.column_names)}."
        )
    if data_args.text_column_name not in raw_datasets.column_names:
        raise ValueError(
            f"--text_column_name '{data_args.text_column_name}' not found in dataset. "
            f"Available columns: {', '.join(raw_datasets.column_names)}."
        )
    if data_args.language_column_name not in raw_datasets.column_names:
        logger.warning(
            f"--language_column_name '{data_args.language_column_name}' not found in dataset. "
            f"Using default language 'ar' for all samples."
        )

    # 6. Load pretrained model, tokenizer, and feature extractor
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
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
    processor = WhisperProcessor.from_pretrained(
        model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=model_args.token,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model.eval()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Check if multilingual
    is_multilingual = hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual
    logger.info(f"Model is multilingual: {is_multilingual}")

    # Clear forced decoder ids for dynamic language generation
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # 7. Resample speech dataset
    sampling_rate = feature_extractor.sampling_rate
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        Audio(sampling_rate=sampling_rate),
    )

    # 8. Preprocessing
    max_input_length = int(data_args.max_duration_in_seconds * sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * sampling_rate)
    max_label_length = data_args.max_label_length
    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    language_column_name = data_args.language_column_name
    model_input_name = feature_extractor.model_input_names[0]
    
    normalizer = BasicTextNormalizer()

    def prepare_dataset(batch):
        """Prepare dataset with audio features and language info."""
        try:
            # Process audio
            sample = batch[audio_column_name]
            if sample is None or sample.get("array") is None:
                return None
            
            inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
            batch[model_input_name] = inputs.get(model_input_name)[0]
            batch["input_length"] = len(sample["array"])
            
            # Get language
            if language_column_name in batch:
                batch["language"] = get_whisper_lang_code(batch[language_column_name])
            else:
                batch["language"] = "ar"
            
            # Get text for reference
            input_str = batch[text_column_name]
            batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids
            batch["reference_text"] = input_str
            
            return batch
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            return None

    # Filter by audio length first
    def is_audio_valid(sample):
        try:
            audio = sample[audio_column_name]
            if audio is None or audio.get("array") is None:
                return False
            length = len(audio["array"])
            return min_input_length < length < max_input_length
        except:
            return False

    logger.info("Filtering dataset by audio length...")
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.filter(
            is_audio_valid,
            num_proc=data_args.preprocessing_num_workers,
            desc="Filtering by audio length",
        )
    logger.info(f"Filtered dataset: {len(raw_datasets)} samples")

    # Prepare dataset
    logger.info("Preprocessing dataset...")
    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            num_proc=data_args.preprocessing_num_workers,
            desc="Preprocessing dataset",
        )

    # Create output directory
    output_dir = training_args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 9. Load Metric
    metric = evaluate.load("wer")

    # 10. Setup DataLoader
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    dataloader_num_workers = training_args.dataloader_num_workers

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 11. Prepare model
    model = accelerator.prepare(model)

    # 12. Generation loop
    logger.info("***** Running Pseudo-labelling *****")
    logger.info(f"  Num examples = {len(vectorized_datasets)}")
    logger.info(f"  Batch size = {per_device_eval_batch_size}")
    logger.info(f"  Return timestamps = {data_args.return_timestamps}")

    eval_dataloader = DataLoader(
        vectorized_datasets,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)

    all_predictions = []
    all_references = []
    all_languages = []
    
    eval_start = time.time()
    batches = tqdm(
        eval_dataloader, 
        desc="Generating transcriptions...", 
        disable=not accelerator.is_local_main_process
    )

    for step, batch in enumerate(batches):
        input_features = batch["input_features"]
        languages = batch.get("language", ["ar"] * len(input_features))
        
        # Generate for each unique language in the batch
        unique_langs = list(set(languages))
        generated_texts = [""] * len(languages)
        
        for lang in unique_langs:
            # Find indices for this language
            lang_indices = [i for i, l in enumerate(languages) if l == lang]
            if not lang_indices:
                continue
            
            # Get input features for this language
            lang_input_features = input_features[lang_indices]
            
            # Generate with language-specific settings
            gen_kwargs = {
                "max_length": max_label_length,
                "num_beams": 1,  # Greedy for speed
                "return_timestamps": data_args.return_timestamps,
                "language": lang,
                "task": data_args.task,
            }
            
            with torch.no_grad():
                generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
                generated_ids = generate_fn(lang_input_features, **gen_kwargs)
            
            # Decode predictions
            pred_texts = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=False, 
                decode_with_timestamps=data_args.return_timestamps
            )
            
            # Store predictions at correct indices
            for i, idx in enumerate(lang_indices):
                generated_texts[idx] = pred_texts[i]
        
        # Gather predictions across processes
        generated_texts_gathered = accelerator.gather_for_metrics(generated_texts)
        languages_gathered = accelerator.gather_for_metrics(languages)
        
        if "labels" in batch:
            labels = batch["labels"]
            labels_gathered = accelerator.gather_for_metrics(labels)
            # Decode reference texts
            labels_gathered = labels_gathered.cpu().numpy()
            for idx in range(len(labels_gathered)):
                labels_gathered[idx][labels_gathered[idx] == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(labels_gathered, skip_special_tokens=True)
            all_references.extend(ref_texts)
        
        all_predictions.extend(generated_texts_gathered)
        all_languages.extend(languages_gathered)
        
        # Log progress
        if step % training_args.logging_steps == 0 and step > 0:
            batches.write(f"Step {step}: Generated {len(all_predictions)} transcriptions")

    eval_time = time.time() - eval_start
    logger.info(f"Pseudo-labelling completed in {eval_time:.2f} seconds")
    logger.info(f"Total transcriptions: {len(all_predictions)}")

    # 13. Compute WER if references available
    if all_references:
        norm_pred_str = [normalizer(pred) for pred in all_predictions]
        norm_label_str = [normalizer(label) for label in all_references]
        
        # Filter empty references
        valid_indices = [i for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        filtered_preds = [norm_pred_str[i] for i in valid_indices]
        filtered_labels = [norm_label_str[i] for i in valid_indices]
        
        if filtered_labels:
            wer = 100 * metric.compute(predictions=filtered_preds, references=filtered_labels)
            logger.info(f"WER on reference transcriptions: {wer:.2f}%")
            
            log_metric(
                accelerator,
                metrics={"wer": wer},
                train_time=eval_time,
                prefix="eval",
            )

    # 14. Save results locally
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Save as CSV
        output_csv = os.path.join(output_dir, "pseudo_labels.csv")
        csv_data = []
        for i in range(len(all_predictions)):
            row = {
                "id": i,
                "whisper_transcript": all_predictions[i],
                "language": all_languages[i] if i < len(all_languages) else "ar",
            }
            if all_references and i < len(all_references):
                row["reference_text"] = all_references[i]
            csv_data.append(row)
        
        with open(output_csv, "w", encoding="UTF8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"Saved pseudo-labels to {output_csv}")
        
        # Also save as parquet for distillation
        try:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            output_parquet = os.path.join(output_dir, "pseudo_labels.parquet")
            df.to_parquet(output_parquet, index=False)
            logger.info(f"Saved pseudo-labels to {output_parquet}")
        except ImportError:
            logger.warning("pandas not installed, skipping parquet export")

    accelerator.end_training()
    logger.info("Done!")


if __name__ == "__main__":
    main()
