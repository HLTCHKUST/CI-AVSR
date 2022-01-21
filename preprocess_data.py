import os, sys
import logging
import numpy as np
import pandas as pd
import argparse
import glob

import torchaudio
import torch
import re
import json 
import librosa
from datasets import DatasetDict
import torchvision.transforms as T
import torchvision

from transformers import (
    set_seed,
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EarlyStoppingCallback
)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import datasets
import pickle

import editdistance
import jieba
from itertools import chain

import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from args_helper import ModelArguments, DataArguments

import datasets
from datasets import load_from_disk, set_caching_enabled

from utils import CHARS_TO_IGNORE, remove_special_characters, extract_all_chars, tokenize_for_mer, tokenize_for_cer
from data_utils import speech_file_to_array_fn, load_dataset 
from data_collator_ctc import DataCollatorCTCWithPadding, DataCollatorMMCTCWithPadding
from mm_wrapper import MMWav2Vec2Model

set_caching_enabled(True)
logger = logging.getLogger(__name__)    

#####
# Main Functions
#####
def run(model_args, data_args, training_args):
    ###
    # Prepare Processor & Model    
    ###
    training_args.gradient_checkpointing = True
    print('Load Wav2Vec2 processor...')
    processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)
    base_path = '/'.join(data_args.test_manifest_path.split('/')[:-1])

    ###
    # Prepare Dataset
    ###
    raw_datasets = DatasetDict()
    print('Loading test dataset...')
    raw_datasets["test"] = load_dataset(data_args.test_manifest_path, data_args.preprocessing_num_workers, 
            data_args.audio_column_name, data_args.text_column_name, data_args.video_column_name)

    print('Preprocess dataset...')

    # Remove ignorable characters
    print('Removing ignorable characters')
    chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
    def remove_special_characters(batch):
        if chars_to_ignore_re is not None:
            batch['transcription'] = re.sub(chars_to_ignore_re, "", batch['transcription']).lower() + " "
        else:
            batch['transcription'] = batch['transcription'].lower() + " "
        return batch

    with training_args.main_process_first(desc="dataset map special characters removal"):
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            num_proc=data_args.preprocessing_num_workers,
            desc="remove special characters from datasets",
            load_from_cache_file=True
        )

    # Preprocess audio sample and label text
    print('Vectorize dataset...')

    def prepare_dataset(batch):
        # Preprocess audio
        batch["input_values"] = processor(batch["speech_sample"]).input_values[0]

        # Preprocess text
        with processor.as_target_processor():
            batch["labels"] = processor(batch['transcription']).input_ids

        return batch

    removable_column_names = raw_datasets["test"].column_names
    if data_args.use_video:
        removable_column_names.remove('video_path')

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=removable_column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess datasets",
            load_from_cache_file=False
        )

    # Preprocess video sample
    if data_args.use_video:
        print('Load video data...')
        img_transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((32,32))
        ])

        def load_video_data(batch):
            image_buffers = []
            video_path = batch["video_path"]
            for image_path in glob.glob(f'{base_path}/{video_path}/*.jpg'):
                image = torchvision.io.read_image(image_path) / 255
                image = img_transforms(image)
                image_buffers.append(image)
            batch["video_values"] = image_buffers # L, C, H, W
            return batch

        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = vectorized_datasets.map(
                load_video_data,
                remove_columns=['video_path'],
                num_proc=data_args.preprocessing_num_workers,
                desc="preprocess datasets",
                load_from_cache_file=False
            )
    vectorized_datasets.save_to_disk(f'{training_args.output_dir}/preprocess_data.arrow')

    logger.info(f"Data preprocessing finished. Files cached at {training_args.output_dir}/preprocess_data.arrow")
    return
    
#####
# Entry Point
#####
def main():
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set random seed
    set_seed(training_args.seed)

    ###
    # Prepare logger
    ###
    # Init logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)
    
    ###
    # RUN RUN RUN!!!
    ###
    run(model_args, data_args, training_args)
    
if __name__ == '__main__':
    main()