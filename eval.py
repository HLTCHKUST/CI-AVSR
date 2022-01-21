import os, sys
import logging
import numpy as np
import pandas as pd
import argparse

import torchaudio
import torch
import re
import json 
import librosa
from datasets import load_from_disk, load_dataset, load_metric

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

from datasets import DatasetDict, load_metric, load_from_disk
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
from datasets import set_caching_enabled

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
    print('Load Wav2Vec2 model and processor...')
    config = Wav2Vec2Config.from_pretrained('ctl/wav2vec2-large-xlsr-cantonese')
    config.update({
        "mask_time_prob": 0,
        "mask_time_length": 0,
        "mask_feature_prob": 0,
        "mask_feature_length": 0,
        "gradient_checkpointing": True,
    })
    processor = Wav2Vec2Processor.from_pretrained('ctl/wav2vec2-large-xlsr-cantonese')
    wav2vec2ctc = Wav2Vec2ForCTC(config=config)

    if data_args.use_video:
        model = MMWav2Vec2Model(wav2vec2ctc)
    else:
        model = wav2vec2ctc
    model.load_state_dict(torch.load(f'{model_args.model_name_or_path}/pytorch_model.bin'))
    model.cuda()
    
    print('Loading cached dataset...')
    vectorized_datasets = datasets.load_from_disk(f'{data_args.test_manifest_path}')

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    
    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')
    
    # Instantiate custom data collator
    if data_args.use_video:
        data_collator = DataCollatorMMCTCWithPadding(processor=processor)
    else:
        data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define compute metric function
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # we do not want to group tokens when computing the metrics
        label_strs = processor.batch_decode(pred.label_ids, group_tokens=False)
        mixed_distance, mixed_tokens = 0, 0
        char_distance, char_tokens = 0, 0

        pred_strs = list(map(lambda pred_str: pred_str[:-1].strip(), pred_strs))
        label_strs = list(map(lambda label_str: label_str.replace('[UNK]','#'), label_strs))
        for pred_str, label_str in zip(pred_strs, label_strs):
            # Calculate 
            m_pred = tokenize_for_mer(pred_str)
            m_ref = tokenize_for_mer(label_str)
            mixed_distance += editdistance.distance(m_pred, m_ref)
            mixed_tokens += len(m_ref)

            c_pred = tokenize_for_cer(pred_str)
            c_ref = tokenize_for_cer(label_str)
            char_distance += editdistance.distance(c_pred, c_ref)
            char_tokens += len(c_ref)
            
        
        f = open(f'{training_args.output_dir}/test.results', 'w')
        f.writelines([item+'\n' for item in pred_strs])
        f.close()
        f = open(f'{training_args.output_dir}/test.label', 'w')
        f.writelines([item+'\n' for item in label_strs])
        f.close()
        
        mer = mixed_distance / mixed_tokens
        cer = char_distance / char_tokens

        return {"mer": mer, "cer": cer}
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["test"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["test"] if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )
    
    ###
    # Evaluation Phase
    ###
    results = {}
    logger.info("*** Test Phase ***")
    metrics = trainer.evaluate(eval_dataset=vectorized_datasets["test"])
    metrics["eval_samples"] = len(vectorized_datasets["test"])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "speech-recognition",
        "tags": ["automatic-speech-recognition", "ASCEND"],
        "dataset_args": "Config: na",
        "dataset": "ASCEND",
        "language": "zh-en"
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results
    
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
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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