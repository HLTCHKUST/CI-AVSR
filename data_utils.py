import torchaudio
import torch
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset, load_metric

#####
# Data Loading Function
#####
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch['path'])
    batch["speech_sample"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    return batch

def load_dataset(manifest_file, num_proc, audio_column_name, text_column_name, video_column_name=None):
    base_path = '/'.join(manifest_file.split('/')[:-1])
    
    manifest_df = pd.read_csv(manifest_file)
    manifest_df['path'] = manifest_df[audio_column_name].apply(lambda path: f'{base_path}/{path}')
    manifest_df['transcription'] = manifest_df[text_column_name].apply(
        lambda path: open(f'{base_path}/{path}', "r").read()
    )
    if video_column_name:
        manifest_df = manifest_df.rename({video_column_name: 'video_path'}, axis='columns')
    
    batches = Dataset.from_pandas(manifest_df)
    batches = batches.map(speech_file_to_array_fn, num_proc=num_proc)
    return batches
