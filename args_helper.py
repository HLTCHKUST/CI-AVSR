from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to utilize.
    """
    model_name_or_path: Optional[str] = field(
        default="ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt", metadata={"help": "The path of the HuggingFace model."}
    )
    mask_time_prob: float = field(
        default=0.065,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )
    mask_time_length: int = field(
        default=3,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.004,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: int = field(
        default=3,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and preprocessing pipeline.
    """
    train_manifest_path: Optional[str] = field(
        default="dataset/train_metadata.csv", metadata={"help": "The path of the training dataset to use."}
    )
    valid_manifest_path: Optional[str] = field(
        default="dataset/validation_metadata.csv", metadata={"help": "The path of the validation dataset to use."}
    )
    test_manifest_path: Optional[str] = field(
        default="dataset/test_metadata.csv", metadata={"help": "The path of the testing dataset to use."}
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the dataset."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only run preprocessing."},
    )
    audio_column_name: Optional[str] = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio path. Defaults to 'audio'"},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    video_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the video data. Defaults to 'None'"},
    )
    use_video: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to incorporate video data for ASR or not. Defaults to 'False'"},
    )