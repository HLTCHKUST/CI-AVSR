import torch
from torch import nn
import torchvision.models as models

_HIDDEN_STATES_START_POSITION = 2

class MMWav2Vec2Model(nn.Module):
    def __init__(self, wav2vec2ctc):
        super(MMWav2Vec2Model, self).__init__()
        # Wav2Vec2 Model
        self.wav2vec2ctc = wav2vec2ctc
        
        # Video Model
        self.video_1d_resnet_18 = models.resnet18(pretrained=True)
        self.video_1d_resnet_18.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.video_1d_resnet_18.fc = nn.Identity()
        self.video_time_conv = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding='same')
        self.video_lm_head = nn.Linear(1024, self.wav2vec2ctc.config.vocab_size)
        
    def gradient_checkpointing_enable(self):
        return self.wav2vec2ctc.gradient_checkpointing_enable()
    
    def encode_audio(self, input_values):
        outputs = self.wav2vec2ctc(input_values)
        return outputs
        
    def encode_video(self, video_values):
        batch_size = video_values.shape[0]
        seq_len, n_channel = video_values.shape[1], video_values.shape[2]
        width, height = video_values.shape[3], video_values.shape[4]

        video_values = video_values.reshape(-1, n_channel, width, height) # [B, L, C, W, H] => [B*L, C, W, H]
        video_output = self.video_1d_resnet_18(video_values)# [B*L, C, W, H] => [B*L, D]
        video_output = video_output.reshape(batch_size, seq_len, -1) # [B*L, D] => [B, L, D]
        video_output = self.video_time_conv(video_output.transpose(1, 2)) # [B, L, D] => [B, D', L]
        video_logits = self.video_lm_head(video_output.transpose(1, 2)) # [B, D', L] => [B, L, Vocab]
        
        return video_logits

    def forward(self, input_values, labels=None, attention_mask=None, video_values=None, return_dict=False, *args, **kwargs):
        # encode audio
        outputs = self.encode_audio(input_values, *args, **kwargs)
        logits = outputs.get('logits')
        
        if video_values is not None:
            # encode video
            video_logits = self.encode_video(video_values, *args, **kwargs)
            
            # fuse multimodal
            video_logits = video_logits.repeat(1,2,1) # Convert 25 FPS to 50 FPS
            logits = logits + video_logits[:,:logits.shape[1],:] # Fuse with Audio

        if labels is not None:
            if labels.max() >= self.wav2vec2ctc.config.vocab_size:
                raise ValueError(f"Label values must be < vocab_size: {self.wav2vec2ctc.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self.wav2vec2ctc._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.wav2vec2ctc.config.pad_token_id,
                    reduction=self.wav2vec2ctc.config.ctc_loss_reduction,
                    zero_infinity=self.wav2vec2ctc.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )