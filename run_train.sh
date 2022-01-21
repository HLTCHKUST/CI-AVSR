# Audio Only
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir=./save_ao/0  \
    --model_name_or_path=ctl/wav2vec2-large-xlsr-cantonese \
    --train_manifest_path=dataset/mm_train_metadata.csv \
    --valid_manifest_path=dataset/mm_valid_metadata.csv \
    --test_manifest_path=dataset/mm_test_metadata.csv \
    --num_workers=8 --preprocessing_num_workers=8 \
    --audio_column_name=audio_path --text_column_name=text_path \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=32 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 \
    --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True
    
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir=./save_ao/1  \
    --model_name_or_path=ctl/wav2vec2-large-xlsr-cantonese \
    --train_manifest_path=dataset/mm_train_metadata.csv \
    --valid_manifest_path=dataset/mm_valid_metadata.csv \
    --test_manifest_path=dataset/mm_test_metadata.csv \
    --num_workers=8 --preprocessing_num_workers=8 \
    --audio_column_name=audio_path --text_column_name=text_path \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=32 --dataloader_pin_memory \
    --seed=1 --num_train_epochs=20 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 \
    --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True
    
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir=./save_ao/2  \
    --model_name_or_path=ctl/wav2vec2-large-xlsr-cantonese \
    --train_manifest_path=dataset/mm_train_metadata.csv \
    --valid_manifest_path=dataset/mm_valid_metadata.csv \
    --test_manifest_path=dataset/mm_test_metadata.csv \
    --num_workers=8 --preprocessing_num_workers=8 \
    --audio_column_name=audio_path --text_column_name=text_path \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=32 --dataloader_pin_memory \
    --seed=2 --num_train_epochs=20 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 \
    --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True
    
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir=./save_ao/3  \
    --model_name_or_path=ctl/wav2vec2-large-xlsr-cantonese \
    --train_manifest_path=dataset/mm_train_metadata.csv \
    --valid_manifest_path=dataset/mm_valid_metadata.csv \
    --test_manifest_path=dataset/mm_test_metadata.csv \
    --num_workers=8 --preprocessing_num_workers=8 \
    --audio_column_name=audio_path --text_column_name=text_path \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=32 --dataloader_pin_memory \
    --seed=3 --num_train_epochs=20 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 \
    --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True
    
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir=./save_ao/4  \
    --model_name_or_path=ctl/wav2vec2-large-xlsr-cantonese \
    --train_manifest_path=dataset/mm_train_metadata.csv \
    --valid_manifest_path=dataset/mm_valid_metadata.csv \
    --test_manifest_path=dataset/mm_test_metadata.csv \
    --num_workers=8 --preprocessing_num_workers=8 \
    --audio_column_name=audio_path --text_column_name=text_path \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=32 --dataloader_pin_memory \
    --seed=4 --num_train_epochs=20 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 \
    --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True