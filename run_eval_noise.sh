# Eval udio Only Noisy
CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/0/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/0/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/1/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/1/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/2/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/2/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/3/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/3/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/4/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/4/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/5/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/5/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/6/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/6/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/7/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/7/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/8/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/8/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/9/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/9/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

CUDA_VISIBLE_DEVICES=1 python eval.py --output_dir=./save_ao_noisy/all/eval \
--model_name_or_path=save_ao/14045/checkpoint-4500  --test_manifest_path=cache_noisy/all/preprocess_data.arrow \
--audio_column_name=audio_path --text_column_name=text_path  --video_column_name=lip_image_path \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16     --dataloader_num_workers=32 \
--dataloader_pin_memory     --seed=0 --num_train_epochs=20 --learning_rate=5e-5  \
--logging_strategy=steps --logging_steps=10 \
--evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=30 \
--save_steps=1 --save_strategy=epoch --save_total_limit=1 \
--metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True

