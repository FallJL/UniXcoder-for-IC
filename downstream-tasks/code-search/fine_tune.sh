lang=python
python run.py \
    --output_dir saved_models/CSN/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file dataset/CSN/$lang/train.jsonl \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \