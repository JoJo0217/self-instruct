batch_dir=data/

python3 self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 52 \
    --seed_tasks_path 'data/seed_tasks_merged.jsonl' \
    --engine "davinci" \
    --api_key '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' \
    --engine "gpt-3.5-turbo-16k-0613"