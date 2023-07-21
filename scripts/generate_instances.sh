batch_dir=data/

python3 self_instruct/generate_instances.py \
    --batch_dir ${batch_dir} \
    --input_file machine_generated_instructions.jsonl \
    --output_file machine_generated_instances.jsonl \
    --max_instances_to_gen 5 \
    --engine "gpt-3.5-turbo-16k-0613" \
    --api_key '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' \
    --request_batch_size 1