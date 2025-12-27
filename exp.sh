#!/bin/bash
set +e 

TASKS=("cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte")
GPU_ID=1

ALPHAS=(0.7)

for alpha in "${ALPHAS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "=================================================="
        echo "🔥 Task: $task | Alpha: $alpha"
        echo "=================================================="
        
        OUTPUT_DIR="final_results/adalora_${task}_alpha${alpha}"
        
        if [ -f "$OUTPUT_DIR/all_results.json" ]; then
            echo "⚠️ $task done, skipping."
            continue
        fi
        
        
        if [ "$task" = "cola" ]; then
            METRIC="matthews_correlation"
            EPOCH=25   
        elif [ "$task" = "mrpc" ]; then
            METRIC="accuracy"
            EPOCH=30   
        elif [ "$task" = "rte" ]; then
            METRIC="accuracy"
            EPOCH=50   
        elif [ "$task" = "stsb" ]; then
            METRIC="pearson"
            EPOCH=25  
        elif [ "$task" = "sst2" ]; then
            METRIC="accuracy"
            EPOCH=20  
        else
            METRIC="accuracy"
            EPOCH=6   
        fi
        
        if [ "$task" = "cola" ] || [ "$task" = "mrpc" ] || [ "$task" = "stsb" ]; then
            T_STEPS=7000
        elif [ "$task" = "rte" ]; then
            T_STEPS=4000
        else
            T_STEPS=60000
        fi

        
        PYTHONUNBUFFERED=1 USE_PEFT_ADALORA=true MAG_ALPHA=$alpha TOTAL_STEPS=$T_STEPS \
        HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU_ID python run_glue.py \
          --model_name_or_path roberta-base \
          --task_name $task \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_device_train_batch_size 128 \
          --learning_rate 3e-4 \
          --num_train_epochs $EPOCH \
          --logging_steps 100 \
          --eval_strategy epoch \
          --save_strategy epoch \
          --metric_for_best_model $METRIC \
          --output_dir $OUTPUT_DIR \
          --overwrite_output_dir \
          --fp16 \
          --report_to none

        rm -rf $OUTPUT_DIR/checkpoint-*
    done
done