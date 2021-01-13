#set hyperparameters
BERT_DIR=../../models/bert-base-cased
OUTPUT_ROOT_DIR=output-bert-base
DATA_ROOT_DIR=../../datasets/squad

accu=1
ep=2
lr=1
batch_size=12
length=512
torch_seed=9580

NAME=squad_base_lr${lr}e${ep}_teacher
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR

python -u examples/question_answering/run_fine_tune.py \
    --model_type bert \
    --data_dir $DATA_ROOT_DIR \
    --do_lower_case \
    --model_name_or_path $BERT_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride 320 \
    --max_seq_length ${length} \
    --per_gpu_train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-4 \
    --ckpt_frequency 1 \
    --thread 40 \
    --schedule slanted_triangular \
    --s_opt1 30 \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --overwrite_output_dir \
    --output_encoded_layers false \
    --output_attention_layers false
