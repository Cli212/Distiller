#set hyperparameters
BERT_DIR=../../models/bert-base-cased
OUTPUT_ROOT_DIR=output-bert-base
DATA_ROOT_DIR=../../datasets/squad

accu=5
ep=3
lr=1
batch_size=12
length=512
torch_seed=9580

NAME=squad_base_lr${lr}e${ep}_teacher
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}
gpu_nums=4

mkdir -p $OUTPUT_DIR

python -m torch.distributed.launch --nproc_per_node=${gpu_nums} examples/question_answering/run_finetune.py \
    --model_type bert \
    --data_dir $DATA_ROOT_DIR \
    --model_name_or_path $BERT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length ${length} \
    --do_train \
    --do_eval \
    --doc_stride 320 \
    --per_gpu_train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-4 \
    --thread 40 \
    --s_opt1 30 \
    --gradient_accumulation_steps ${accu} \
    --overwrite_output_dir \
    --save_steps 1000 \
    --do_lower_case \
    --output_encoded_layers false \
    --output_attention_layers false
