#set hyperparameters
BERT_DIR=TinyBERT_4L_312D
OUTPUT_ROOT_DIR=tinybert-base-uncased
DATA_ROOT_DIR=../../datasets/squad

accu=1
ep=30
lr=1
batch_size=32
length=512
torch_seed=9580

NAME=squad_base_cased_lr${lr}e${ep}_4L_student
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
    --doc_stride 128 \
    --per_gpu_train_batch_size ${batch_size} \
    --seed ${torch_seed} \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-4 \
    --do_lower_case \
    --thread 50 \
    --s_opt1 30 \
    --gradient_accumulation_steps ${accu} \
    --overwrite_output_dir \
    --save_steps 1000 \
    --output_encoded_layers false \
    --output_attention_layers false
