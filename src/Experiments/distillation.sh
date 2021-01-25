#set hyperparameters
#BERT_DIR=output-bert-base/squad_base_cased_lr3e2_teacher
BERT_DIR=../../models/roberta-large-finetuned-squad2
OUTPUT_ROOT_DIR=output-bert-base-student
DATA_ROOT_DIR=../../datasets/squad

STUDENT_CONF_DIR=configs/roberta_large_config/L4t.json
accu=4
ep=50
lr=10
batch_size=8
temperature=8
length=512
torch_seed=9580

NAME=squad2_base_lr${lr}e${ep}_roberta_bert_student
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}

gpu_nums=4


mkdir -p $OUTPUT_DIR

python -m torch.distributed.launch --nproc_per_node=${gpu_nums} examples/question_answering/run_distiller.py \
    --model_type roberta \
    --model_type_student bert \
    --data_dir $DATA_ROOT_DIR \
    --model_name_or_path $BERT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length ${length} \
    --bert_config_file_S $STUDENT_CONF_DIR \
    --do_train \
    --do_eval \
    --doc_stride 128 \
    --per_gpu_train_batch_size ${batch_size} \
    --seed ${torch_seed} \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --thread 40 \
    --s_opt1 30 \
    --version_2_with_negative \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature} \
    --overwrite_output_dir \
    --save_steps 1000 \
    --do_lower_case \
    --output_encoded_layers false \
    --output_attention_layers false \
    --output_att_score true \
    --output_att_sum false  \
    --matches L4_hidden_smmd_large \
              L4t_hidden_mse_large \
              L4_attention_ce_large \
    --tag RB \
