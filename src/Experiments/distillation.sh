#set hyperparameters
BERT_DIR=output-bert-base/squad_base_lr1e3_teacher
OUTPUT_ROOT_DIR=output-bert-base-student
DATA_ROOT_DIR=../../datasets/squad

STUDENT_CONF_DIR=./configs/bert_base_uncased_config/bert_config_L6.json
accu=1
ep=50
lr=1
batch_size=8
temperature=8
length=512
torch_seed=9580

NAME=squad_base_lr${lr}e${ep}_L6_student
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}



mkdir -p $OUTPUT_DIR

python -u examples/question_answering/run_distiller.py \
    --model_type bert \
    --data_dir $DATA_ROOT_DIR \
    --model_name_or_path $BERT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length ${length} \
    --bert_config_file_S $STUDENT_CONF_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride 320 \
    --per_gpu_train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-4 \
    --thread 40 \
    --s_opt1 30 \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature} \
    --overwrite_output_dir \
    --save_steps 1000 \
    --do_lower_case \
    --output_encoded_layers false \
    --output_attention_layers false \
    --output_att_score true \
    --output_att_sum false  \
    --matches L3_hidden_mse \
              L3_hidden_smmd \
    --tag RB \
