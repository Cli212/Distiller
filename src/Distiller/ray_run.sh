 TEACHER_DIR=howey/electra-base-mnli
STUDENT_DIR=huawei-noah/TinyBERT_General_4L_312D
DATA_ROOT_DIR=../../datasets/glue_data/MNLI
OUTPUT_ROOT_DIR=output-student

#STUDENT_CONF_DIR=student_configs/bert_base_cased_L4.json
accu=1
ep=30
lr=1
#augmenter_config_path=augmenter_config.json
intermediate_strategy=skip
intermediate_features=hidden
intermediate_loss_type=mse
## if you use mixup or augmenter, then the actual batch size will be batch_size * 2
batch_size=32
temperature=8
length=128
torch_seed=9580
task_name=mnli
task_type=glue
NAME=electra_tiny_lr${lr}e-4_e${ep}_${task_type}_${task_name}
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
gpu_nums=1

mkdir -p $OUTPUT_DIR

python ray_run.py \
    --task_type ${task_type} \
    --task_name ${task_name} \
    --data_dir $DATA_ROOT_DIR \
    --T_model_name_or_path $TEACHER_DIR \
    --S_model_name_or_path $STUDENT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length ${length} \
    --train \
    --eval \
    --doc_stride 128 \
    --per_gpu_train_batch_size ${batch_size} \
    --seed ${torch_seed} \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-4 \
    --thread 64 \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature}