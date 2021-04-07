#set hyperparameters
#BERT_DIR=output-bert-base/squad_base_cased_lr3e2_teacher
BERT_DIR=../../models/bert-base-cased-squad2
OUTPUT_ROOT_DIR=output-bert-base-student
DATA_ROOT_DIR=../../datasets/squad_v2

STUDENT_CONF_DIR=student_configs/bert_base_cased_L4.json
accu=1
ep=30
lr=10
augmenter_config_path=augmenter_config.json
intermediate_strategy=skip
## if you use mixup or augmenter, then the actual batch size will be batch_size * 2
batch_size=4
temperature=8
length=512
torch_seed=9580

NAME=L4_squad_base_lr${lr}e${ep}_bert_student
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${NAME}

gpu_nums=4


mkdir -p $OUTPUT_DIR

python -m torch.distributed.launch --nproc_per_node=${gpu_nums} examples/question_answering/run_distiller.py \
    --task_type squad2 \
    --data_dir $DATA_ROOT_DIR \
    --T_model_name_or_path $BERT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length ${length} \
    --S_config_file $STUDENT_CONF_DIR \
    --train \
    --doc_stride 128 \
    --per_gpu_train_batch_size ${batch_size} \
    --augmenter_config_path ${augmenter_config_path} \
    --intermediate_strategy ${intermediate_strategy} \
    --seed ${torch_seed} \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --thread 64 \
    --gradient_accumulation_steps ${accu} \
    --temperature ${temperature} \
    --save_steps 1000 \
    --kd_loss_weight 1.0 \
    --kd_loss_type ce \
    --intermediate_features hidden \
    --mixup
