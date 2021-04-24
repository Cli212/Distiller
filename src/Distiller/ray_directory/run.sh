TEACHER_DIR=howey/electra-base-mnli
STUDENT_DIR=huawei-noah/TinyBERT_General_4L_312D

python run.py --T_model_name_or_path $TEACHER_DIR --S_model_name_or_path STUDENT_DIR --output_dir ./outputs/