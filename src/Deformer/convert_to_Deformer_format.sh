deformer_dir=data/datasets/deformer
mkdir -p ${deformer_dir}

# squad v1.1
for version in 1.1; do
    data_dir=data/datasets/squad_v${version}
    for split in dev train; do
        python tools/convert_squad.py ${data_dir}/${split}-v${version}.json \
        ${deformer_dir}/squad_v${version}-${split}.jsonl
    done
done

# mnli
data_dir=data/datasets/mnli
python tools/convert_pair_dataset.py ${data_dir}/train.tsv ${deformer_dir}/mnli-train.jsonl -t mnli
python tools/convert_pair_dataset.py ${data_dir}/dev_matched.tsv ${deformer_dir}/mnli-dev.jsonl  -t mnli

# qqp
data_dir=data/datasets/qqp
python tools/convert_pair_dataset.py ${data_dir}/train.tsv ${deformer_dir}/qqp-train.jsonl -t qqp
python tools/convert_pair_dataset.py ${data_dir}/dev.tsv ${deformer_dir}/qqp-dev.jsonl -t qqp

# boolq
data_dir=data/datasets/BoolQ
python tools/convert_pair_dataset.py ${data_dir}/train.jsonl ${deformer_dir}/boolq-train.jsonl -t boolq
python tools/convert_pair_dataset.py ${data_dir}/val.jsonl ${deformer_dir}/boolq-dev.jsonl -t boolq

# race
data_dir=data/datasets/RACE
python tools/convert_race.py ${data_dir}/train ${deformer_dir}/race-train.jsonl
python tools/convert_race.py ${data_dir}/dev ${deformer_dir}/race-dev.jsonl

cd ${deformer_dir}

cat squad_v1.1-train.jsonl | shuf > squad_v1.1-train-shuf.jsonl
head -n8760 squad_v1.1-train-shuf.jsonl > squad_v1.1-tune.jsonl
tail -n78839 squad_v1.1-train-shuf.jsonl > squad_v1.1-train.jsonl

cat boolq-train.jsonl | shuf > boolq-train-shuf.jsonl
head -n943 boolq-train-shuf.jsonl > boolq-tune.jsonl
tail -n8484 boolq-train-shuf.jsonl > boolq-train.jsonl

cat race-train.jsonl | shuf > race-train-shuf.jsonl
head -n8786 race-train-shuf.jsonl > race-tune.jsonl
tail -n79080 race-train-shuf.jsonl > race-train.jsonl

cat qqp-train.jsonl | shuf > qqp-train-shuf.jsonl
head -n36385 qqp-train-shuf.jsonl > qqp-tune.jsonl
tail -n327464 qqp-train-shuf.jsonl > qqp-train.jsonl

cat mnli-train.jsonl | shuf > mnli-train-shuf.jsonl
head -n39270 mnli-train-shuf.jsonl > mnli-tune.jsonl
tail -n353432 mnli-train-shuf.jsonl > mnli-train.jsonl

for model in bert ebert; do
  for task in squad_v1.1 mnli qqp boolq race; do
    python prepare.py -m ${model} -t ${task} -s dev
    python prepare.py -m ${model} -t ${task} -s tune
    python prepare.py -m ${model} -t ${task} -s train -sm tf
  done
done