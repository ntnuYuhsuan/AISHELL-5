#!/bin/bash

stage=0
stop_stage=1



data_dir=$1
dataset=$2

dataset_prefix=$(echo "$dataset" | cut -d '_' -f 1)
if [ "${dataset_prefix}" == "eval" ]; then
  dataset_prefix=$(echo "$dataset" | cut -d _ -f 1-2)
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "[local/run_vad.sh] stage 0 generate wav.scp file for ${dataset} set"
  python local/get_raw.py \
    --data-dir $data_dir/${dataset} \
    --save-path data/vad/${dataset}
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[local/run_vad.sh] stage 1 run vad for ${dataset} set"
  python3 local/do_vad_by_silero_vad.py \
    --wav_scp data/vad/${dataset}/wav.scp \
    --save_path exp/${dataset}
fi

