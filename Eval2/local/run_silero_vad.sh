. ./path.sh || exit 1

stage=0
stop_stage=2

. tools/parse_options.sh

enhanced_data_root=$1
dataset=$2

dataset_prefix=$(echo "$dataset" | cut -d '_' -f 1)
if [ "${dataset_prefix}" == "eval" ]; then
  dataset_prefix=$(echo "$dataset" | cut -d _ -f 1-2)
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "[local/run_vad.sh] stage 0 generate wav.scp file for ${dataset} set"
  mkdir -p data/vad/${dataset}
  ls ${enhanced_data_root}/${dataset_prefix}/*/DX0[1-4]C01.wav | awk -F/ '{print $(NF-1)"_"substr($NF, 1, length($NF)-4), $0}' \
    > data/vad/${dataset}/wav.scp
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[local/run_vad.sh] stage 1 run vad for ${dataset} set"
  python3 local/do_vad_by_silero_vad.py \
    --wav_scp data/vad/${dataset}/wav.scp \
    --save_path exp/silero_vad/${dataset}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[local/run_vad.sh] stage 2 merge session rttms for ${dataset} set"
  python3 local/merge_session_rttms.py \
    --segments_path exp/silero_vad/${dataset} \
    --save_path exp/silero_vad/${dataset}/silero_vad.rttm
fi