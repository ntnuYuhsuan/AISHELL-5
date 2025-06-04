#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.



export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# for debug purpose, please set it to 1 otherwise, set it to 0
export CUDA_LAUNCH_BLOCKING=0

stage=1 # start from 0 if you need to start from data preparation
stop_stage=1


data_dir=/home/38_data1/yhdai/data_raw/test_all

datasets="test_ysw"

nj=6

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#   echo "stage -1: Enhance Eval2 by trained Spatialnet model"
#   # for more details, please refer to https://github.com/wenet-e2e/wenet/tree/main/wenet/bin/download_icmc_asr_data.py
#   python3 wenet/bin/download_icmc_asr_data.py --data_root $data_enhanced --dataset $test_set
# fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Do VAD for enhanced audio data"
  for x in ${datasets} ; do
    # enhanced_data_root dataset threshold HUGGINGFACE_ACCESS_TOKEN
    local/run_silero_vad.sh $data_dir $x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Segment audio data based on VAD results"
  output_dir=/home/38_data1/yhdai/workspace/wenet/examples/aishell/ICMC-ASR_Baseline/track2_asdr/VAD/data/silero_out/${x}
  for x in ${datasets} ; do
    python3 local/do_split_by_vad_rttm.py --input_audio_scp data/vad/$x/wav.scp --rttm_file_dir exp/${x}/ --output_dir $output_dir
  done
fi