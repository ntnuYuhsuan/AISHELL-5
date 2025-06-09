import os
import argparse

parser = argparse.ArgumentParser(description="get raw wav")
parser.add_argument("--data-dir", type=str, help="data root dir")
parser.add_argument("--save-path", type=str, help="output dir")
args = parser.parse_args()
data_dir = args.data_dir
save_path = args.save_path

# os.makedirs(save_path, exist_ok=True)

# Open the file in write mode ('w'), not read mode ('r')
# with open(os.path.join(save_path, "wav.scp"), "w") as wf :
#     for wav_name in os.listdir(data_dir):
#         wav_path = os.path.join(data_dir, wav_name)
#         key = wav_name.split(".")[0]
#         wf.write(f"{key} {wav_path}\n")

def get_raw(save_path, data_dir):
    with open(os.path.join(save_path, "wav.scp"), "w") as wf, open(os.path.join(save_path, "text"), "w") as tf :
        for wav_name in os.listdir(data_dir):
            wav_path = os.path.join(data_dir, wav_name)
            key = wav_name.replace(".wav", "")
            wf.write(f"{key} {wav_path}\n")
            tf.write(f"{key} 空\n")

if __name__ == '__main__':
    root="/home/38_data1/yhdai/workspace/wenet/examples/aishell/ICMC-ASR_Baseline/track2_asdr/VAD/data/silero_out/test_ysw"
    sets = ['20230414123246_1096394047034208256', '20230418114211_1097831018218496000', '20230505093001_1103958901431672832', '20230505143802_1104036908526186496']
    for dataset in sets:
        save_path = f"/home/38_data1/yhdai/workspace/wenet/examples/aishell/ICMC-ASR_Baseline/track2_asdr/VAD/data/silero_out/test_ysw/{dataset}"
        data_dir = save_path + "/wavs"
        os.makedirs(save_path, exist_ok=True)
        get_raw(save_path, data_dir)