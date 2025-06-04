import os
import re
from tqdm import tqdm  # 导入 tqdm
import numpy as np
from scipy.io import wavfile



def get_parser_args():
    '''
    get data process related parameters
    
    '''
    parser = argparse.ArgumentParser(description="Data Preparation")

    parser.add_argument("aishell5_data_root", type=str, help="Path to AISHELL-5 data root directory.")

    parser.add_argument("enchanced_aishell5_data_root", type=str, help="Path to Enchaned AISHELL-5 data root directory.")

    parser.add_argument("aishell5_set_types", type=list, help="List of ICMC set types.")

    parser.add_argument("spk_zone_id_dir", type=str, help="Path to spk_zone_id_dir.")
    
    parser.add_argument("merged_multichannel_wav_dir", type=str, help="Path to merged multichannel wav directory.")
    
    parser.add_argument("split_info_dir", type=str, help="Path to split info directory.")
    
    parser.add_argument("split_wav_dir", type=str, help="Path to split wav directory.")
    
    parser.add_argument("noise_list_path", type=str, help="Path to noise list file.")
    
    parser.add_argument("label_data_path", type=str, help="Path to label data file.")
    
    parser.add_argument("multichannel_eval_data_list_dir", type=str, help="Path to multichannel eval data list directory.")

    args = parser.parse_args()
    
    return args


class AudioProcessor:
    def __init__(self, aishell_data_root, icmc_set_types, enhanced_data_root, noise_data_root, output_dirs):
        self.aishell_data_root = aishell_data_root
        self.icmc_set_types = icmc_set_types
        self.enhanced_data_root = enhanced_data_root
        self.noise_data_root = noise_data_root
        self.output_dirs = output_dirs  # 字典，保存所有输出目录
        self._verify_paths()

    def _verify_paths(self):
        # 检查路径是否存在
        for path in [self.aishell_data_root, self.enhanced_data_root, self.noise_data_root]:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist. Please check the input paths.")

    def process_data(self):
        # 步骤 1: 获取说话人和区域信息
        self._get_spk_zone_id()

        # 步骤 2: 合并单通道 WAV 文件
        self._merge_single_channel_wavs()

        # 步骤 3: 截取每个音频片段的信息
        self._get_split_info()

        # 步骤 4: 截取多通道音频片段
        self._split_multichannel_wavs()

        # 步骤 5: 生成训练数据列表
        self._generate_train_list()

        # 步骤 6: 合并噪声数据
        self._merge_noise_wavs()

        # 步骤 7: 生成标注数据
        self._get_label_data()

        # 步骤 8: 生成多通道评估数据列表
        self._get_multichannel_eval_data_list()

    def _get_skp(self, file_path):
        # 读取TextGrid文件内容并提取 name 值
        with open(file_path, 'r', encoding='utf-8') as f:
            textgrid_content = f.read()

        pattern = r'name = "(.*?)"'
        matches = re.findall(pattern, textgrid_content)
        return matches[0] if matches else None

    def _merge_wav_channels(self, wav_files, output_file):
        # 合并单通道的 wav 文件为多通道的 wav 文件
        channels = []
        sample_rate = None
        for wav_file in wav_files:
            sr, data = wavfile.read(wav_file)
            if sample_rate is None:
                sample_rate = sr
            else:
                assert sample_rate == sr, "All WAV files must have the same sample rate."

            if len(data.shape) != 1:
                raise ValueError(f"The file {wav_file} is not a single channel WAV file.")

            channels.append(data)

        merged_data = np.stack(channels, axis=1)
        wavfile.write(output_file, sample_rate, merged_data)

    def _split_wav_by_timestamp(self, wav_file, timestamp_info, output_dir):
        # 按时间戳切分音频文件
        sample_rate, audio_data = wavfile.read(wav_file)
        os.makedirs(output_dir, exist_ok=True)

        for line in timestamp_info:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            start_time_ms = int(parts[3])
            end_time_ms = int(parts[4])

            start_sample = int(start_time_ms * sample_rate / 1000)
            end_sample = int(end_time_ms * sample_rate / 1000)

            segment = audio_data[start_sample:end_sample]

            formatted_start_time = str(start_time_ms).zfill(6)
            formatted_end_time = str(end_time_ms).zfill(6)
            output_file = os.path.join(output_dir, f"{parts[0]}_{parts[1]}_{formatted_start_time}_{formatted_end_time}.wav")
            wavfile.write(output_file, sample_rate, segment)

    def _get_spk_zone_id(self):
        # 获取说话人和区域信息
        root_dir = self.aishell_data_root
        set_types = self.icmc_set_types
        txt_outdir = self.output_dirs["spk_zone_id_dir"]

        for set_type in set_types:
            txt_dir = os.path.join(txt_outdir, set_type)
            os.makedirs(txt_dir, exist_ok=True)
            txt_path = os.path.join(txt_dir, 'spk_zone_id.txt')
            with open(txt_path, 'w') as wf:
                set_dir = os.path.join(root_dir, set_type)
                for sub_dir in os.listdir(set_dir):
                    sub_dir_path = os.path.join(set_dir, sub_dir)
                    wav_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.TextGrid') and f.startswith('DA')]
                    for file in tqdm(wav_files, desc=f'Processing {set_type}/{sub_dir}', ncols=100):
                        textgrid_path = os.path.join(sub_dir_path, file)
                        spk = self._get_skp(textgrid_path)
                        zone_id = file.split('.')[0].split('0')[1]
                        channel_id = file.split('.')[0]
                        wf.write(f'{sub_dir} {channel_id} {spk} {zone_id}\n')

    def _merge_single_channel_wavs(self):
        # 合并单通道 WAV 文件为多通道
        multi_wav_outdir = self.output_dirs["merged_multichannel_wav_dir"]
        os.makedirs(multi_wav_outdir, exist_ok=True)
        wav_type = ['DX01C01.wav', 'DX02C01.wav', 'DX03C01.wav', 'DX04C01.wav']

        for set_type in self.icmc_set_types:
            set_dir = os.path.join(self.aishell_data_root, set_type)
            for sub_dir in os.listdir(set_dir):
                sub_dir_path = os.path.join(set_dir, sub_dir)
                wav_files = [
                    os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path)
                    if f in wav_type
                ]
                wav_files.sort()
                output_file = os.path.join(multi_wav_outdir, f'{set_type}_{sub_dir}.wav')
                try:
                    self._merge_wav_channels(wav_files, output_file)
                except Exception as e:
                    print(f"Error merging WAVs for {set_type}/{sub_dir}: {e}")

    def _get_split_info(self):
        # 获取音频片段分割信息
        root_dir = self.enhanced_data_root
        set_types = self.icmc_set_types
        txt_outdir = self.output_dirs["split_info_dir"]

        for set_type in set_types:
            set_dir = os.path.join(root_dir, set_type)
            txt_path = os.path.join(txt_outdir, set_type, 'split_info.txt')
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, 'w') as wf:
                sub_dirs = [os.path.join(set_dir, sub_dir) for sub_dir in os.listdir(set_dir)]
                for sub_dir in tqdm(sub_dirs, desc=f"Processing {set_type}", unit="sub_dir"):
                    for file in os.listdir(sub_dir):
                        if os.path.isdir(os.path.join(sub_dir, file)):
                            splited_wavs_path = os.path.join(sub_dir, file)
                            for file_name in os.listdir(splited_wavs_path):
                                body = file_name.split('.')[0]
                                parts = body.split('_')
                                if len(parts) != 4:
                                    continue
                                spk, num, zone_id, time_delay = parts
                                start, end = time_delay.split('-')
                                zone_id = zone_id.split('0')[-1]
                                id = sub_dir.split('/')[-1]
                                wf.write(f'{id} {spk} {zone_id} {start} {end}\n')

    def _split_multichannel_wavs(self):
        # 截取多通道音频片段
        root_dir = self.output_dirs["merged_multichannel_wav_dir"]
        txt_dir = self.output_dirs["split_info_dir"]
        out_dir = self.output_dirs["split_wav_dir"]
        os.makedirs(out_dir, exist_ok=True)

        for wav in os.listdir(root_dir):
            wav_path = os.path.join(root_dir, wav)
            wav_name = os.path.splitext(wav)[0]
            type_name, id = wav_name.rsplit('_', 1)
            info = []
            txt_path = os.path.join(txt_dir, type_name, 'split_info.txt')
            out_subdir = os.path.join(out_dir, type_name, id)
            os.makedirs(out_subdir, exist_ok=True)
            if os.listdir(out_subdir):
                print(f"Skipping {out_subdir}: Directory already exists.")
                continue

            with open(txt_path, 'r') as rf:
                for line in rf:
                    info_id, spk, zone_id, start, end = line.strip().split()
                    if info_id == id:
                        info.append(f"{start} {end}")

            if info:
                try:
                    info_with_timestamps = [f"{id} {spk} {zone_id} {start} {end}" for start, end in [line.split() for line in info]]
                    self._split_wav_by_timestamp(wav_path, info_with_timestamps, out_subdir)
                except Exception as e:
                    print(f"Error splitting WAV {wav}: {e}")

    def _generate_train_list(self):
        # 生成训练数据列表
        root_dir = self.output_dirs["split_wav_dir"]
        txt_dir = self.output_dirs["spk_zone_id_dir"]

        for set_type in self.icmc_set_types:
            spk_zone_id_path = os.path.join(txt_dir, set_type, 'spk_zone_id.txt')
            spk_zone_id_dict = {}
            with open(spk_zone_id_path, 'r') as rf:
                for line in rf:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    id, spk, zone_id = parts[:3]
                    spk_zone_id_dict[spk] = zone_id

            lst_path = os.path.join(txt_dir, set_type, f'{set_type}.lst')
            set_dir = os.path.join(root_dir, set_type)
            with open(lst_path, 'w') as wf:
                for sub_dir in os.listdir(set_dir):
                    sub_dir_path = os.path.join(set_dir, sub_dir)
                    for file in os.listdir(sub_dir_path):
                        file_name = os.path.splitext(file)[0]
                        parts = file_name.split('_')
                        if len(parts) < 4:
                            continue
                        spk = parts[1]
                        zone_id = spk_zone_id_dict.get(spk, '')
                        file_path = os.path.join(sub_dir_path, file)
                        wf.write(f'{file_path} {spk} {zone_id}\n')

    def _merge_noise_wavs(self):
        # 合并噪声数据
        noise_types = ['noise_DX01C01.wav', 'noise_DX02C01.wav', 'noise_DX03C01.wav', 'noise_DX04C01.wav']
        noise_list_path = self.output_dirs["noise_list_path"]

        with open(noise_list_path, 'w') as wf:
            for sub_dir in os.listdir(self.noise_data_root):
                sub_dir_path = os.path.join(self.noise_data_root, sub_dir)
                wav_files = [os.path.join(sub_dir_path, nt) for nt in noise_types]
                output_file = os.path.join(sub_dir_path, 'noise.wav')
                try:
                    self._merge_wav_channels(wav_files, output_file)
                except Exception as e:
                    print(f"Error merging noise WAVs for {sub_dir}: {e}")
                wf.write(f'{output_file}\n')

    def _get_label_data(self):
        # 获取标注数据
        root_dir = "/home/38_data1/yhdai/data_raw/ICMC-split-near"
        label_data_path = self.output_dirs["label_data_path"]
        set_types = self.icmc_set_types

        with open(label_data_path, 'w') as wf:
            for set_type in set_types:
                set_dir = os.path.join(root_dir, set_type)
                for sub_dir in os.listdir(set_dir):
                    sub_dir_path = os.path.join(set_dir, sub_dir)
                    for channel_dir in os.listdir(sub_dir_path):
                        channel_dir_path = os.path.join(sub_dir_path, channel_dir)
                        for file in os.listdir(channel_dir_path):
                            if file.endswith('.wav'):
                                wav_path = os.path.join(channel_dir_path, file)
                                parts = os.path.splitext(file)[0].split('_')
                                if len(parts) < 4:
                                    continue
                                spk, sub_dir_id, channel_id, timestamp = parts
                                timestamp = timestamp.replace('-', '_')
                                id = f'{sub_dir_id}_{spk}_{timestamp}'
                                wf.write(f'{id} {wav_path}\n')

    def _get_multichannel_eval_data_list(self):
        # 生成多通道评估数据列表
        root_dir = self.output_dirs["merged_multichannel_wav_dir"]
        output_dir = self.output_dirs["multichannel_eval_data_list_dir"]

        for set_type in self.icmc_set_types:
            out_lst_path = os.path.join(output_dir, set_type, f'wav.lst')
            os.makedirs(os.path.dirname(out_lst_path), exist_ok=True)
            with open(out_lst_path, 'w') as wf:
                for file in os.listdir(root_dir):
                    if file.startswith(set_type):
                        file_path = os.path.join(root_dir, file)
                        wf.write(f'{file_path}\n')

if __name__ == "__main__":
    # 配置输入输出路径
    args = get_parser_args()
    aishell5_data_root = args.aishell5_data_root
    enhanced_aishell5_data_root = args.enchanced_aishell5_data_root
    aishell5_set_types = args.aishell5_set_types
    output_dirs = {
        "spk_zone_id_dir": args.spk_zone_id_dir,
        "merged_multichannel_wav_dir": args.merged_multichannel_wav_dir,
        "split_info_dir": args.split_info_dir,
        "split_wav_dir": args.split_wav_dir,
        "noise_list_path": args.noise_list_path,
        "label_data_path": args.label_data_path,
        "multichannel_eval_data_list_dir": args.multichannel_eval_data_list_dir
    }

    noise_data_root = args.noise_data_root  # 替换为你的噪声数据集根目录


    icmc_set_types = ['train', 'dev', 'eval_track1', 'eval_track2']
    processor = AudioProcessor(aishell5_data_root, icmc_set_types, enhanced_aishell5_data_root, noise_data_root, output_dirs)
    processor.process_data()
