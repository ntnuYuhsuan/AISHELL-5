
import os
import re
from tqdm import tqdm  # 导入 tqdm
import numpy as np
from scipy.io import wavfile


def get_skp(file_path):
    # 假设文件名为 DA02.TextGrid
    # 这里已经在函数中定义了 file_path，实际上不需要覆盖文件路径
    # file_path = 'DA02.TextGrid'

    # 读取TextGrid文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        textgrid_content = f.read()

    # 使用正则表达式提取 name 对应的值
    pattern = r'name = "(.*?)"'

    matches = re.findall(pattern, textgrid_content)

    # 输出所有匹配到的 name 值
    if matches:
        print("匹配到的 name 值:", matches)
    else:
        print('没有匹配到 name 值。')

    return matches[0]



def merge_wav_channels(wav_files, output_file):
    # 读取四个单通道的 wav 文件
    channels = []
    for wav_file in wav_files:
        # 读取 .wav 文件
        sample_rate, data = wavfile.read(wav_file)
        
        # 确保每个文件都是单通道的
        if len(data.shape) != 1:
            raise ValueError(f"The file {wav_file} is not a single channel wav.")
        
        # 将数据存储在列表中
        channels.append(data)
    
    # 合并为四通道数据
    # 假设所有文件的采样率和长度相同
    # 通过 np.stack 将它们堆叠成一个 (N, 4) 的数组，其中 N 是样本数
    merged_data = np.stack(channels, axis=1)  # 堆叠在第二个维度（通道维度）
    
    # 保存为四通道的 wav 文件
    wavfile.write(output_file, sample_rate, merged_data)
    print(f"Saved merged WAV to {output_file}")

def split_wav_by_timestamp(wav_file, timestamp_info, output_dir):
    """
    按照 timestamp_file 中提供的时间戳对 wav 文件进行切分并保存到 output_dir。

    :param wav_file: 输入的完整音频 wav 文件路径
    :param timestamp_info: 含有时间戳的列表，每行格式为: id speaker zone_id start_time end_time（时间单位：毫秒）
    :param output_dir: 输出目录，保存切分后的音频片段
    """
    # 读取音频文件
    sample_rate, audio_data = wavfile.read(wav_file)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for line in timestamp_info:
        # 每行格式: id speaker zone_id start_time end_time
        parts = line.strip().split()
        file_id = parts[0]  # 音频文件id
        spk = parts[1]
        start_time_ms = int(parts[3])  # 开始时间（毫秒）
        end_time_ms = int(parts[4])  # 结束时间（毫秒）

        # 将时间（毫秒）转换为采样点索引
        start_sample = int(start_time_ms * sample_rate / 1000)
        end_sample = int(end_time_ms * sample_rate / 1000)

        # 提取对应的音频片段
        segment = audio_data[start_sample:end_sample]

        # 将开始时间和结束时间转换为六位字符串，不足六位前面补0
        formatted_start_time = str(start_time_ms).zfill(6)
        formatted_end_time = str(end_time_ms).zfill(6)

        # 创建输出文件名
        output_file = os.path.join(output_dir, f"{file_id}_{spk}_{formatted_start_time}_{formatted_end_time}.wav")

        # 保存切分后的音频片段
        wavfile.write(output_file, sample_rate, segment)
        print(f"Saved: {output_file}")

def merge_noise_wav_channels(wav_files, output_file):
    # 读取四个单通道的 wav 文件
    channels = []
    for wav_file in wav_files:
        # 读取 .wav 文件
        sample_rate, data = wavfile.read(wav_file)
        
        # 确保每个文件都是单通道的
        if len(data.shape) != 1:
            raise ValueError(f"The file {wav_file} is not a single channel wav.")
        
        # 将数据存储在列表中
        channels.append(data)
    
    # 合并为四通道数据
    # 假设所有文件的采样率和长度相同
    # 通过 np.stack 将它们堆叠成一个 (N, 4) 的数组，其中 N 是样本数
    merged_data = np.stack(channels, axis=1)  # 堆叠在第二个维度（通道维度）
    
    # 保存为四通道的 wav 文件
    wavfile.write(output_file, sample_rate, merged_data)
    print(f"Saved merged WAV to {output_file}")



switch = 8
set_types = ['train', 'dev', 'eval_track1', 'eval_track2']

if switch == 1:
    # step 1 : get spk and zone_id
    root_dir = "/home/38_data1/yhdai/data_raw/ICMC-ASR"
    set_types = ['train', 'dev', 'eval_track1', 'eval_track2']
    txt_outdir = "../configs/data/data_icmc/label_data"
    os.makedirs(txt_outdir, exist_ok=True)
    # 使用 tqdm 显示进度条
    for set_type in set_types:
        txt_dir = os.path.join(txt_outdir, set_type)
        os.makedirs(txt_dir, exist_ok=True)
        txt_path = os.path.join(txt_dir, 'spk_zone_id.txt')
        with open(txt_path, 'w') as wf:
            set_dir = os.path.join(root_dir, set_type)
            for sub_dir in os.listdir(set_dir):
                sub_dir_path = os.path.join(set_dir, sub_dir)
                wav_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.TextGrid') and f.startswith('DA')]
                
                # 通过 tqdm 包装迭代器，显示进度条
                for file in tqdm(wav_files, desc=f'Processing {set_type}/{sub_dir}', ncols=100):
                    textgrid_path = os.path.join(sub_dir_path, file)
                    spk = get_skp(textgrid_path)
                    zone_id = file.split('.')[0].split('0')[1]
                    channel_id = file.split('.')[0]
                    wf.write(f'{sub_dir} {channel_id} {spk} {zone_id}\n')


if switch == 2:
    # step 2 : get multi_channel wav 
    multi_wav_outdir = "/home/38_data1/yhdai/data_raw/ICMC-ASR/fe_expdata/fe_expdata_original_wav"
    os.makedirs(multi_wav_outdir, exist_ok=True)
    wav_type = ['DX01C01.wav', 'DX02C01.wav', 'DX03C01.wav', 'DX04C01.wav']
    root_dir = "/home/38_data1/yhdai/data_raw/ICMC-ASR"
    for set_type in set_types:
        set_dir = os.path.join(root_dir, set_type)
        for sub_dir in os.listdir(set_dir):
            sub_dir_path = os.path.join(set_dir, sub_dir)
            wav_files = [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) if f in wav_type]
            
            # 对wav_files进行排序
            wav_files.sort()  # 按文件名排序，排序规则是文件名中的数字部分
            
            # 生成输出文件路径
            output_file = os.path.join(multi_wav_outdir, f'{set_type}_{sub_dir}.wav')
            
            # 调用merge_wav_channels并显示进度条
            merge_wav_channels(wav_files, output_file)


if switch == 3:
    # step 3 : get split_info.txt
    root_dir = "/home/38_data1/yhdai/data_raw/ICMC-ASR_ENHANCED"
    set_types = ['train', 'dev', 'eval_track1', 'eval_track2']
    txt_outdir = "/home/38_data1/yhdai/workspace/front_end/spatialnet/NBSS/tools_yhdai"
    
    for set_type in set_types:
        set_dir = os.path.join(root_dir, set_type)
        txt_path = os.path.join(txt_outdir, set_type)
        os.makedirs(txt_path, exist_ok=True)
        txt_file = os.path.join(txt_path, 'split_info.txt')
        
        with open(txt_file, 'w') as wf:
            # 获取所有子目录路径，并用 tqdm 包装
            sub_dirs = [os.path.join(set_dir, sub_dir) for sub_dir in os.listdir(set_dir)]
            for sub_dir in tqdm(sub_dirs, desc=f"Processing {set_type}", unit="sub_dir"):
                sub_dir_path = sub_dir
                for file in os.listdir(sub_dir_path):
                    # print(f"file: {file}")
                    if os.path.isdir(os.path.join(sub_dir_path, file)):
                        splited_wavs_path = os.path.join(sub_dir_path, file)
                        # 获取所有要处理的文件，并用 tqdm 包装
                        file_names = [file_name for file_name in os.listdir(splited_wavs_path)]
                        for file_name in tqdm(file_names, desc=f"Processing {file}", unit="file", leave=False):
                            body = file_name.split('.')[0]
                            spk, num, zone_id, time_delay = body.split('_')
                            start, end = time_delay.split('-')
                            zone_id = zone_id.split('0')[-1]
                            id = sub_dir.split('/')[-1]
                            # 输出到txt文件: 编号 文件名 说话人标签 座位编号 开始时间 结束时间
                            wf.write(f'{id} {spk} {zone_id} {start} {end}\n')

if switch == 4:
    # step 4 : get multi_channel splitted wav
    root_dir = "/home/38_data1/yhdai/data_raw/ICMC-ASR/fe_expdata/fe_expdata_original_wav"
    os.makedirs(root_dir, exist_ok=True)
    set_types = ['train', 'dev', 'eval_track1', 'eval_track2']
    txt_dir = "../configs/data/data_icmc"
    out_dir = "/home/38_data1/yhdai/data_raw/ICMC-ASR/fe_expdata/split_wav"
    os.makedirs(out_dir, exist_ok=True)
    for wav in os.listdir(root_dir):
        wav_name = wav.split('.')[0]
        type_name, id = wav_name.rsplit('_', 1)
        info = []
        txt_path = os.path.join(txt_dir, type_name, 'split_info.txt')
        # split_wav_by_timestamp函数参数
        wav_path = os.path.join(root_dir, wav)
        out_subdir = os.path.join(out_dir, type_name, id)
        os.makedirs(out_subdir, exist_ok=True)
        if len(os.listdir(out_subdir)) != 0:
            print(f"Skipping {out_subdir}: Directory already exists.")
            continue
        os.makedirs(out_subdir, exist_ok=True)
        with open(txt_path, 'r') as rf:
            for line in rf:
                info_id, spk, zone_id, start, end = line.strip().split()
                if info_id == id:
                    info.append(line)
        # 切分音频
        split_wav_by_timestamp(wav_path, info, out_subdir)
        print(f"Splitting {wav} completed.")

if switch == 5:
    # 生成train.lst
    root_dir = "/home/38_data1/yhdai/data_raw/ICMC-ASR/fe_expdata/split_wav"
    set_types = ['train', 'dev', 'eval_track1', 'eval_track2']
    set_types = ['eval_track1', 'eval_track2']
    txt_dir = "/home/38_data1/yhdai/workspace/front_end/spatialnet/NBSS/configs/data/data_icmc"
    for set_type in set_types:
        spk_zone_id_dict = {}
        spk_zone_id_path = os.path.join(txt_dir, set_type, 'spk_zone_id.txt')
        # 获取spk_zone_id.txt
        with open(spk_zone_id_path, 'r') as rf:
            for line in rf:
                id, spk, zone_id = line.strip().split(' ')
                spk_zone_id_dict[spk] = (zone_id)
        # 生成train.lst
        set_dir = os.path.join(root_dir, set_type)
        lst_path = os.path.join(txt_dir, set_type, f'{set_type}.lst')
        with open(lst_path, 'w') as wf:
            for sub_dir in os.listdir(set_dir):
                sub_dir_path = os.path.join(set_dir, sub_dir)
                for file in os.listdir(sub_dir_path):
                    file_name = file.split('.')[0]
                    id, spk, start, end = file_name.split('_')
                    zone_id = spk_zone_id_dict[spk]
                    file_path = os.path.join(sub_dir_path, file)
                    wf.write(f'{file_path} {spk} {zone_id}\n')
        print(f"Generating {set_type}.lst completed.")

if switch == 6:
    # get noise data
    root_dir = "/home/38_data1/yhdai/data_raw/car_noise"
    noise_types = ['noise_DX01C01.wav', 'noise_DX02C01.wav', 'noise_DX03C01.wav', 'noise_DX04C01.wav']
    with open('noise_list.txt', 'w') as wf:
        for sub_dir in os.listdir(root_dir):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            wav_files = []
            for noise_type in noise_types:
                noise_path = os.path.join(root_dir, sub_dir, noise_type)
                wav_files.append(noise_path)
            outfile = os.path.join(root_dir, sub_dir, f'noise.wav')
            merge_noise_wav_channels(wav_files, outfile)
            wf.write(f'{outfile}\n')
            print(f"Merging {outfile} completed.")
            
if switch == 7:
    # get label of train data
    print(f"stage07: get label of train data")
    root_dir = "/home/38_data1/yhdai/data_raw/ICMC-split-near"
    set_types = ['train', 'dev', 'eval_track1', 'eval_track2']
    set_types = ['train']
    train_list = "/home/38_data1/yhdai/workspace/front_end/spatialnet/NBSS/configs/data/data_icmc/train/speech.lst"
    label_dir = "/home/38_data1/yhdai/workspace/front_end/spatialnet/NBSS/configs/data/data_icmc/label_data"


    label_data_path = os.path.join(label_dir, 'speech_list_near.lst')
    with open(label_data_path, 'w') as wf:
        for type_name in set_types:
            set_dir = os.path.join(root_dir, type_name)
            for sub_dir in os.listdir(set_dir):
                sub_dir_path = os.path.join(set_dir, sub_dir)
                for channel_dir in os.listdir(sub_dir_path):
                    channel_dir_path = os.path.join(sub_dir_path, channel_dir)
                    for file in os.listdir(channel_dir_path):
                        if file.endswith('.wav'):
                            wav_path = os.path.join(channel_dir_path, file)
                            spk, sub_dir_id, channel_id, timestamp = file.split('.')[0].split('_')
                            # 用_替换原本的-
                            timestamp = timestamp.replace('-', '_') 
                            id = f'{sub_dir_id}_{spk}_{timestamp}'
                            wf.write(f'{id} {wav_path}\n')
    print(f"stage07: get label of train data completed.")

if switch == 8:
    # get multi-channel eval data
    print(f"get multi-channel eval data list")
    root_dir = "/home/38_data1/yhdai/data_raw/ICMC-ASR/fe_expdata/fe_expdata_original_wav"
    set_types = ['dev', 'eval_track1', 'eval_track2']
    outdir = "/home/38_data1/yhdai/workspace/front_end/spatialnet/NBSS/configs/data/data_icmc"

    for set_type in set_types:
        out_dir = os.path.join(outdir, set_type)
        os.makedirs(out_dir, exist_ok=True)
        our_lst = os.path.join(out_dir, f'wav.lst')
        with open(our_lst, 'w') as wf:
            for file in os.listdir(root_dir):
                if file.startswith(set_type):
                    file_path = os.path.join(root_dir, file)
                    wf.write(f'{file_path}\n')
        print(f"{set_type} is done")
    print(f"get multi-channel eval data list completed.")


