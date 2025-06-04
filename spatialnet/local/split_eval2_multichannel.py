import os
import librosa
import soundfile as sf
import numpy as np

def split_audio(wav_path, num_splits=10, target_sr=16000):
    """
    将四通道音频切分成指定数量的等份，并保存每一份。
    """
    # 读取音频文件
    audio_data, sr = librosa.load(wav_path, sr=target_sr, mono=False)
    
    # 确保音频是四通道
    if audio_data.shape[0] != 4:
        raise ValueError(f"音频文件 {wav_path} 不是四通道音频，当前通道数为 {audio_data.shape[0]}")

    # 获取音频的时长
    total_length = audio_data.shape[-1]
    
    # 每一份音频的时长
    split_length = total_length // num_splits
    
    # 保存切分后的音频路径
    saved_paths = []
    
    # 按照每份时长切分音频并保存
    for i in range(num_splits):
        start = i * split_length
        end = (i + 1) * split_length if i < num_splits - 1 else total_length
        
        # 切割音频
        split_audio = audio_data[:, start:end]
        
        # 生成保存路径
        split_filename = f"{os.path.splitext(os.path.basename(wav_path))[0]}_part{i+1}.wav"
        split_path = os.path.join(os.path.dirname(wav_path), split_filename)
        
        # 保存切分后的音频文件
        sf.write(split_path, split_audio.T, target_sr)
        
        # 保存文件路径
        saved_paths.append(split_path)
    
    return saved_paths


def process_wav_scp(wav_scp_path, output_txt_path, num_splits=10):
    """
    处理 wav.scp 文件，读取音频路径，进行切割，并将切割后的路径写入到输出文件。
    """
    with open(wav_scp_path, 'r') as f:
        wav_paths = [line.strip() for line in f.readlines()]
    
    all_split_paths = []
    
    for wav_path in wav_paths:
        split_paths = split_audio(wav_path, num_splits)
        all_split_paths.extend(split_paths)
    
    # 将所有切分后的音频文件路径写入到指定文件
    with open(output_txt_path, 'w') as out_f:
        for path in all_split_paths:
            out_f.write(f"{path}\n")
    
    print(f"切割后的音频路径已经保存到 {output_txt_path}。")

# 使用示例
wav_scp_path = 'path_to_your_wav.scp'  # wav.scp 的路径
output_txt_path = 'output_paths.txt'   # 输出路径文件
process_wav_scp(wav_scp_path, output_txt_path, num_splits=10)
