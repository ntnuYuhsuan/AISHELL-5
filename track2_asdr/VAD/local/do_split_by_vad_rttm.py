import argparse
import os
from pydub import AudioSegment
from tqdm import tqdm  # 导入 tqdm 库


def cut_audio(input_audio_path, start, duration, output_audio_path):
    """
    从原始音频中切割指定时间段的音频片段
    :param input_audio_path: 原始音频路径
    :param start: 起始时间（秒）
    :param duration: 持续时间（秒）
    :param output_audio_path: 输出音频片段路径
    """
    # 使用pydub来加载和切割音频
    audio = AudioSegment.from_wav(input_audio_path)
    start_ms = start * 1000  # 转换为毫秒
    duration_ms = duration * 1000  # 转换为毫秒
    end_ms = start_ms + duration_ms
    segment = audio[start_ms:end_ms]
    segment = segment.set_channels(1)  # 立体声转化为单通道声音
    segment.export(output_audio_path, format="wav")


def convert_stereo_to_mono(input_audio_path, output_audio_path):
    """
    将2通道（立体声）音频转换为单通道（单声道）音频。
    :param input_audio_path: 输入音频路径（应为立体声）
    :param output_audio_path: 输出音频路径（保存单声道音频）
    """
    # 使用pydub加载音频文件
    audio = AudioSegment.from_wav(input_audio_path)
    
    # 转换为单通道（mono）
    mono_audio = audio.set_channels(1)

    # 保存为单声道音频
    mono_audio.export(output_audio_path, format="wav")



def main(args):
    input_audio_scp = args.input_audio_scp
    rttm_file_dir = args.rttm_file_dir
    output_dir = args.output_dir
    wav_dict = {}

    # 读取 wav.scp 文件并保存映射关系
    with open(input_audio_scp, 'r') as rf:
        for line in rf:
            key, input_audio_path = line.strip().split(' ')
            wav_dict[key] = input_audio_path

    # 遍历每个音频文件
    for key, input_audio_path in wav_dict.items():
        # 创建该音频文件对应的输出目录
        split_audio_root_dir = os.path.join(output_dir, key)
        if not os.path.exists(split_audio_root_dir):
            os.makedirs(split_audio_root_dir)

        split_audio_outdir = os.path.join(split_audio_root_dir, 'wavs')
        if not os.path.exists(split_audio_outdir):
            os.makedirs(split_audio_outdir)


        # 读取 RTTM 文件
        rttm_file = os.path.join(rttm_file_dir, f"{key}.rttm")
        with open(rttm_file, 'r') as f:
            lines = f.readlines()

        # 生成 wav.scp 文件内容
        wav_scp = []

        # 使用 tqdm 为 RTTM 文件中的每一行添加进度条
        for line in tqdm(lines, desc=f"Processing {key}", unit="segment"):
            # 解析 RTTM 格式的每一行
            fields = line.strip().split()
            session_id = fields[1]  # 会话ID
            start = float(fields[3])  # 起始时间
            duration = float(fields[4])  # 持续时间

            # 为每个切割后的音频片段生成唯一的文件名
            segment_filename = f"{session_id}_{round(start, 2)}_{round(duration, 2)}.wav"
            segment_filepath = os.path.join(split_audio_outdir, segment_filename)

            # 切割音频
            cut_audio(input_audio_path, start, duration, segment_filepath)

            # 将每个片段的 key 和路径写入 wav.scp
            wav_scp.append(f"{session_id}_{round(start, 2)} {segment_filepath}")

        # 保存 wav.scp 文件
        scp_path = os.path.join(split_audio_root_dir, 'wav.scp')
        with open(scp_path, 'w') as f:
            for line in wav_scp:
                f.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio_scp", type=str, required=True, help="原始音频文件路径")
    parser.add_argument("--rttm_file_dir", type=str, required=True, help="RTTM 文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="切割后的音频片段保存目录")
    
    args = parser.parse_args()
    main(args)
