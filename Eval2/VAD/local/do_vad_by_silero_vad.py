import os
import argparse
from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


def main(args):
    if os.path.exists(args.save_path):
        os.system(f"rm -rf {args.save_path}")
    os.makedirs(args.save_path)

    wav_scp = {}
    with open(args.wav_scp, "r") as f:
        for line in f.readlines():
            utt, path = line.strip().split()
            wav_scp[utt] = path

    # Load Silero VAD model
    vad_model = load_silero_vad()

    for utt, path in tqdm(wav_scp.items()):
        # Read audio and get speech timestamps
        wav = read_audio(path)
        speech_timestamps = get_speech_timestamps(wav, vad_model, return_seconds=True)
        
        # Write speech segments to RTTM file
        with open(os.path.join(args.save_path, f"{utt}.rttm"), "w") as rttm:
            for segment in speech_timestamps:
                start = segment['start']
                end = segment['end']
                # Write segment information in RTTM format (example format)
                rttm.write(f"SPEAKER {utt} 1 {start:.3f} {end-start:.3f} <NA> <NA> <NA> <NA>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_scp', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    main(parser.parse_args())
