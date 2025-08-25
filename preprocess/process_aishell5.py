import os
import re
import pandas as pd

def parse_textgrid(filepath):
    """
    Parses a TextGrid file and extracts text intervals.
    """
    intervals = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regular expression to find interval blocks and their text
    # It looks for 'intervals [N]:' followed by xmin, xmax, and text
    pattern = re.compile(
        r'intervals \[\d+\]:\s*\n'
        r'\s*xmin = (?P<xmin>\d+\.?\d*)\s*\n'
        r'\s*xmax = (?P<xmax>\d+\.?\d*)\s*\n'
        r'\s*text = "(?P<text>[^"]*)"'
    )
    for match in pattern.finditer(content):
        intervals.append({
            'xmin': float(match.group('xmin')),
            'xmax': float(match.group('xmax')),
            'text': match.group('text')
        })
    return intervals

def process_aishell5_corpus(corpus_root_paths, output_csv_path="aishell5_full_data.csv"):
    """
    Processes the AISHELL-5 corpus to extract audio paths, concatenated text content,
    total speech duration, and character count for all specified corpus paths.
    """
    data = []
    
    for corpus_root_path in corpus_root_paths:
        print(f"Processing corpus from: {corpus_root_path}")
        for root, _, files in os.walk(corpus_root_path):
            for file in files:
                if file.endswith(".TextGrid"):
                    textgrid_path = os.path.join(root, file)
                    audio_filename = file.replace(".TextGrid", ".wav")
                    audio_path = os.path.join(root, audio_filename)

                    if os.path.exists(audio_path):
                        intervals = parse_textgrid(textgrid_path)
                        
                        full_text_parts = []
                        total_duration = 0.0
                        for interval in intervals:
                            if interval['text']:
                                full_text_parts.append(interval['text'])
                                total_duration += (interval['xmax'] - interval['xmin'])
                        
                        full_text = " ".join(full_text_parts)
                        character_count = len(full_text)
                        
                        if full_text: # Only add if there is any content after concatenation
                            data.append({
                                'audio_path': audio_path,
                                'text': full_text,
                                'total_speech_duration': total_duration,
                                'character_count': character_count
                            })
                    else:
                        print(f"Warning: Corresponding audio file not found for {textgrid_path}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data successfully saved to {output_csv_path}")

if __name__ == "__main__":
    base_aishell5_path = "/share/nas169/andyfang/aishell5"
    corpus_paths_to_process = [
        os.path.join(base_aishell5_path, "train"),
        os.path.join(base_aishell5_path, "Dev"),
        os.path.join(base_aishell5_path, "Eval1"),
        os.path.join(base_aishell5_path, "Eval2")
    ]
    process_aishell5_corpus(corpus_paths_to_process)