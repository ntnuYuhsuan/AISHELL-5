## TextGrid 資料清洗與分析（preprocess）

本資料夾提供針對 AISHELL-5 原始 TextGrid 標記與對應音檔的彙整、去重與基本分析。

### 目錄說明

- `process_aishell5.py`: 掃描資料集中的 `.TextGrid` 與對應 `.wav`，彙整為單一 CSV
- `aishell5_cleaner.py`: 以 `text` 去重、彙整多版本音檔，並進行簡單意圖分類與統計
- 產出檔：
  - `aishell5_full_data.csv`: 原始彙整（每筆對應一組 `.TextGrid`/`.wav`）
  - `aishell5_grouped_by_text_data.csv`: 以 `text` 去重並彙整 `audio_paths`
  - `aishell5_intent_analysis.csv`: 於去重結果上加入分類與統計欄位

### 先決條件

- Python 3.9+
- 安裝依賴（專案根 `requirements.txt` 已含 `pandas`）：

```Shell
pip install -r requirements.txt
```

### Step 1：由 TextGrid 產生完整彙整 CSV

`process_aishell5.py` 會：
- 解析每個 TextGrid 的語段，串接文字、累積語音時長
- 產出欄位：`audio_path`, `text`, `total_speech_duration`, `character_count`

使用方式：

```Shell
cd AISHELL-5/preprocess

# 直接執行（請先在檔案內調整 base_aishell5_path 與子資料夾名稱）
python3 process_aishell5.py

# 產出：
# ./aishell5_full_data.csv
```

注意：目前腳本以檔內常數指定資料根路徑與子資料夾清單，若你的資料夾命名為 `dev`、`eval_track1`、`eval_track2`，請對應調整 `corpus_paths_to_process` 清單，確保實際存在。

### Step 2：去重與意圖分析

`aishell5_cleaner.py` 會：
- 以 `text` 分組，彙整同文句的 `audio_paths`（多版本音檔）
- 計算 `char_per_second`、`length_quartile`，並透過關鍵詞標記 `content_category`

使用方式：

```Shell
cd AISHELL-5/preprocess

# 讀取 Step 1 產生的 CSV，輸出去重與分析檔
python3 aishell5_cleaner.py

# 產出：
# ./aishell5_grouped_by_text_data.csv
# ./aishell5_intent_analysis.csv
```

### 欄位說明（重點）

- `audio_path`: `.wav` 絕對路徑
- `text`: 由 TextGrid 語段串接的完整文字
- `total_speech_duration`: 語音段總時長（秒）
- `character_count`: 文字字元數
- `audio_paths`:（僅於去重結果）對應同一 `text` 的所有音檔清單
- `num_audio_versions`: 同文句音檔版本數
- `char_per_second`: 字元密度（字元/秒）
- `content_category`: 關鍵詞分類標記（車載相關/日常對話/教育話題/其他）
- `length_quartile`: 字數分位（短/中短/中長/長）

### 常見注意事項

- 請確認 TextGrid 搭配的 `.wav` 檔案存在，否則會被跳過
- 簡繁轉換僅做非常輕量化的示例，若需嚴謹的跨語系匹配，建議導入專用轉換套件
- 若資料量極大，建議先以子集驗證流程與輸出格式


