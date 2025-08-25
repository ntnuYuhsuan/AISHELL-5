### AISHELL-5 前處理與文字清洗復現指南（Stage 0 + TextGrid 清洗）

本文件說明如何：
- 以 Eval1 的 `run_aishell5.sh` 僅執行 Stage 0 完成前處理（含 AEC + IVA），產出已增強與切分後的音檔與對應的資料清單。
- 使用 `preprocess/` 目錄下的腳本，針對 TextGrid 標記資料進行清洗與統計，輸出 CSV 檔以利後續分析（含意圖偵測前置統計）。

本指南僅涉及可復現流程與必要的中介結果；請勿將原始音檔與龐大中間檔上傳至 Git。

---

## 1) 前置環境

- 系統：Linux（本專案測試於 Ubuntu 22.04）
- 建議以 Conda/Python3.8+ 環境執行 Wenet 相關工具
- 已取得 AISHELL-5 原始語料（未隨倉庫提供）

---

## 2) Eval1 資料前處理（Stage 0：AEC + IVA）

Eval1 的前處理由 `Eval1/run_aishell5.sh` 驅動，內部會呼叫 `local/icmcasr_data_prep.sh`，於 Stage 0 完成 AEC + IVA 增強，以及後續的切分與資料檔案準備（由 `segment_wavs.py` 與 `data_prep.py` 處理）。

### 2.1 路徑變數與輸出位置

請確認以下變數（可直接在 `Eval1/run_aishell5.sh` 內設定或以參數覆寫）：
- `data`：AISHELL-5 語料根目錄（建議填入絕對路徑）
- `data_enhanced`：AEC + IVA 後之增強音檔輸出根目錄，README 中假設為 `data/aishell5_enhanced`

本倉庫的預設片段（僅示意）：
```bash
data=/share/nas169/andyfang/AISHELL-5
data_enhanced=/share/nas169/andyfang/AISHELL-5/data/aishell5_enhanced
```

常見資料集代號：
- `train_aec_iva_near`
- `dev_aec_iva`
- `eval_track1_aec_iva`（或 `eval_track2`）

### 2.2 只執行 Stage 0

`run_aishell5.sh` 支援以 `--stage` 與 `--stop_stage` 指定要執行的範圍。只執行 Stage 0：
```bash
cd AISHELL-5/Eval1
bash run_aishell5.sh --stage 0 --stop_stage 0
```

流程摘要：
- 於 Stage 0 內部呼叫 `local/icmcasr_data_prep.sh`，進一步觸發：
  - `local/enhancement.sh`：進行 AEC + IVA 增強，將結果寫入 `data_enhanced` 中對應子目錄
  - `local/segment_wavs.py`：依標記/規則切分語音
  - `local/data_prep.py`：生成 `data/<dataset>/` 內必要的 `wav.scp`、`text`、`utt2spk` 等

執行完成後，重點輸出包含：
- `data/aishell5_enhanced/`：存放 AEC + IVA 過後之增強音檔
- `data/<dataset>/`：對應資料集的 Kaldi/Wenet 格式清單（例如 `data/train_aec_iva_near/`）

若要同時處理多個資料集，可在 `run_aishell5.sh` 中調整迴圈或參數（預設示例通常包含 `train/dev/eval`）。

---

## 3) TextGrid 標記資料清洗與分析（`preprocess/`）

`preprocess/` 目錄提供兩支腳本：
- `process_aishell5.py`：遍歷語料樹，讀取所有 `.TextGrid` 與對應 `.wav`，合併同檔內各 interval 的文字為一筆紀錄，彙整為完整 CSV。
- `aishell5_cleaner.py`：
  - 以 `text` 為鍵進行內容去重與音檔路徑聚合，輸出群組化 CSV。
  - 產出意圖偵測前置分析欄位（如字元密度、關鍵詞分類、長度分位數）並另存分析 CSV。

### 3.1 路徑與輸入輸出

請先確認以下兩支腳本的路徑變數：
- `process_aishell5.py` 內部 `__main__` 預設：
  ```python
  base_aishell5_path = "/share/nas169/andyfang/aishell5"
  corpus_paths_to_process = [
      os.path.join(base_aishell5_path, "train"),
      os.path.join(base_aishell5_path, "Dev"),
      os.path.join(base_aishell5_path, "Eval1"),
      os.path.join(base_aishell5_path, "Eval2"),
  ]
  ```
  請依實際語料路徑調整（注意大小寫差異，例如 `AISHELL-5` vs `aishell5`）。

- `aishell5_cleaner.py` 預期輸入輸出（預設皆放於 `AISHELL-5/preprocess/`）：
  - 輸入：`aishell5_full_data.csv`（由 `process_aishell5.py` 產生）
  - 輸出一：`aishell5_grouped_by_text_data.csv`（以 `text` 聚合、附帶多版本音檔清單）
  - 輸出二：`aishell5_intent_analysis.csv`（加入關鍵詞類別、字元密度、長度分位數）

### 3.2 執行步驟

建議於 `AISHELL-5/preprocess` 目錄內執行以下指令：
```bash
cd AISHELL-5/preprocess
python3 process_aishell5.py

# 完成後，將在當前目錄生成：aishell5_full_data.csv

python3 aishell5_cleaner.py

# 完成後，將在當前目錄生成：
# - aishell5_grouped_by_text_data.csv
# - aishell5_intent_analysis.csv
```

輸出 CSV 欄位摘要：
- `aishell5_full_data.csv`
  - `audio_path`：對應 `.wav` 的絕對路徑
  - `text`：同檔案內所有 interval 文字合併後的完整轉寫
  - `total_speech_duration`：合計發話時長（秒）
  - `character_count`：`text` 的字元數

- `aishell5_grouped_by_text_data.csv`
  - `text`（唯一鍵）
  - `audio_paths`：相同文字對應的所有音檔路徑列表
  - `total_speech_duration`、`character_count`（擇一或首筆代表值）
  - `num_audio_versions`：同文字對應音檔版本數

- `aishell5_intent_analysis.csv`
  - 繼承 `grouped` 版本欄位
  - `char_per_second`、`content_category`（關鍵詞類別）、`length_quartile`（長度分位數）

---

## 4) 常見問題與注意事項

- 請務必使用絕對路徑設定 `data` 與 `data_enhanced`，避免工作目錄造成的相對路徑解析問題。
- 若只需前處理，不需訓練/解碼，僅執行 Stage 0 即可。
- `process_aishell5.py` 預設遍歷 `train/Dev/Eval1/Eval2`，若實際語料目錄命名不同，請調整。
- 簡繁轉換在 `aishell5_cleaner.py` 僅作示意轉換，如需嚴謹處理，建議導入專門庫（例如 OpenCC）。

---

## 5) 版本控管建議

- 請僅將本 README 與 `preprocess/` 目錄提交至版本控制；避免提交原始音檔或龐大的中介結果至 Git。
- 如需長期保存大量資料，建議使用物件儲存或資料湖（例如 S3、GCS）並以路徑引用方式整合。


