### AISHELL-5 TextGrid 清洗與分析（preprocess）

此目錄包含針對 AISHELL-5 的 TextGrid 標記資料清洗與統計：

- `process_aishell5.py`：遍歷語料樹，收集 `.TextGrid` 與對應 `.wav`，合併同檔案內各 interval 的文字為單筆紀錄，輸出 `aishell5_full_data.csv`。
- `aishell5_cleaner.py`：
  - 以 `text` 為鍵將內容去重並聚合多個音檔路徑，輸出 `aishell5_grouped_by_text_data.csv`。
  - 生成意圖偵測前置統計（字元密度、關鍵詞類別、長度分位數），輸出 `aishell5_intent_analysis.csv`。

執行方式（建議於本目錄內）：
```bash
python3 process_aishell5.py
python3 aishell5_cleaner.py
```

完整的 Stage 0（AEC + IVA）前處理與 TextGrid 清洗整體說明，請參見倉庫根目錄：`README_AISHELL5_Preprocess.md`。


