import os
import re
import pandas as pd
from collections import defaultdict
# import hashlib # Removed as get_text_hash is no longer central to the new logic

# Removed parse_textgrid function

# Removed extract_recording_id function

# Removed get_text_hash function

def process_text_duplicates(df_raw, output_csv_path="aishell5_grouped_by_text_data.csv"):
    """
    Processes the raw DataFrame by grouping by 'text' and collecting audio paths.
    """
    print("\n開始根據文字內容去重和合併音頻路徑...")

    # Group by 'text' and aggregate other columns
    # For audio_path, collect all paths into a list
    # For other columns, take the first value (assuming consistency for identical text)
    grouped_df = df_raw.groupby('text').agg(
        audio_paths=('audio_path', lambda x: x.tolist()),
        total_speech_duration=('total_speech_duration', 'first'),
        character_count=('character_count', 'first')
    ).reset_index()

    # Add a count of alternative paths
    grouped_df['num_audio_versions'] = grouped_df['audio_paths'].apply(len)

    # Print statistics
    print(f"\n資料處理完成！")
    print(f"原始資料筆數: {len(df_raw)}")
    print(f"按文本去重後資料筆數: {len(grouped_df)}")
    print(f"平均每個唯一文本對應的音檔版本數: {grouped_df['num_audio_versions'].mean():.2f}")

    # Save to CSV
    grouped_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n資料已儲存至: {output_csv_path}")

    return grouped_df

def analyze_for_intent_detection(df, output_analysis_path="aishell5_intent_analysis.csv"):
    """
    分析清理後的資料，為意圖偵測篩選準備
    """
    print("\n開始分析資料以進行意圖偵測篩選...")
    
    # 添加一些分析欄位
    df_analysis = df.copy()
    
    # 計算平均字元密度（字元數/時長）
    # 避免除以零
    df_analysis['char_per_second'] = df_analysis.apply(
        lambda row: row['character_count'] / row['total_speech_duration'] if row['total_speech_duration'] > 0 else 0,
        axis=1
    )
    
    # 標記可能的對話類型
    def categorize_content(text):
        """基於關鍵詞簡單分類內容類型
        根據輸入的文本，檢查是否包含預先定義的關鍵詞，
        並返回對應的內容類別。
        """
        categories = []

        # 車載相關關鍵詞（擴充且更具口語化）
        car_keywords = [
            # 導航與路況
            '导航', '导航去', '导航到', '带我去', '去哪里', '找一下', '路线', '怎么走', '路况', '堵车',
            '回家', '回公司', '去单位', '附近有什么', '最近的', '最短的', '最快的', '避开', '绕路',
            '取消导航', '结束导航', '停止导航', '下一个路口', '前方', '掉头',
            '加油站', '充电站', '停车场', '服务区',
            
            # 車輛控制
            '开窗', '关窗', '升窗', '降窗', '把窗户打开', '把窗户关上',
            '开天窗', '关天窗', '把天窗打开', '天窗关上', '天窗打开', '天窗关掉', '打开天窗',
            '开后备箱', '关后备箱', '打开后备箱', '后备箱打开', '后备箱关上', '把后备箱打开',
            '锁车门', '解锁车门', '车门上锁', '车门解锁', '锁门', '解开车门', '打开车门',
            '开雨刷', '关雨刷', '雨刷快一点', '雨刷慢一点', '雨刮器',
            '开大灯', '关大灯', '打开大灯', '大灯打开', '远光灯', '近光灯',
            '开雾灯', '关雾灯', '打开雾灯',
            '开双闪', '关双闪', '危险警示灯',
            '开空调', '关空调', '打开空调', '空调打开', '把空调打开', '把空调关上',
            '温度高一点', '温度低一点', '调高温度', '调低温度', '冷一点', '热一点',
            '风量大一点', '风量小一点', '风速',
            '开除雾', '前挡风', '后挡风', '前挡风玻璃', '后挡风玻璃',
            '开座椅加热', '关座椅加热', '座椅加热打开', '把座椅加热打开',
            '开方向盘加热', '关方向盘加热',
            '折叠后视镜', '展开后视镜', '收起后视镜', '后视镜收起来',
            '播放音乐', '放首歌', '上一首', '下一首', '切歌', '暂停', '继续播放',
            '调大音量', '调小音量', '大声点', '小声点', '静音', '调高音量',
            '打电话给', '打个电话', '拨打', '发短信给',
            '开蓝牙', '关蓝牙', '连接蓝牙', '断开蓝牙',
            '屏幕亮度', '调亮屏幕', '调暗屏幕', '屏幕亮一点', '屏幕暗一点',
            '行车记录仪', '开始录像', '拍照', '拍个照',
            '驾驶模式', '舒适模式', '运动模式', '节能模式', '雪地模式',
            '充电', '开始充电', '停止充电', '充电桩',
            '车速', '车速慢一点', '车速快一点', '限速',
        ]

        # 日常對話關鍵詞
        daily_keywords = [
            '你好', '谢谢', '不客气', '请问', '对不起', '不好意思',
            '什么', '怎么', '哪里', '多少', '几点',
            '是', '不是', '对', '不对', '好的', '明白', '好的', '收到',
            '天气', '天气预报', '几度', '下雨', '下雪', '晴天', '阴天',
            '备忘录', '提醒我', '记一下', '日历',
            '今天', '明天', '昨天', '后天'
        ]

        # 教育相關關鍵詞
        edu_keywords = [
            '学校', '高考', '考试', '学习', '上课', '毕业', '期末', '期中',
            '老师', '学生', '同学',
            '科目', '语文', '数学', '英语', '物理', '化学', '生物', '历史', '地理',
            '作业', '练习', '复习', '预习', '知识点', '查一下'
        ]

        # 簡體字轉換為繁體字，以便關鍵詞匹配
        text_t = text.replace('你','妳').replace('台','臺').replace('的','地') # 這裡只是舉例，實際應該使用更完整的簡繁轉換庫

        if any(keyword in text_t for keyword in car_keywords):
            categories.append('車載相關')
        if any(keyword in text_t for keyword in daily_keywords):
            categories.append('日常對話')
        if any(keyword in text_t for keyword in edu_keywords):
            categories.append('教育話題')

        return '|'.join(categories) if categories else '其他'
    
    df_analysis['content_category'] = df_analysis['text'].apply(categorize_content)
    
    # 計算文字長度分位數
    df_analysis['length_quartile'] = pd.qcut(df_analysis['character_count'], 
                                             q=4, 
                                             labels=['短', '中短', '中長', '長'],
                                             duplicates='drop') # 添加 duplicates='drop' 以處理唯一值過少的情況
    
    # 儲存分析結果
    df_analysis.to_csv(output_analysis_path, index=False, encoding='utf-8')
    
    print(f"\n內容類別分布:")
    category_stats = df_analysis['content_category'].value_counts()
    print(category_stats)
    
    print(f"\n長度分布:")
    length_stats = df_analysis['length_quartile'].value_counts()
    print(length_stats)
    
    return df_analysis

if __name__ == "__main__":
    # 設定路徑
    base_aishell5_path = "/share/nas169/andyfang/AISHELL-5"
    full_data_csv_path = os.path.join(base_aishell5_path, "preprocess", "aishell5_full_data.csv")
    grouped_output_csv_path = os.path.join(base_aishell5_path, "preprocess", "aishell5_grouped_by_text_data.csv") # New output file
    analysis_output_csv_path = os.path.join(base_aishell5_path, "preprocess", "aishell5_intent_analysis.csv")

    # 步驟1: 讀取 aishell5_full_data.csv
    if not os.path.exists(full_data_csv_path):
        print(f"錯誤: 未找到 {full_data_csv_path}。請先運行 process_aishell5.py 生成完整數據。")
    else:
        df_raw_data = pd.read_csv(full_data_csv_path)
        print(f"已從 {full_data_csv_path} 讀取 {len(df_raw_data)} 筆資料。")
        print(f"原始資料欄位: {df_raw_data.columns.tolist()}")

        # 步驟2: 根據文本內容去重和合併音頻路徑
        df_grouped = process_text_duplicates(
            df_raw_data,
            output_csv_path=grouped_output_csv_path
        )
        
        # 步驟3: 分析資料以準備意圖偵測
        df_analyzed = analyze_for_intent_detection(
            df_grouped, # Pass the grouped DataFrame
            output_analysis_path=analysis_output_csv_path
        )