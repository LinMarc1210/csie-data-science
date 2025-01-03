#%%
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import numpy as np
import re
import datetime
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk




#%%
########### 日期格式前處理：分組 ##############
data = pd.read_csv("titles.csv")

# 替換月份全稱為縮寫
month_map = {
    "January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr",
    "May": "May", "June": "Jun", "July": "Jul", "August": "Aug",
    "September": "Sep", "October": "Oct", "November": "Nov", "December": "Dec"
}
for full, short in month_map.items():
    data['date'] = data['date'].str.replace(full, short)



#%%
########### 第一部分：vader情感分析 ###########
def preprocess_vader(df):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for title in df['title'].values:
        vs = analyzer.polarity_scores(title)
        results.append(vs)
    results_df = pd.DataFrame(results)
    results_df = pd.concat([df, results_df], axis=1)
    return results_df
    
data = preprocess_vader(data)
print(data)

#%%
vader_df = data.groupby(['date']).agg({'neg': 'sum', 'neu': 'sum', 'pos': 'sum', 'compound': 'sum'}).reset_index()
vader_df['date'] = pd.to_datetime(vader_df['date'])
vader_df = vader_df.sort_values(by='date')
print(vader_df)


#%%
########## 第二部分：LDA找主題 ##########
# 按日期分組，將標題合併為單個文本
data['date'] = pd.to_datetime(data['date'], format='%d %b %Y')
grouped_data = data.groupby("date")["title"].apply(lambda titles: " ".join(titles)).reset_index()
grouped_data = grouped_data.sort_values(by='date', ascending=True)
print(grouped_data)

# 下載必要資源
nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 停用詞與詞幹化
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # 小寫化 + 分詞
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  # 過濾非字母 + 詞幹化
    tokens = [word for word in tokens if word not in stop_words]  # 移除停用詞
    return tokens

# 對所有文本進行預處理
grouped_data["processed_text"] = grouped_data["title"].apply(preprocess_text)
print(grouped_data)





#%%
# 定義函數，執行 LDA 並輸出主題分佈
def lda(data, n_topics=3, n_top_words=5):
    # 創建詞袋模型
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    term_matrix = vectorizer.fit_transform(data)
    # 訓練 LDA 模型
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(term_matrix)
    # 獲取全局主題關鍵詞
    terms = vectorizer.get_feature_names_out()
    global_topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [terms[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        global_topics.append(f"Topic {topic_idx + 1}: {', '.join(top_keywords)}")
    
    return lda, vectorizer, global_topics


def compute_daily_topic_distribution(grouped_data, lda, vectorizer):
    daily_distributions = {}
    
    for _, row in grouped_data.iterrows():
        day_texts = [row["processed_text"]]
        # 使用全局詞表轉換當天文本
        day_term_matrix = vectorizer.transform(day_texts)
        # 計算主題分布
        day_topic_distribution = lda.transform(day_term_matrix)
        daily_distributions[row['date']] = day_topic_distribution
    
    return daily_distributions


n_topics = 10
n_top_words = 5
all_texts = grouped_data["processed_text"].tolist()
lda_model, vectorizer, global_topics = lda(all_texts, n_topics, n_top_words)
print("Global Topics:")
# 使用 join 方法將 global_topics 列表中的元素連接成一個字符串，每個元素之間用換行符分隔
print("\n".join(global_topics))
print("-" * 40)

# 計算每天的主題分布
daily_topic_distributions = compute_daily_topic_distribution(grouped_data, lda_model, vectorizer)


# %%

lda_dist_dict = {f'Topic {i+1}': [daily_topic_distributions[date][0][i] for date in daily_topic_distributions] for i in range(n_topics)}
lda_df = pd.DataFrame(lda_dist_dict)
final_df = pd.concat([vader_df, lda_df], axis=1)

# 補值：補上缺失的天數
full_date_range = pd.date_range(start=final_df['date'].min(), end=final_df['date'].max(), freq='D')
full_date_df = pd.DataFrame({'date': full_date_range})
final_df = pd.merge(full_date_df, final_df, on='date', how='left')
# 填充缺失值（目前全用 0）
final_df = final_df.fillna(0)
final_df = final_df[:-1]  # 刪除最後一行 2024/12/14 (配合數值型資料)
final_df.to_csv('nlp.csv', index=False)

with open('nlp.txt', 'w', encoding='utf-8') as f:
    for topic in global_topics:
        f.write(topic + '\n')
# %%
