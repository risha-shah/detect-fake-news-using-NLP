# Import Libraries
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

news_df = pd.read_csv('datasets/news.csv', encoding='utf-8')

# compute sentiment scores (polarity) and labels
sentiment_scores_tb = [round(TextBlob(article).sentiment.polarity, 3) for article in news_df['clean_text']]
sentiment_category_tb = ['positive' if score > 0 
                             else 'negative' if score < 0 
                                 else 'neutral' 
                                     for score in sentiment_scores_tb]

# sentiment statistics per news category
df = pd.DataFrame([list(news_df['news_category']), sentiment_scores_tb, sentiment_category_tb]).T
df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
df['sentiment_score'] = df.sentiment_score.astype('float')
df.groupby(by=['news_category']).describe()
for i in range(74):
    print(df['news_category'][i], end = " ")
    print(i, end = " ")
    print(df['sentiment_score'][i])
df.head()

maxSenti = max(df['sentiment_score'])
print('maxSenti: ', maxSenti)
minSenti = min(df['sentiment_score'])
print('minSenti: ', minSenti)


maxTech = df['sentiment_score'][0]
minTech = df['sentiment_score'][0]
index = 0
while index in range(74):
    score = df['sentiment_score'][index]
    category = df['news_category'][index]
    index+=1
    if(category != 'technology'): 
        break
    if(score > maxTech):
        maxTech = score
    if(score < minTech):
        minTech = score

maxSports = df['sentiment_score'][index]
minSports = df['sentiment_score'][index]
while index in range(74):
    score = df['sentiment_score'][index]
    category = df['news_category'][index]
    index+=1
    if(category != 'sports'): 
        break
    if(score > maxSports):
        maxSports = score
    if(score < minSports):
        minSports = score

maxWorld = df['sentiment_score'][index]
minWorld = df['sentiment_score'][index]
while index in range(74):
    score = df['sentiment_score'][index]
    index+=1
    if(score > maxWorld):
        maxWorld = score
    if(score < minWorld):
        minWorld = score

print('maxTech: ', maxTech)
print('minTech: ', minTech)
print('maxSports: ', maxSports)
print('minSports: ', minSports)
print('maxWorld: ', maxWorld)
print('minWorld ', minWorld)

fc = sns.factorplot(x="news_category", hue="sentiment_category", 
                    data=df, kind="count", 
                    palette={"negative": "#FE2020", 
                             "positive": "#BADD07", 
                             "neutral": "#68BFF5"})

#pos_idx = df[(df.news_category=='world') & (df.sentiment_score == 0.7)].index[0]
#neg_idx = df[(df.news_category=='world') & (df.sentiment_score == -0.296)].index[0]

pos_idx = df[df.sentiment_score == maxSenti].index[0]
neg_idx = df[df.sentiment_score == minSenti].index[0]
pos_idx_tech = df[(df.news_category=='technology') & (df.sentiment_score == maxTech)].index[0]
neg_idx_tech = df[(df.news_category=='technology') & (df.sentiment_score == minTech)].index[0]
pos_idx_sports = df[(df.news_category=='sports') & (df.sentiment_score == maxSports)].index[0]
neg_idx_sports = df[(df.news_category=='sports') & (df.sentiment_score == minSports)].index[0]
pos_idx_world = df[(df.news_category=='world') & (df.sentiment_score == maxWorld)].index[0]
neg_idx_world = df[(df.news_category=='world') & (df.sentiment_score == minWorld)].index[0]

print('Most Negative Article:', news_df.iloc[neg_idx][['news_article']][0])
print()
print('Most Positive Article:', news_df.iloc[pos_idx][['news_article']][0])
print()
print('Most Negative Tech News Article:', news_df.iloc[neg_idx_tech][['news_article']][0])
print()
print('Most Positive Tech News Article:', news_df.iloc[pos_idx_tech][['news_article']][0])
print()
print('Most Negative Sports News Article:', news_df.iloc[neg_idx_sports][['news_article']][0])
print()
print('Most Positive Sports News Article:', news_df.iloc[pos_idx_sports][['news_article']][0])
print()
print('Most Negative World News Article:', news_df.iloc[neg_idx_world][['news_article']][0])
print()
print('Most Positive World News Article:', news_df.iloc[pos_idx_world][['news_article']][0])