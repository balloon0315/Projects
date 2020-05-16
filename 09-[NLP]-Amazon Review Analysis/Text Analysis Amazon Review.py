#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:24:55 2020

@author: balloon_n
"""

import numpy as np
import pandas as pd
# For visualizations
import matplotlib.pyplot as plt
# For regular expressions
import re
# For handling string
import string
# For performing mathematical operations
import math

# Preprocessing

# Import dataset
df=pd.read_csv('/Users/balloon_n/Documents/F/study/extra/Python/NLP/Amazon review/dataset.csv') 
print("Shape of data=>",df.shape)

print(df.columns)

#Drop unnecessary col
df=df[['name','reviews.text','reviews.doRecommend','reviews.numHelpful']]
print("Shape of data=>",df.shape)
print(df.head(5))

# check missing value
df.isnull().sum()
# drop null values
df.dropna(inplace=True) #Default=F,If True, do operation inplace, it returns none
df.isnull().sum()

#only considering those products that have at least 500 reviews.
# filter(lambda x: x < 3, data) ；  [x for x in data if x < 3]
df=df.groupby('name').filter(lambda x:len(x)>500).reset_index(drop=True)# drops the current index of the DataFrame and replaces it with an index of increasing integers
print('Number of products=>',len(df['name'].unique()))

df['reviews.doRecommend']=df['reviews.doRecommend'].astype(int)
df['reviews.numHelpful']=df['reviews.numHelpful'].astype(int)

df['name'].unique()

# text split
#Some product names contain repeating names separated by three consecutive commas (,,,)
df['name']=df['name'].apply(lambda x: x.split(',,,')[0])


for index,text in enumerate(df['reviews.text'][35:40]):
  print('Review %d:\n'%(index+1),text)
  #print('Review',index,":",text)

# list(enumerate(df['reviews.text'][35:40]))
# [(0, "I love everything about this tablet! The imaging is sharp and clear. It's fast and light weight. Love it!"),
#  (1, 'Overall a nice product for traveling purposes Value for money'),
#  (2, 'My children love this table great quality of pictures, excellent camera'),
#  (3, 'I bought 3 tablets and my family was not disappointed.'),
#  (4, 'Great tablet fast screen good size never gives me problem')]

############################################
# #Expand contractions (don’t for do not)
# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}


# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
# re.compile(r"(ain't|'s|aren't|can't|can't've|'cause|could've|c...)

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)
# Expanding Contractions in the reviews
df['reviews.text']=df['reviews.text'].apply(lambda x:expand_contractions(x))




# Lowercase the reviews
# Remove digits and words containing digits
# Remove punctuations
# Remove extra space

def clean_text_round(text):
    '''Make text lowercase,remove punctuation/words containing numbers/extra space'''
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text) # W*:A-Z  d:digit
    text = re.sub(' +',' ',text) # Removing extra spaces
    return text

cln = lambda x: clean_text_round(x)
df['cleaned']=df['reviews.text'].apply(cln)

for index,text in enumerate(df['cleaned'][35:40]):
  print('Review %d:\n'%(index+1),text)
  
  
# EDA
    # Document Term Matrix  
        # Stopwords Removal
        # Lemmatization
        # Create Document Term Matrix
  
  
import spacy
# Loading model
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

# Lemmatization with stopwords removal
df['lemmatized']=df['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))



# combine multiple reviews
    # # We are going to change this to key: comedian, value: string format
    # def combine_text(list_of_text):
    #     '''Takes a list of text and combines them into one large chunk of text.'''
    #     combined_text = ' '.join(list_of_text)
    #     return combined_text
    # data_combined = {key: [combine_text(value)] for (key, value) in data.items()}
    # data_combined

df_grouped=df[['name','lemmatized']].groupby(by='name').agg(lambda x:' '.join(x))
df_grouped.head()


# Creating Document Term Matrix
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df_grouped['lemmatized'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index=df_grouped.index
df_dtm.head(3)

#Wordcloud ########################################################################
from wordcloud import WordCloud
from textwrap import wrap  #wrap title


df_dtm=df_dtm.transpose()


# Plotting word cloud for each product
# using review (character) or document matrix(# of value).
for index,product in enumerate(df_dtm.columns):
    dt=df_dtm[product].sort_values(ascending=False) 
    wc = WordCloud(width=400, height=330, max_words=150,background_color="white",colormap="Dark2").generate_from_frequencies(dt)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('\n'.join(wrap(product,60)),fontsize=13)
    #tt=df_dtm.columns 
    #plt.title(tt[index])
    plt.show()
    
     # LOVE, USE, BUY, GREAT, and EASY are the most frequently occurring words for almost every product. This means that users are loving products from Amazon and found purchasing them a great decision. They also found them easy to use.

#Sentiment Analysis  (positive/opinionated)###########################################
# don't use dtm here!    
from textblob import TextBlob
df['polarity']=df['lemmatized'].apply(lambda x:TextBlob(x).sentiment.polarity)

print("3 Random Reviews with Highest Polarity:")
for index,review in enumerate(df.iloc[df['polarity'].sort_values(ascending=False)[:3].index]['reviews.text']):
  print('Review {}:\n'.format(index+1),review)

print("3 Random Reviews with Lowest Polarity:")
for index,review in enumerate(df.iloc[df['polarity'].sort_values(ascending=True)[:3].index]['reviews.text']):
  print('Review {}:\n'.format(index+1),review)

#  polarity ###############
product_polarity_sorted=pd.DataFrame(df.groupby('name')['polarity'].mean().sort_values(ascending=True))

plt.figure(figsize=(16,8))
plt.xlabel('Polarity')
plt.ylabel('Products')
plt.title('Polarity of Different Amazon Product Reviews')
polarity_graph=plt.barh(np.arange(len(product_polarity_sorted.index)),product_polarity_sorted['polarity'],color='purple',)

# Writing product names on bar
for bar,product in zip(polarity_graph,product_polarity_sorted.index):
  plt.text(0.005,bar.get_y()+bar.get_width(),'{}'.format(product),va='center',fontsize=11,color='white')

# Writing polarity values on graph
for bar,polarity in zip(polarity_graph,product_polarity_sorted['polarity']):
  plt.text(bar.get_width()+0.001,bar.get_y()+bar.get_width(),'%.3f'%polarity,va='center',fontsize=11,color='black')
  
plt.yticks([])
plt.show()

# recommend percentage ###############
recommend_percentage=pd.DataFrame(((df.groupby('name')['reviews.doRecommend'].sum()*100)/df.groupby('name')['reviews.doRecommend'].count()).sort_values(ascending=True))

plt.figure(figsize=(16,8))
plt.xlabel('Recommend Percentage')
plt.ylabel('Products')
plt.title('Percentage of reviewers recommended a product')
recommend_graph=plt.barh(np.arange(len(recommend_percentage.index)),recommend_percentage['reviews.doRecommend'],color='green')

# Writing product names on bar
for bar,product in zip(recommend_graph,recommend_percentage.index):
  plt.text(0.5,bar.get_y()+0.4,'{}'.format(product),va='center',fontsize=11,color='white')

# Writing recommendation percentage on graph
for bar,percentage in zip(recommend_graph,recommend_percentage['reviews.doRecommend']):
  plt.text(bar.get_width()+0.5,bar.get_y()+0.4,'%.2f'%percentage,va='center',fontsize=11,color='black')

plt.yticks([])
plt.show()

# Fire Kids Edition Tablet has the lowest recommendation percentage. It’s reviews also have the lowest polarity. So, we can say that the polarity of reviews affects the chances of a product getting recommended.

# conda install -c conda-forge textstat
import textstat
df['dale_chall_score']=df['reviews.text'].apply(lambda x: textstat.dale_chall_readability_score(x))
df['flesh_reading_ease']=df['reviews.text'].apply(lambda x: textstat.flesch_reading_ease(x))
df['gunning_fog']=df['reviews.text'].apply(lambda x: textstat.gunning_fog(x))

print('Dale Chall Score of upvoted reviews=>',df[df['reviews.numHelpful']>1]['dale_chall_score'].mean())
print('Dale Chall Score of not upvoted reviews=>',df[df['reviews.numHelpful']<=1]['dale_chall_score'].mean())

print('Flesch Reading Score of upvoted reviews=>',df[df['reviews.numHelpful']>1]['flesh_reading_ease'].mean())
print('Flesch Reading Score of not upvoted reviews=>',df[df['reviews.numHelpful']<=1]['flesh_reading_ease'].mean())

print('Gunning Fog Index of upvoted reviews=>',df[df['reviews.numHelpful']>1]['gunning_fog'].mean())
print('Gunning Fog Index of not upvoted reviews=>',df[df['reviews.numHelpful']<=1]['gunning_fog'].mean())

# Dale Chall Score of upvoted reviews=> 6.148739837398377
# Dale Chall Score of not upvoted reviews=> 5.695477993940365
# Flesch Reading Score of upvoted reviews=> 81.98257113821124
# Flesch Reading Score of not upvoted reviews=> 84.8586581087524
# Gunning Fog Index of upvoted reviews=> 7.980264227642276
# Gunning Fog Index of not upvoted reviews=> 6.8617784244936715

# There is very little difference in the Dale Chall Score and the Gunning Fog Index for helpful and not helpful reviews. But there is a considerable amount of variation in the Flesch Reading Score.

#Still, we cannot tell the difference in the readability of the two. The textstat library has a solution for this as well. It provides the text_standard() function. that uses various readability checking formulas, combines the result and returns the grade of education required to understand a particular document completely

# readability 
df['text_standard']=df['reviews.text'].apply(lambda x: textstat.text_standard(x))

print('Text Standard of upvoted reviews=>',df[df['reviews.numHelpful']>1]['text_standard'].mode())
print('Text Standard of not upvoted reviews=>',df[df['reviews.numHelpful']<=1]['text_standard'].mode())

# Text Standard of upvoted reviews=> 0    5th and 6th grade
# Text Standard of not upvoted reviews=> 0    5th and 6th grade
# Both upvoted and not upvoted reviews are easily understandable by anyone who has completed the 5th or 6th grade in school.


# read time
df['reading_time']=df['reviews.text'].apply(lambda x: textstat.reading_time(x))

print('Reading Time of upvoted reviews=>',df[df['reviews.numHelpful']>1]['reading_time'].mean())
print('Reading Time of not upvoted reviews=>',df[df['reviews.numHelpful']<=1]['reading_time'].mean())

# Reading Time of upvoted reviews=> 3.4542174796747958
# Reading Time of not upvoted reviews=> 1.7917397544251397
# the reading time of upvoted reviews is twice that of not upvoted reviews. It means that people usually find longer reviews helpful.


##Conclusion
# 1.Customers love products from Amazon. They find them a great purchase and easy to use
# 2.Amazon needs to work on the Fire Kids Edition Tablet because it has the most negative reviews. It is also the least recommended product
# 3.The majority of reviews are written in simple English and are easily understandable by anyone who has 5th or 6th grade of school
# 4.The reading time of helpful reviews is twice that of non-helpful reviews which means people find longer reviews helpful













