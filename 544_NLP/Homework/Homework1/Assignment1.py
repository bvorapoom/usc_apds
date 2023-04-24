#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import re
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# In[2]:


# ! pip install bs4 # in case you don't have it installed
# ! pip install contractions

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# # 1. Dataset Preparation

# ### Read Data

# In[3]:


df = pd.read_csv('data.tsv', sep='\t', on_bad_lines='skip')


# ### Keep Reviews and Ratings

# In[4]:


df = df.loc[:, ['review_body', 'star_rating']]
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
df = df[~df['star_rating'].isna()]
df['star_rating'] = df['star_rating'].astype(int)


# In[5]:


df = df.loc[:, ['review_body', 'star_rating']]


# ### Group ratings to 3 classes

# In[6]:


mapping = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
df = df.replace({'star_rating': mapping})


#  ### We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[7]:


# drop duplicates
df = df.drop_duplicates()


# In[8]:


r_state = 555
list_sample = [df[df.star_rating == 1].sample(n=20000, random_state=r_state),
            df[df.star_rating == 2].sample(n=20000, random_state=r_state),
            df[df.star_rating == 3].sample(n=20000, random_state=r_state)]
df_sample = pd.concat(list_sample)


# In[9]:





# # 2. Data Cleaning
# 
# 

# ### Convert to lowercase

# In[10]:


df_clean = df_sample.copy()
df_clean.columns = ['review', 'stars']

df_clean['review'] = df_clean['review'].apply(str.lower)


# ### remove the HTML and URLs from the reviews

# In[11]:


# remove HTML
df_clean['review'] = df_clean['review'].str.replace(r'<[^<>]*>', '', regex=True)

# remove URLs
def remove_urls(text):
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

df_clean['review'] = df_clean['review'].apply(remove_urls)


# ### perform contractions on the reviews

# In[12]:


def perform_contractions(text):
    return ' '.join([contractions.fix(word) for word in text.split()])

df_clean['review'] = df_clean['review'].apply(perform_contractions)


# ### remove non-alphabetical characters

# In[13]:


def remove_non_alpha_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

df_clean['review'] = df_clean['review'].apply(remove_non_alpha_chars)


# ### remove extra spaces

# In[14]:


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text)

df_clean['review'] = df_clean['review'].apply(remove_extra_spaces)


# In[15]:


# printing average lengths before/after data cleaning
print('Average length of reviews before and after data cleaning: ',       df_sample['review_body'].str.len().mean(), ', ', df_clean['review'].str.len().mean(), sep='')


# # 3. Pre-processing

# ### Remove the stop words 

# In[16]:


stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df_preproc = df_clean.copy()
df_preproc['review'] = df_preproc['review'].apply(remove_stop_words)


# ### Perform lemmatization  

# In[17]:


lemmatizer = WordNetLemmatizer()

def perform_lemmatization(lemmatizer, text):
    lemmatized_list = []
    for word, pos_tag in nltk.pos_tag(text.split()):
        if pos_tag.startswith('V'):
            lemmatized_list.append(lemmatizer.lemmatize(word, 'v'))
        elif pos_tag.startswith('J'):
            lemmatized_list.append(lemmatizer.lemmatize(word, 'a'))
        else:
            lemmatized_list.append(lemmatizer.lemmatize(word))
    return ' '.join(lemmatized_list)

df_lemma = df_preproc.copy()
df_lemma['review'] = df_lemma['review'].apply(lambda x: perform_lemmatization(lemmatizer, x))


# In[18]:


# printing average lengths before/after data preprocessing
print('Average length of reviews before and after data preprocessing: ',       df_clean['review'].str.len().mean(), ', ', df_lemma['review'].str.len().mean(), sep='')


# # 4. TF-IDF Feature Extraction

# In[19]:


vectorizer = TfidfVectorizer(max_features=10000)

df_model = df_lemma.copy()
tfidf = vectorizer.fit_transform(df_model['review'])
df_X = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())


# In[20]:


df_y = df_model['stars'].reset_index(drop=True)
df_y = df_y.astype(int)


# ### Split training/testing set - 80/20

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=r_state, stratify=df_y)



# # 5. Perceptron

# In[22]:


def report_scores(clf, X_test, y_test, clf_name):
    y_pred = clf.predict(X_test)
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    key_list = ['1', '2', '3', 'weighted avg']
    report_name_list = ['(Class 1)', '(Class 2)', '(Class 3)', '(Average)']
    for i, k in enumerate(key_list):
        temp_report = dict_report[k]
        precision = str(temp_report['precision'])
        recall = str(temp_report['recall'])
        f1_score = str(temp_report['f1-score'])
        print('Precision, Recall, and f1-score for the testing split for ' + clf_name + ' ' + report_name_list[i] + ': ',              precision + ',' + recall + ',' + f1_score)
        
        


# In[23]:


clf_perceptron = Perceptron(random_state=r_state, penalty='elasticnet')
clf_perceptron.fit(X_train, y_train)


# In[24]:


report_scores(clf_perceptron, X_test, y_test, 'Perceptron')


# # 6. SVM

# In[25]:


clf_SVC = LinearSVC(random_state=r_state, multi_class='ovr', dual=True, max_iter=50000)
clf_SVC.fit(X_train, y_train)


# In[26]:


report_scores(clf_SVC, X_test, y_test, 'SVM')


# # 7. Logistic Regression

# In[27]:


clf_logreg = LogisticRegression(random_state=r_state, multi_class='ovr', max_iter=5000)
clf_logreg.fit(X_train, y_train)


# In[28]:


report_scores(clf_logreg, X_test, y_test, 'Logistic Regression')


# # 8. Naive Bayes

# In[29]:


clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)


# In[30]:


report_scores(clf_nb, X_test, y_test, 'Naive Bayes')


# ## References
# - https://stackoverflow.com/questions/45999415/removing-html-tags-in-pandas
# - https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
# - https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
# 
