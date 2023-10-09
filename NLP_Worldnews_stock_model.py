import pandas as pd
import numpy as np
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#for word embedding
import gensim
from gensim.models import Word2Vec

whole_dataset = pd.read_csv('Input/Combined_News_DJIA.csv', header=0)
index = whole_dataset.loc[whole_dataset['Date'] == '2015-01-02'].index
num_of_rows = len(whole_dataset)
#print("HELLOHELLO", index[0])

#We add +1 to correct the zero-indexing
rows_to_keep = [*range(0, index[0]+1, 1)]

#rows_to_keep = [whole_dataset[0:index[0]]]
df_train= pd.read_csv('Input/Combined_News_DJIA.csv', header=0, skiprows = lambda x: x not in rows_to_keep)
print("TRAINTRAINTRAINTRAIN:", df_train)
df_test = pd.read_csv('Input/Combined_News_DJIA.csv', header=0, skiprows = lambda x: x in rows_to_keep)
print("TESTTESTTEST:", df_test)


