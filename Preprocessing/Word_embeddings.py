# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:19:47 2020

@author: adrie
"""

import fasttext
import fasttext.util

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


df_preprocessed_covid= pd.read_csv('C:/Users/adrie/Documents/CORD-19/df_covid_preprocessed.csv')
#%%
#We only take article in english:
df_preprocessed_covid=df_preprocessed_covid[df_preprocessed_covid['language']=='en']


#%%
#fasttext.util.download_model('en', if_exists='ignore')  # English


#%% Loading the fastText model:

model = fasttext.load_model('C:/Users/adrie/Documents/CORD-19/embeddings_models/cc.en.300.bin')
#%%TFIDF function:
def transform_text(df_preprocessed_covid, column='body_text'):
    """
    Function transform_textn fitting the tfidf transformation and applying it to transform the articles
    @Inputs: 
    df_preprocessed_covid
    column
    @Outputs: 
    tfidf, the tfidf model
    tfidf_matrix, the tfidf matrix of the set of articles
    """
    df_preprocessed_covid[column] = df_preprocessed_covid[column].fillna(' ')
    # Initializing TFIDF :
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,1))
    
    #Defining the features needed :
    tfidf_matrix = tfidf.fit_transform(df_preprocessed_covid[column]).toarray()
    return tfidf, tfidf_matrix

#%% Performing the tfidf:
#With the custom function described above,we are able to obtain both the tfidf model and the tfidf matrix associated
#with the articles we are considering
tfidf, tfidf_matrix=transform_text(df_preprocessed_covid, column='body_text')
#%%Inspecting the model and getting the vector for the words in the tfidf:

%time embedding_vocab_vector=[model.get_word_vector(word) for word in tqdm(tfidf.get_feature_names())]
#%%we want an array, and we perform the dot product:
array_embedding_vocab_vector=np.array(embedding_vocab_vector)
%time embedding_tfidf_matrix=tfidf_matrix.dot(array_embedding_vocab_vector)
#%%Saving the results:
import pickle
#Saving:
with open('en_embedding_tfidf_matrix.pickle', 'wb') as f:
    pickle.dump([embedding_tfidf_matrix], f)
#%%
#Saving:
with open('en_embedding_vocab_vector.pickle', 'wb') as f:
    pickle.dump([embedding_vocab_vector], f)
 #%%   
#Saving:
with open('en_whole_tfidf_matrix.pickle', 'wb') as f:
    pickle.dump([tfidf_matrix], f)
#%%   
#Saving:
with open('en_tfidf.pickle', 'wb') as f:
    pickle.dump([tfidf], f) 

#%%
tfidf_query=tfidf.transform(['clara cells'])
#%%
%time embedding_tfidf_query=tfidf_query.dot(array_embedding_vocab_vector)



