# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:15:54 2020

@author: adrie


Preprocessing.py file to be able to multithread rhe preprocessing part of the project, because it tends to be 
very long to put in place. 
This can also help to make several preprocessing of the data, depending on the purpose of the task.
"""
#Loading packages
import numpy as np
import pandas as pd
import sys
sys.path.append("C:/Users/adrie/Documents/CORD-19")
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp

import pandas as pd
import numpy as np
import datetime
import os
import nest_asyncio
nest_asyncio.apply()

import string
import spacy 
import nltk
from normalise import normalise
from nltk.tokenize import word_tokenize
global spacy_nlp
spacy_nlp = spacy.load('en_core_web_sm')



#Loading data : 
df_covid= pd.read_csv('C:/Users/adrie/Documents/CORD-19/cord19_df.csv')
#%%
#Functions to be used: 
def normalize(text):
    # some issues in normalise package
    try:
        return ' '.join(normalise(text,tokenizer=word_tokenize,verbose=False))
    except:
        return text
def remove_punct(doc):
    return [t for t in doc if t.text not in string.punctuation]
def remove_stop_words(doc):
    return [t for t in doc if not t.is_stop]
def lemmatize(doc):
    return ' '.join([t.lemma_ for t in doc])
    
def preprocessing_part(text):
    normalized_text=normalize(text).lower()
    doc=spacy_nlp(normalized_text)
    removed_punct=remove_punct(doc)
    removed_stop_words=remove_stop_words(removed_punct)
    return lemmatize(removed_stop_words)

#Stocking the results:
All_results=[]
#Being able to perform the pre processing part :
#We change the maximum length for spacy (https://datascience.stackexchange.com/questions/38745/increasing-spacy-max-nlp-limit)
spacy_nlp.max_length = 1800000
#%%  Applying the preprocessing part, we use 8 cores, since there are 8 on my personal computer:
# Make the Pool of workers
pool = ThreadPool(8)
# Open every article in their own threads and return the results
%time results = pool.map(preprocessing_part,df_covid['body_text'][27000:].astype(str))
# Close the pool and wait for the work to finish
pool.close()
pool.join()
#%% If the results are ok :
All_results.extend(results)
#Saving the preprocessed articles
#df_covid.body_text[:10000]=results
#df_covid.to_csv('C:/Users/adrie/Documents/CORD-19/cord19_df_pre_full_text_file.csv', index=False)



#%%Saving what hat been done:
d={'body_text_post_15000' : All_results}
df_all_results_post_15000=pd.DataFrame(data=d)
df_all_results_post_15000.to_csv('C:/Users/adrie/Documents/CORD-19/cord19_df_pre_full_text_file_post_15000.csv', index=False)
#pd.read_csv('C:/Users/adrie/Documents/CORD-19/cord19_df_pre_full_text_file_15000.csv')