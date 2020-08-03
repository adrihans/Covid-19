# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 01:08:58 2020

@author: adrie

Python file for the sentence indexing
"""

#Loading packages
import numpy as np
import pandas as pd
import sys
sys.path.append("C:/Users/adrie/Documents/CORD-19")
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp


import datetime
import os
import nest_asyncio
nest_asyncio.apply()
import spacy 
global spacy_nlp
spacy_nlp = spacy.load('en_core_web_sm')


        
#Loading data : 
df_covid= pd.read_csv('C:/Users/adrie/Documents/CORD-19/cord19_df.csv')
#%%Defining some variables:

#Stocking the results:
All_results=[]
#Being able to perform :
#We change the maximum length for spacy (https://datascience.stackexchange.com/questions/38745/increasing-spacy-max-nlp-limit)
spacy_nlp.max_length = 1800000

#%% Function to index the sentences:
def sentence_indexer(text_article):
    doc=spacy_nlp(text_article)
    return [sentence.text for sentence in doc.sents]


#%%  Applying the function part, we use 8 cores, since there are 8 on my personal computer:
# Make the Pool of workers
pool = ThreadPool(8)
# Open every article in their own threads and return the results
%time results = pool.map(sentence_indexer,df_covid['body_text'][:10000].astype(str))
# Close the pool and wait for the work to finish
pool.close()
pool.join()

#%% If the results are ok :
All_results.extend(results)

#%%Saving what hat been done:
d={'sentences' : All_results}
df_covid_sentences_pre_10000=pd.DataFrame(data=d)
df_covid_sentences_pre_10000.to_csv('C:/Users/adrie/Documents/CORD-19/cord19_df_sentences_pre_10000.csv', index=False)

#%%

df_covid_sentences000.to_csv('C:/Users/adrie/Documents/CORD-19/cord19_df_list_sentences_19327.csv', index=False)