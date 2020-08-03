# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:01:43 2020

@author: adrie

Detection of the language in the articles
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
from spacy_langdetect import LanguageDetector
global spacy_nlp

spacy_nlp = spacy.load('en_core_web_sm')
spacy_nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)


#Loading data : 
df_preprocessed_covid= pd.read_csv('C:/Users/adrie/Documents/CORD-19/df_covid_preprocessed.csv')
#%%
#Function to detect the language:
def detect_articles_language(text):
    doc=spacy_nlp(text)
    return doc._.language

#Stocking the results:
All_results=[]
#Being able to perform :
#We change the maximum length for spacy (https://datascience.stackexchange.com/questions/38745/increasing-spacy-max-nlp-limit)
spacy_nlp.max_length = 1800000
#%%  Applying the function part, we use 8 cores, since there are 8 on my personal computer:
# Make the Pool of workers
pool = ThreadPool(8)
# Open every article in their own threads and return the results
%time results = pool.map(detect_articles_language,df_preprocessed_covid['body_text'].astype(str))
# Close the pool and wait for the work to finish
pool.close()
pool.join()
#%% If the results are ok :
All_results.extend(results)





#%% Verifying that we can access the languages and scores just like every dict : 
#for i in results:
#    print(i['score'])




#%%Saving what hat been done:
d={'language_score' : results}
df_covid_language=pd.DataFrame(data=d)
df_covid_language.to_csv('C:/Users/adrie/Documents/CORD-19/cord19_df_language.csv', index=False)
#pd.read_csv('C:/Users/adrie/Documents/CORD-19/cord19_df_pre_full_text_file_15000.csv')