# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 19:28:32 2020

@author: marti
"""
import os
import re
import pickle
from collections import Counter

#import pandas as pd
# set your current working directory

working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"

os.chdir(working_directory)

#%%

dp_ner_pred = pickle.load(open( "deeppavlov/model_pred_tagging_list_all.obj", "rb" ))

#%%
# note: the first dim. is observation, the second has original text at 0 and 
# the predictions (labels) at 1, and the third is redundatn
# and the forth is the order of the word within the text
print(dp_ner_pred[0][0][0][60])

#%%
pred_companies_all = []

for k in range(500):

    org_list_words = []
    org_list_labels = []
    
    for i, word in enumerate(dp_ner_pred[k][0][0]):
        if (dp_ner_pred[k][1][0][i] == 'B-ORG') or (dp_ner_pred[k][1][0][i] == 'I-ORG'):
            org_list_words.append(dp_ner_pred[k][0][0][i])
            org_list_labels.append(dp_ner_pred[k][1][0][i])
             
    merged_orgs = [None] * org_list_labels.count( 'B-ORG')
    j = 0
    for i, word in enumerate(org_list_labels):
        if (word == 'B-ORG'):
            merged_orgs[j] = org_list_words[i]
            j = j + 1
        else:
            merged_orgs[j-1] = merged_orgs[j-1] + ' ' + org_list_words[i]
    
    regex = re.compile(r'(^суд)| суд|NEWPAGE|ИНН|ОГРН|Торгово *- *промышленной палате|^ООН$|Содружества Независимых Государств|^АПК$|Арбитраж|^Обществ(о|а)$|^Компании$|^Товарищества(о|а)$|^Президиум$', re.IGNORECASE)
    
    #filtered = filter(lambda i: not regex.search(i), full)
    merged_orgs = [i for i in merged_orgs if not regex.search(i)]
    
    counted_merged_orgs = Counter(merged_orgs)    
    pred_involved_companies = counted_merged_orgs.most_common(2)
    pred_companies_all.append(pred_involved_companies)
