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
print(dp_ner_pred[0][1][0][40:60])

#%%
pred_companies_first = []
pred_companies_first_n_mentions = []
pred_companies_second = []
pred_companies_second_n_mentions = []

for k in range(len(dp_ner_pred)):

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
    
    regex = re.compile(r'(^суд)| суд|NEWPAGE|^ООО$|^Банка*$|^КПП$|^БИК$|^ТПП$|^МКАС$|закон|ИНН|ОГРН|Торгово *-* *промышленной палате|^ООН$|Содружества Независимых Государств|^АПК$|Арбитраж|^Обществ(о|а|у)$|^Компании$|^Товариществ(о|а|у)$|^Президиум$|№', re.IGNORECASE)
    
    #filtered = filter(lambda i: not regex.search(i), full)
    merged_orgs = [i for i in merged_orgs if not regex.search(i)]
    
    counted_merged_orgs = Counter(merged_orgs)    
    counted_merged_orgs = Counter(el for el in counted_merged_orgs.elements() if counted_merged_orgs[el] >= 2)
    pred_involved_companies = counted_merged_orgs.most_common(2)
    try:
        pred_companies_first.append(pred_involved_companies[0][0])
        pred_companies_first_n_mentions.append(pred_involved_companies[0][1])
    except IndexError:
        pred_companies_first.append(None)
        pred_companies_first_n_mentions.append(None)
    try:
        pred_companies_second.append(pred_involved_companies[1][0])
        pred_companies_second_n_mentions.append(pred_involved_companies[1][1])
    except IndexError:
        pred_companies_second.append(None)
        pred_companies_second_n_mentions.append(None)
        


#%%
import pandas as pd
from navec import Navec
from slovnet import NER
from ipymarkup import show_span_ascii_markup as show_markup
#import numpy as np

arbitrage_rulings_df = pd.read_csv("arbitrage_rulings.csv", encoding="UTF-8")

arbitrage_rulings_df["proc_text"] = arbitrage_rulings_df.text.str.replace("NEWPAGE \n\d", "",
                                                                       regex=True)
text = 'Европейский союз добавил в санкционный список девять политических деятелей из самопровозглашенных республик Донбасса — Донецкой народной республики (ДНР) и Луганской народной республики (ЛНР) — в связи с прошедшими там выборами. Об этом говорится в документе, опубликованном в официальном журнале Евросоюза. В новом списке фигурирует Леонид Пасечник, который по итогам выборов стал главой ЛНР. Помимо него там присутствуют Владимир Бидевка и Денис Мирошниченко, председатели законодательных органов ДНР и ЛНР, а также Ольга Позднякова и Елена Кравченко, председатели ЦИК обеих республик. Выборы прошли в непризнанных республиках Донбасса 11 ноября. На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР.'
text = 'Кассационную  жалобу  Федерального  казенного  учреждения «Объединенное стратегическое командование Южного военного Округа» от 04.09.2013 № 3/12651 по делу № А06-8060/2012 возвратить заявителю. '

text = str(arbitrage_rulings_df.proc_text[1])

navec = Navec.load('slovnet/navec_news_v1_1B_250K_300d_100q.tar')
ner = NER.load('slovnet/slovnet_ner_news_v1.tar')
ner.navec(navec)

markup = ner(text)

#%%
one_span = markup.spans[0]
print(pred_involved_companies[0][0])