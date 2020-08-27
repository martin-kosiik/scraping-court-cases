# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 06:52:16 2020

@author: marti
"""
import os
import re
import pandas as pd

working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"
os.chdir(working_directory)


arbitrage_rulings_df = pd.read_csv("arbitrage_rulings.csv", encoding="UTF-8")
arbitrage_rulings_df["proc_text"] = arbitrage_rulings_df.text.str.replace("NEWPAGE \n\d", "", regex=True)
              
# filter out the Crimea and Sevastopol cases                                                                         
arbitrage_rulings_df = arbitrage_rulings_df[~arbitrage_rulings_df['court_id'].isin(['А83', 'А84'])]

#%%


#arbitrage_rulings_df["proc_text"].str.extractall(r'\\.(.+) .*обратил(а|о)сь в .*суд … с [исковым] заявлением (иском) к B [далее - …] … [в взыскании задолженности (долга) Х]')


matches = arbitrage_rulings_df["proc_text"].str.lower().str.extract(r'(?:\n|\.)(?P<plaintiff>.+)[ \t\r\f\v](?:\(далее –?.*\))?.*обратил(?:а|о)сь\sв\s.*(?:суд|с\.)')


print(matches.dropna())

#%%


matches = arbitrage_rulings_df["proc_text"].str.lower().str.extract(r'(?:установил:?\s*)(?P<plaintiff>.+)[ \t\r\f\v](?:\(далее –?.*\))?.*обратил(?:а|о)сь\sв\s.*суд')


print(matches.dropna())


#%%
#print(pd.Series('«agro green farm» ').str.extract(r'( green)'))

matches = arbitrage_rulings_df["proc_text"].str.lower().str.extract(r' (.+) .*обратил(?:а|о)сь в .*суд .*к (.+) ')

print(matches.dropna())

#%% other pattern
#А обратилось в … суд … с заявлением о признании и приведении в исполнение (признании обязательным) иностранного арбитражного решения (решений международного арбитража) (решения Арбитража) (… суда …).

matches_other_pattern = arbitrage_rulings_df["proc_text"].str.lower().str.extract(r'(?:\n|\.)(.+)[ \t\r\f\v](?:\(далее –?.*\))?.*обратил(?:а|о)сь\sв\s.*суд.*?(к\s?(.+))?(?:\n|\.)')

print(matches_other_pattern.dropna())

#%%
#print(arbitrage_rulings_df["proc_text"][154])

sample_text= """при участии представителей: 
от заявителя – Гюнтер В.А. дов. № 4-1470 от 17.04.2010г. 
от ответчика – Добрянская Н.Л. дов. № ПР/330 от 28.06.2011г. 
УСТАНОВИЛ: 
ЗАО НПО «Арктур» обратилось в Арбитражный суд г.Москвы с заявлением к 
Торгово-промышленной палате РФ о признании незаконными действий, направленных 
на вмешательство в международный коммерческий арбитраж по делам №№ 75/2008, 
120/2009;  признании  недействительным  Приказа  №  49  от  09.06.2011г.  в  части, 
устанавливающей  изменение  ст.  1  Регламента  Третейского  суда  для  разрешения 
экономических споров при ТПП РФ; обязании устранить допущенные нарушения . 
В соответствии с оспариваемым в части Приказом в Регламент Третейского суда 
для разрешения экономических споров при Торгово-промышленной палате РФ в п.2 """


sample_text= """при участии представителей: 
от заявителя – Гюнтер В.А. дов. № 4-1470 от 17.04.2010г. 
от ответчика – Добрянская Н.Л. дов. № ПР/330 от 28.06.2011г. 
УСТАНОВИЛ: 
B-ORGI-ORG I-ORG обратилось в Арбитражный суд г.Москвы с заявлением к 
Торгово-промышленной палате РФ о признании незаконными действий, направленных 
на вмешательство в международный коммерческий арбитраж по делам №№ 75/2008, 
120/2009;  признании  недействительным  Приказа  №  49  от  09.06.2011г.  в  части, 
устанавливающей  изменение  ст.  1  Регламента  Третейского  суда  для  разрешения 
экономических споров при ТПП РФ; обязании устранить допущенные нарушения . 
В соответствии с оспариваемым в части Приказом в Регламент Третейского суда 
для разрешения экономических споров при Торгово-промышленной палате РФ в п.2 """

sample_captures = re.findall('(?P<Plaintiff>B-ORG.+ORG)\s.*обратил(?:а|о)сь\sв\s.*суд', sample_text)

print(sample_captures)
#%%
print(arbitrage_rulings_df["proc_text"][894])


#%% 
import pickle
dp_ner_pred = pickle.load(open( "deeppavlov/model_pred_tagging_list_all.obj", "rb" ))

#%%
#dp_ner_pred[5][0][0]

# š corresponds to begining token of the name of an organization
# ř corresponds to continuation of the name of an organization
name_of_companies_list = []

for k in range(len(dp_ner_pred)):

    dp_ner_pred_repl_tags = []
    for i, token in enumerate(dp_ner_pred[k][0][0]):
        if dp_ner_pred[k][1][0][i] == 'B-ORG':
            #dp_ner_pred_repl_tags.append(dp_ner_pred[k][1][0][i])
            dp_ner_pred_repl_tags.append('š' * len(token)) # we multiply by the length of the token
                                                            # to preserve the length of the string
        elif dp_ner_pred[k][1][0][i] == 'I-ORG':
            #dp_ner_pred_repl_tags.append(dp_ner_pred[k][1][0][i])
            dp_ner_pred_repl_tags.append('ř' * len(token))   # I am only using some character that is not
                                                            # in the text of the rulings (any such charcter would do)
        else:
            dp_ner_pred_repl_tags.append(dp_ner_pred[k][0][0][i])
    
    dp_ner_pred_repl_tags_text = ' '.join(dp_ner_pred_repl_tags)
    original_text = ' '.join(dp_ner_pred[k][0][0])
    
    assert (len(dp_ner_pred_repl_tags_text) == len(dp_ner_pred_repl_tags_text)), 'length of the strings is not the same'
    
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*)(?!.*B-ORG) обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*)^[B]\sобратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*).*\sобратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    
    captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    
    if captured_match is None:
        name_of_companies_list.append(None)
    else:
        name_of_company = original_text[captured_match.start(1):(captured_match.end(1)+1)]
        name_of_companies_list.append(name_of_company)

#
#
#%%
print(pd.Series(name_of_companies_list).dropna())

