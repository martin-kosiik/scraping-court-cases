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

sample_captures = re.findall('(?:\n|\.)(?P<Plaintiff>.+)\s.*обратил(?:а|о)сь\sв\s.*суд', sample_text)
#%%
print(arbitrage_rulings_df["proc_text"][894])


#%%
