# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 06:52:16 2020

@author: Martin Kosík
"""
import os
import re
import pandas as pd
import numpy as np
import pickle


working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"
os.chdir(working_directory)

#%%

arbitrage_rulings_df = pd.read_csv("arbitrage_rulings.csv", encoding="UTF-8")
arbitrage_rulings_df["proc_text"] = arbitrage_rulings_df.text.str.replace("NEWPAGE \n\d", "", regex=True)
              

#%%


#arbitrage_rulings_df["proc_text"].str.extractall(r'\\.(.+) .*обратил(а|о)сь в .*суд … с [исковым] заявлением (иском) к B [далее - …] … [в взыскании задолженности (долга) Х]')


matches_only_regex = arbitrage_rulings_df["proc_text"].str.lower().str.extract(r'(?:\n|\.)(?P<plaintiff>.+)[ \t\r\f\v](?:\(далее –?.*\))?.*обратил(?:а|о)сь\sв\s.*(?:суд|с\.)')


print(matches_only_regex.dropna())




#%% 
dp_ner_pred = pickle.load(open( "deeppavlov/model_pred_tagging_list_all.obj", "rb" ))

#%%
#dp_ner_pred[5][0][0]

# š corresponds to begining token of the name of an organization
# ř corresponds to continuation of the name of an organization
# ů corresponds to the name of court (i.e. string (Арбитражный) суд(а))
name_of_companies_list = []
context_list = []

#k=3
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
    
    sud_match = re.finditer('\s(?:Арбитражный\s)?суда?\s', original_text, flags=re.IGNORECASE)

    for matched_obj in sud_match:
        s_st = matched_obj.start()
        end_st = matched_obj.end()
        dp_ner_pred_repl_tags_text = dp_ner_pred_repl_tags_text[:s_st] + ' ' + ('ů') * (len(matched_obj.group())-2) + ' ' + dp_ner_pred_repl_tags_text[end_st:]

    
    assert (len(dp_ner_pred_repl_tags_text) == len(dp_ner_pred_repl_tags_text)), 'length of the strings is not the same'
    
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*)(?!.*B-ORG) обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*)^[B]\sобратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*).*\sобратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
   # captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд?(.*с\s.*заявлением\s.*к\s(?P<Defendant>š+(?: ř+)*))', dp_ner_pred_repl_tags_text)
    
  #  captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
  #  captured_match = re.search('(?P<Context>(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд)', dp_ner_pred_repl_tags_text)
  #  captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв[^ů]+', dp_ner_pred_repl_tags_text)
    
    if captured_match is None:
        name_of_companies_list.append(None)
        context_list.append(None)
    else:
        name_of_company = original_text[captured_match.start('Plaintiff'):(captured_match.end('Plaintiff'))]
        name_of_companies_list.append(name_of_company)  
        context_list.append(captured_match.group())

#
#
#%%
print(pd.Series(name_of_companies_list).dropna())


#%% extract the names of defendant firms
name_of_plaintiffs_list = []
name_of_defendants_list = []

#k=3
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

    sud_match = re.finditer('\s(?:Арбитражный\s)?суда?\s', original_text, flags=re.IGNORECASE)

    for matched_obj in sud_match:
       # print(original_text[matched_obj.start():matched_obj.end()])
       # print(dp_ner_pred_repl_tags_text[matched_obj.start():matched_obj.end()])
        s_st = matched_obj.start()
        end_st = matched_obj.end()
        dp_ner_pred_repl_tags_text = dp_ner_pred_repl_tags_text[:s_st] + ' ' + ('ů') * (len(matched_obj.group())-2) + ' ' + dp_ner_pred_repl_tags_text[end_st:]
    
    assert (len(dp_ner_pred_repl_tags_text) == len(dp_ner_pred_repl_tags_text)), 'length of the strings is not the same'
    
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*)(?!.*B-ORG) обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*)^[B]\sобратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    #sample_captures = re.findall('(?P<Plaintiff>B-ORG(?: I-ORG)*).*\sобратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s[^ů]+(.*с\s.*заявлением\s.*к\s(?P<Defendant>š+(?: ř+)*))*', dp_ner_pred_repl_tags_text)
#    captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s[^ů]+([^řš]+к\s(?P<Defendant>š+(?: ř+)*))*', dp_ner_pred_repl_tags_text)
    
  #  captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    if captured_match is None:
        name_of_plaintiffs_list.append(None)
        name_of_defendants_list.append(None)
    else:
        if captured_match.group('Plaintiff') is None:
            name_of_plaintiffs_list.append(None)
        else:
            name_of_company = original_text[captured_match.start('Plaintiff'):(captured_match.end('Plaintiff'))]
            name_of_plaintiffs_list.append(name_of_company)
        
        if captured_match.group('Defendant') is None:
            name_of_defendants_list.append(None)
        else:
            name_of_defendant = original_text[captured_match.start('Defendant'):(captured_match.end('Defendant'))]
            name_of_defendants_list.append(name_of_defendant)


#%%
print(pd.Series(name_of_defendants_list).dropna())
      

#%% BERT NER predictions
bert_ner_pred = pickle.load(open( "deeppavlov/bert_pred_tagging_list_all_2.obj", "rb" ))

#%% 
#now we have additional dimension for sentences
# First dimension - observation
# second dim - sentence 
# thrid - 0 original text, 1 labels
# fourth - empty dimension
# fifth - tokens

print(bert_ner_pred[0][12][0][0])   
print(len(bert_ner_pred[0]))   

#%%
dp_ner_pred_repl_tags_text_list = [None] * len(bert_ner_pred)
original_text_list = [None] * len(bert_ner_pred)


for k in range(len(bert_ner_pred)):
    dp_ner_pred_repl_tags = []
    original_text = []
    
    for s in range(len(bert_ner_pred[k])):

        for i, token in enumerate(bert_ner_pred[k][s][0][0]):
            original_text.append(token)
            
            if bert_ner_pred[k][s][1][0][i] == 'B-ORG':
                #dp_ner_pred_repl_tags.append(dp_ner_pred[k][1][0][i])
                dp_ner_pred_repl_tags.append('š' * len(token)) # we multiply by the length of the token
                                                                # to preserve the length of the string
            elif bert_ner_pred[k][s][1][0][i] == 'I-ORG':
                #dp_ner_pred_repl_tags.append(dp_ner_pred[k][1][0][i])
                dp_ner_pred_repl_tags.append('ř' * len(token))   # I am only using some character that is not
                                                                # in the text of the rulings (any such charcter would do)
            else:
                dp_ner_pred_repl_tags.append(token)
    
    dp_ner_pred_repl_tags_text_list[k] = ' '.join(dp_ner_pred_repl_tags)
    original_text_list[k] = ' '.join(original_text)
 
    sud_match = re.finditer('\s(?:Арбитражный\s)?суда?\s', original_text_list[k], flags=re.IGNORECASE)

    for matched_obj in sud_match:
        s_st = matched_obj.start()
        end_st = matched_obj.end()
        dp_ner_pred_repl_tags_text_list[k] = dp_ner_pred_repl_tags_text_list[k][:s_st] + ' ' + ('ů') * (len(matched_obj.group())-2) + ' ' + dp_ner_pred_repl_tags_text_list[k][end_st:]

    
  #  assert (len(dp_ner_pred_repl_tags_text) == len(dp_ner_pred_repl_tags_text)), 'length of the strings is not the same'

#%%
filehandler = open('auxiliary_files/bert_ner_pred_repl_tags_text.obj', 'wb') 
pickle.dump(dp_ner_pred_repl_tags_text_list, filehandler)
    
filehandler = open('auxiliary_files/bert_ner_pred_original_text.obj', 'wb') 
pickle.dump(original_text_list, filehandler)
#%%

name_of_companies_list_bert = []
context_list_bert = []

for k in range(len(bert_ner_pred)):
    
   # captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text_list[k])
    captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв[^ů]+', dp_ner_pred_repl_tags_text_list[k])
    
    if captured_match is None:
        name_of_companies_list_bert.append(None)
        context_list_bert.append(None)
    else:
        name_of_company = original_text_list[k][captured_match.start(1):(captured_match.end(1)+1)]
        name_of_companies_list_bert.append(name_of_company)    
        context_list_bert.append(captured_match.group())

print(pd.Series(name_of_companies_list_bert).dropna())


#%% extract names of defendants - BERT
name_of_plaintiffs_list_bert = []
name_of_defendants_list_bert = []

for k in range(len(bert_ner_pred)):

    captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s[^ů]+(.*с\s.*заявлением\s.*к\s(?P<Defendant>š+(?: ř+)*))*', dp_ner_pred_repl_tags_text_list[k])
#    captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s[^ů]+([^řš]+к\s(?P<Defendant>š+(?: ř+)*))*', dp_ner_pred_repl_tags_text)
    
  #  captured_match = re.search('(?P<Plaintiff>š+(?: ř+)*)[^řš]+обратил(?:а|о)сь\sв\s.*суд', dp_ner_pred_repl_tags_text)
    if captured_match is None:
        name_of_plaintiffs_list_bert.append(None)
        name_of_defendants_list_bert.append(None)
    else:
        if captured_match.group('Plaintiff') is None:
            name_of_plaintiffs_list_bert.append(None)
        else:
            name_of_company = original_text[captured_match.start('Plaintiff'):(captured_match.end('Plaintiff'))]
            name_of_plaintiffs_list_bert.append(name_of_company)
        
        if captured_match.group('Defendant') is None:
            name_of_defendants_list_bert.append(None)
        else:
            name_of_defendant = original_text[captured_match.start('Defendant'):(captured_match.end('Defendant'))]
            name_of_defendants_list_bert.append(name_of_defendant)


#%%
#name_of_companies_list_bert = [re.sub('NEWPAGE|NEWPAGE \n\d', '', x) for x in name_of_companies_list_bert]

def clean_comp_name(list_of_comp_names):
    name_of_companies_list_clean = []
    for x in list_of_comp_names:
        if x is None:
            name_of_companies_list_clean.append(x)
        else:
            x = re.sub('NEWPAGE|NEWPAGE \n\d|УСТАНОВИЛ', '', x, flags=re.IGNORECASE)
            x = re.sub('\s\s+', ' ', x, flags=re.IGNORECASE)
            name_of_companies_list_clean.append(x)

    return name_of_companies_list_clean


#%% cleaning the names of companies
name_of_companies_list_bert = clean_comp_name(name_of_companies_list_bert)
name_of_companies_list = clean_comp_name(name_of_companies_list)
name_of_defendants_list = clean_comp_name(name_of_defendants_list)
        
#%%

arbitrage_rulings_df['extracted_plaint_names_ner'] = pd.Series(name_of_companies_list_bert)
arbitrage_rulings_df.extracted_plaint_names_ner.fillna(value=np.nan, inplace=True)

arbitrage_rulings_df['context_plaint_names_ner'] = pd.Series(context_list_bert)
arbitrage_rulings_df.context_plaint_names_ner.fillna(value=np.nan, inplace=True)


filter_na_mask = arbitrage_rulings_df.extracted_plaint_names_ner.isna()
arbitrage_rulings_df.loc[filter_na_mask, 'extracted_plaint_names_ner'] = pd.Series(name_of_companies_list)[filter_na_mask]
arbitrage_rulings_df.extracted_plaint_names_ner.fillna(value=np.nan, inplace=True)

arbitrage_rulings_df.loc[filter_na_mask, 'context_plaint_names_ner'] = pd.Series(context_list)[filter_na_mask]
arbitrage_rulings_df.context_plaint_names_ner.fillna(value=np.nan, inplace=True)


#%%
print(arbitrage_rulings_df.extracted_plaint_names_ner.dropna())

#%%
matches_only_regex = matches_only_regex.plaintiff.str.replace('NEWPAGE|NEWPAGE \n\d|УСТАНОВИЛ', '', regex=True, flags=re.IGNORECASE)
matches_only_regex = matches_only_regex.str.replace('\s\s+', ' ', regex=True, flags=re.IGNORECASE)

#%%
arbitrage_rulings_df['extracted_plaint_names_regex'] = matches_only_regex
arbitrage_rulings_df.extracted_plaint_names_regex.fillna(value=np.nan, inplace=True)

arbitrage_rulings_df['extracted_plaint_names_any'] = arbitrage_rulings_df['extracted_plaint_names_ner']
filter_na_mask = arbitrage_rulings_df.extracted_plaint_names_ner.isna()
arbitrage_rulings_df.loc[filter_na_mask, 'extracted_plaint_names_any'] = arbitrage_rulings_df['extracted_plaint_names_regex'][filter_na_mask.values]
arbitrage_rulings_df.extracted_plaint_names_any.fillna(value=np.nan, inplace=True)

#%%
arbitrage_rulings_df['extracted_def_names_ner'] = pd.Series(name_of_defendants_list)
arbitrage_rulings_df.extracted_def_names_ner.fillna(value=np.nan, inplace=True)

#%%


arbitrage_rulings_df['ruling_date'] = arbitrage_rulings_df.case_id.str.extract('__(\d{8})( *\(\d\))*$')[0]
arbitrage_rulings_df['ruling_date'] = pd.to_datetime(arbitrage_rulings_df['ruling_date'], format='%Y%m%d')

arbitrage_rulings_df['starting_year'] = arbitrage_rulings_df.case_id.str.extract('(\d{4})__\d{8} *(\(\d\))*$')[0]


arbitrage_rulings_df['case_identifier'] = arbitrage_rulings_df['starting_year'].astype(int).astype(str) + '-' + arbitrage_rulings_df['case_spec_digit'].astype(str)

#%%
arbitrage_rulings_df['case_start_date_est'] = arbitrage_rulings_df.groupby('case_identifier').ruling_date.transform('min')
arbitrage_rulings_df['case_end_date_est'] = arbitrage_rulings_df.groupby('case_identifier').ruling_date.transform('max')

arbitrage_rulings_df.rename({'case_id': 'file_name'}, inplace=True)

#%%


# filter out the Crimea and Sevastopol cases                                                                         
arbitrage_rulings_df = arbitrage_rulings_df[~arbitrage_rulings_df['court_id'].isin(['А83', 'А84'])]

arbitrage_rulings_df.drop(['proc_text'], axis=1, inplace=True)

arbitrage_rulings_df.to_csv('arbitrage_rulings_more_info.csv', index=False, encoding='utf-8')



