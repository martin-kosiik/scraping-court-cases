import os
import re
import pandas as pd
import numpy as np
import pickle


working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"
os.chdir(working_directory)

only_courts_list = pd.read_excel('only_courts_list.xlsx')
ukr_regions = pd.read_excel('ukr_regions.xlsx')

ukr_regions['region_clean'] = ukr_regions.region.str.lower().str.replace('область', '')
ukr_regions.stem_region = ukr_regions.stem_region.str.lower()




#!pip install pymystem3

from pymystem3 import Mystem
#text = "Красивая мама красиво мыла раму"
m = Mystem()
#>>> lemmas = m.lemmatize(text)
#>>> print(''.join(lemmas))

from nltk.stem.snowball import SnowballStemmer
from razdel import tokenize

tokens_to_remove = [',', ')', '(', '.', '-']

def tokenize_and_stem(list_of_texts):
    tokens_of_all_cases = [list(tokenize(x)) for x in list_of_texts]
    stemmer = SnowballStemmer("russian")
    stemmed_tokens_all_cases = []
    for tokens_list in tokens_of_all_cases:
        lemmatized_tokens = [stemmer.stem(word) for word in [_.text for _ in tokens_list] if stemmer.stem(word) not in tokens_to_remove]
        stemmed_tokens_all_cases.append(lemmatized_tokens)
    return stemmed_tokens_all_cases

def make_tokens(list_of_texts):
    tokens_of_all_cases = [list(tokenize(x)) for x in list_of_texts]
    stemmed_tokens_all_cases = []
    for tokens_list in tokens_of_all_cases:
        lemmatized_tokens = [word for word in [_.text for _ in tokens_list] if word not in tokens_to_remove]
        stemmed_tokens_all_cases.append(lemmatized_tokens)
    return stemmed_tokens_all_cases


tokenized_courts = tokenize_and_stem(only_courts_list['entity'])
tokenized_ukr_regions = tokenize_and_stem(ukr_regions['region_clean'])
import itertools
tokenized_ukr_regions_flat = list(itertools.chain(*tokenized_ukr_regions))



full_text_token_courts = []
for court in make_tokens(only_courts_list['entity']):
    full_text_token_courts.append(" ".join(court))



import regex

def reg_match(string, to_match, errs = 1):
    return (regex.search('\s(' + to_match +  '){e<=' + str(errs) + '}', string) is not None) *1

any_match = [reg_match(court, '|'.join(tokenized_ukr_regions_flat[1:])) for court in full_text_token_courts]

def match_to_courts(list_of_strings=full_text_token_courts, to_match=tokenized_ukr_regions_flat[1:]):
    court_matches = []

    for court in list_of_strings:
        part_court_match = []
        for ukr_court in to_match:
            if len(ukr_court) <= 4:
                if reg_match(court, ukr_court, errs = 0) == 1:
                    part_court_match.append(ukr_court)
            else:
                if reg_match(court, ukr_court, errs = 1) == 1:
                    part_court_match.append(ukr_court)

        if part_court_match == []:
            part_court_match.append('No match')
        court_matches.append(part_court_match)
    court_matches = list(itertools.chain(*court_matches))
    return court_matches




#full_text_token_courts[114]

reg_match(full_text_token_courts[114], ukr_regions.stem_region[6], errs = 0)

court = full_text_token_courts[114]
ukr_court = ukr_regions.stem_region[6]
part_court_match = []
if len(ukr_court) <= 4:
    if reg_match(court, ukr_court, errs = 0) == 1:
        part_court_match.append(ukr_court)
else:
    if reg_match(court, ukr_court, errs = 1) == 1:
        part_court_match.append(ukr_court)


only_courts_list['ukr_court_matches'] = match_to_courts(to_match = ukr_regions.stem_region)



tpp_matches = [bool(reg_match(court, 'тпп украин', 1)) for court in full_text_token_courts]
tpp_matches_full = [bool(reg_match(court, 'при Торгово-промышленной палате Украины', 6)) for court in only_courts_list['entity']]

# при Торгово-промышленной палате Украины
# тпп украин

only_courts_list.loc[tpp_matches_full, 'entity']

only_courts_list.loc[tpp_matches,]

only_courts_list.loc[tpp_matches_full ,'ukr_court_matches'] = 'тпп Украины'
only_courts_list.loc[tpp_matches ,'ukr_court_matches'] = 'тпп Украины'


only_courts_list.to_excel('ukr_courts_list_2.xlsx')

#https://ru.wikipedia.org/wiki/%D0%90%D0%B4%D0%BC%D0%B8%D0%BD%D0%B8%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B8%D0%B2%D0%BD%D0%BE%D0%B5_%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5_%D0%A3%D0%BA%D1%80%D0%B0%D0%B8%D0%BD%D1%8B#2016[%D0%9A_18]

# международн коммерческ арбитражн суд при тпп украин



final_dataset = pd.read_csv('final_dataset_2.csv')

courts_list = pd.read_excel('courts_list.xlsx')
import ast
courts_list.courts_list.apply(ast.literal_eval)
courts_list['courts_in_text'] = courts_list.courts_list.apply(ast.literal_eval)

tpp_matches = courts_list['courts_in_text'].apply(lambda x: any([bool(reg_match(court, 'тпп украин', 1)) for court in x]))
tpp_matches_full = courts_list['courts_in_text'].apply(lambda x: any([bool(reg_match(court, 'при Торгово-промышленной палате Украины', 6)) for court in x]))

(tpp_matches_full | tpp_matches).sum()

courts_list['ukr_court_matches'] = courts_list['courts_in_text'].apply(lambda x: match_to_courts(list_of_strings=x, to_match=ukr_regions.stem_region))

from helper_functions import flatten_lists
courts_list['ukr_court_matches'] = courts_list['ukr_court_matches'].apply(lambda x: set(x).difference({'No match'})).apply(list)

# In total 134 ukrainian courts - no case of ukrainian courts from 2 or more regions in a single case
courts_list['ukr_court_matches'].apply(len).value_counts()

courts_list['ukr_court_matches'] = courts_list['ukr_court_matches'].apply(flatten_lists)


courts_list['ukr_court'] = (1 -courts_list['ukr_court_matches'].isna()) * (1)

assert all(courts_list.loc[tpp_matches_full | tpp_matches, 'ukr_court_matches'].isna()), 'Tpp overlaps with regional ukr. courts'

courts_list.loc[tpp_matches_full | tpp_matches, 'ukr_court_matches'] = 'тпп Украины'

courts_list_agg = courts_list[['Номер дела', 'courts_in_text', 'ukr_court', 'ukr_court_matches']].groupby('Номер дела', as_index=False).agg({'ukr_court' : [('ukr_court' , 'sum')], 'ukr_court_matches' :[('ukr_court_matches',lambda x:list(x))],
                                                             'courts_in_text': [('courts_in_text', lambda x:list(x))]})

courts_list_agg.columns = courts_list_agg.columns.get_level_values(1)
courts_list_agg.rename({'': 'Номер дела'}, axis=1, inplace=True)
courts_list_agg['ukr_court'].value_counts()


final_dataset = pd.merge(final_dataset, courts_list_agg, on = 'Номер дела', how='left')

assert final_dataset.ukr_court.isna().sum() == 0, 'Some cases were not joined'

final_dataset.to_csv('final_dataset_3.csv', index=False, encoding='utf-8')
