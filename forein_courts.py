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



tokenized_courts = tokenize_and_stem(only_courts_list['entity'])
tokenized_ukr_regions = tokenize_and_stem(ukr_regions['region_clean'])
import itertools
tokenized_ukr_regions_flat = list(itertools.chain(*tokenized_ukr_regions))

full_text_token_courts = []
for court in tokenized_courts:
    full_text_token_courts.append(" ".join(court))

import regex

def reg_match(string, to_match, errs = 1):
    return (regex.search('(' + to_match +  '){e<=' + str(errs) + '}', string) is not None) *1

any_match = [reg_match(court, '|'.join(tokenized_ukr_regions_flat[1:])) for court in full_text_token_courts]

court_matches = []

for court in full_text_token_courts:
    part_court_match = []
    for ukr_court in tokenized_ukr_regions_flat[1:]:
        if reg_match(court, ukr_court) == 1:
            part_court_match.append(ukr_court)
    if part_court_match == []:
        part_court_match.append('No match')
    court_matches.append(part_court_match)

len(court_matches)
court_matches = list(itertools.chain(*court_matches))


only_courts_list['ukr_court_matches'] = court_matches

full_text_token_courts[4]

kiev_matches = [bool(reg_match(court, 'киев*(\s|\.)', 0)) for court in only_courts_list['entity']]
sum(kiev_matches)

tpp_matches = [bool(reg_match(court, 'тпп украин', 1)) for court in full_text_token_courts]
tpp_matches_full = [bool(reg_match(court, 'при Торгово-промышленной палате Украины', 6)) for court in only_courts_list['entity']]

# при Торгово-промышленной палате Украины
# тпп украин

only_courts_list.loc[tpp_matches_full, 'entity']

only_courts_list.loc[tpp_matches,]

only_courts_list.loc[tpp_matches_full ,'ukr_court_matches'] = 'тпп Украины'
only_courts_list.loc[tpp_matches ,'ukr_court_matches'] = 'тпп Украины'


only_courts_list.to_excel('ukr_courts_list.xlsx')

#https://ru.wikipedia.org/wiki/%D0%90%D0%B4%D0%BC%D0%B8%D0%BD%D0%B8%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B8%D0%B2%D0%BD%D0%BE%D0%B5_%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5_%D0%A3%D0%BA%D1%80%D0%B0%D0%B8%D0%BD%D1%8B#2016[%D0%9A_18]

# международн коммерческ арбитражн суд при тпп украин

np.sum(kiev_matches)
any_match.__len__()
np.sum(any_match)


ukr_regions['region_clean'].apply(lambda x: ''.join(m.lemmatize(x)))

courts_list_flat_lem = [''.join(m.lemmatize(text)) for text in courts_list_flat]
