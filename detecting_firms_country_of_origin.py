"""
This script tries to automate the detection of country of origin of the firms involved in the arbitrage cases.
"""
import pandas as pd
import os
import numpy as np
import re

working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"
os.chdir(working_directory)

spark_cases = pd.read_excel('spark_cases_export.xlsx', sheet_name='report', header=1, skiprows=2)


spark_cases['№'] =spark_cases['№'].fillna(method='ffill').astype(int)

spark_per_case = spark_cases.groupby('№').agg(list)

# this removes nan from every list in the dataframe since nan is not equal to nan
spark_per_case = spark_per_case.applymap(lambda list_in_cell: [x for x in list_in_cell if x == x])
spark_per_case

def flatten_lists(the_list):
    if len(the_list) is 0:
        out_list = np.nan
    elif len(the_list) is 1:
        out_list = the_list[0]
    else:
        out_list = the_list
    return out_list

spark_per_ruling = spark_per_case.explode('Резолютивная часть').reset_index()


spark_per_case


import itertools

plaintiff_firms = list(itertools.chain.from_iterable(spark_per_case['Истец link'].tolist()))
def_firms = list(itertools.chain.from_iterable(spark_per_case['Ответчик link'].tolist()))
third_party_firms = list(itertools.chain.from_iterable(spark_per_case['Третьи лица link'].tolist()))

all_firms_list = plaintiff_firms + def_firms + third_party_firms
len(all_firms_list)

# Remove nan and 1 (which denotes the firms that are in the Spark database and thus are Russian)
all_firms_list = [x for x in all_firms_list if x == x and x != 1]

# Remove any duplicates in all_firms_list
all_firms_list = list(dict.fromkeys(all_firms_list))

len(all_firms_list)

all_firms_list

# We want to drop any names that contain phrase "суд" since they are not firms

only_firms_list = [x for x in all_firms_list if not bool(re.search('\sсуд\s', x, flags=re.IGNORECASE))]
len(only_firms_list)

only_firms_list

from googlesearch import search

query = "Карпатски Петролеум Корпорейшн"
for i in search(query,  lang='en', safe='off', num=10, start=0, stop=10, pause=2.0, country='', extra_params=None, user_agent=None):
    print(i)

[x for x in search(query,  lang='en', safe='off', num=10, start=0, stop=10, pause=2.0, country='', extra_params=None, user_agent=None)]

queries_list = only_firms_list
only_firms_list
search_results_list = []
for query in queries_list:
    one_firm_results_list = []
    for link in search(query,  lang='en', safe='off', num=8, start=0, stop=10,
                       pause=2.0, country='', extra_params=None, user_agent=None):
        one_firm_results_list.append(link)

    search_results_list.append(one_firm_results_list)


import pickle
pickle.dump(search_results_list, open( "auxiliary_files/search_results_list.p", "wb" ) )

search_results_list = pickle.load( open( "auxiliary_files/search_results_list.p", "rb" ) )
search_results_list

queries_list

# regex pattern to match the top level domain of the URL
tld_pattern = '(?:[-a-zA-Z0-9@:%_\+~.#=]{2,256}\.)?[-a-zA-Z0-9@:%_\+~#=]*\.([a-z]{2,3}\b)(?:[-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)'
tld_pattern = '(?:[-a-zA-Z0-9@:%_\+~.#=]{2,256}\.)?[-a-zA-Z0-9@:%_\+~#=]*\.([a-z]{2,3}\b)(?:[-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)'

import tldextract
tldextract.extract(search_results_list[0][1])


tdl_list = [[tldextract.extract(link).suffix for link in search_results] for search_results in search_results_list]
site_list = [[tldextract.extract(link).domain for link in search_results] for search_results in search_results_list]

# remove 'irrelevant sites' - those contain a database of the ruling texts and
# therefore are not informative about the country of origin of the firms
irrelev_sites = ['sudact', 'garant', 'consultant']
tdl_list = [[domain for j, domain in enumerate(domain_list) if not site_list[i][j] in irrelev_sites] for i, domain_list in enumerate(tdl_list)]

# We want to choose only the last domin (i.e. only 'kz' in 'gov.kz')
tdl_list = [[domain.split(sep='.')[-1] for domain in domain_list] for domain_list in tdl_list]

from collections import Counter

tdl_list_counts = [Counter(domain_list) for domain_list in tdl_list]

#matched_pat = re.search(tld_pattern, search_results_list[0][0], flags=re.IGNORECASE)
#print(matched_pat)

def add_counts(prev_count, choose_item='by', count_obj=tdl_list_counts[0]):
    return prev_count + count_obj[choose_item]

# We got some .su domains (Soviet Union) :-)
from functools import reduce
other_nation_domains = ['by', 'kz', 'ee', 'de', 'tr', 'it', 'lv', 'uk', 'cz',
                        'sk', 'eu', 'kg', 'uz', 'bg', 'az', 'lu']
internat_domains = ['com', 'org', 'info', 'biz', 'site', 'edu', 'gov', 'name']

tdl_list_total_counts = [sum(counter_obj.values()) for counter_obj in tdl_list_counts]
tdl_list_ru_counts = [counter_obj['ru'] for counter_obj in tdl_list_counts]
tdl_list_ua_counts = [counter_obj['ua'] for counter_obj in tdl_list_counts]
tdl_list_other_nat_counts = [reduce(lambda x,y: add_counts(x, y, count_obj=counter_obj), other_nation_domains, 0) for counter_obj in tdl_list_counts]
tdl_list_internat_counts = [reduce(lambda x,y: add_counts(x, y, count_obj=counter_obj), internat_domains, 0) for counter_obj in tdl_list_counts]

def prop(n, totals=10):
    return n/totals


def decision_rule(ru_counts, ua_counts, other_nat_counts, totals):
    def prop(n, total=totals):
        return n/total
    if totals == 0:
        output = 'no links'
    elif prop(ua_counts) > 0.2 or ua_counts >= 2:
        output = 'ukrainian'
    elif prop(other_nat_counts) > 0.2 or other_nat_counts >= 2:
         output = 'other country'
    elif prop(ru_counts) > 0.7:
        output = 'russian'
    else:
        output = 'missing'
    return output

n = 6
decision_rule(tdl_list_ru_counts[n], tdl_list_ua_counts[n], tdl_list_other_nat_counts[n], tdl_list_total_counts[n])
classified_firms = [decision_rule(tdl_list_ru_counts[n], tdl_list_ua_counts[n], tdl_list_other_nat_counts[n], tdl_list_total_counts[n]) for n in range(len(tdl_list_ru_counts))]

Counter(classified_firms)



def merge_lists(list_to_merge=spark_per_case['Истец link'], index_list=only_firms_list, fill_with=tdl_list_ua_counts):
    outupt_list = []
    for firms_for_case in list_to_merge:
        firms_for_case_pred = []
        for firm_name in firms_for_case:
            if firm_name in only_firms_list:
                firms_for_case_pred.append(fill_with[index_list.index(firm_name)])
            elif firm_name == 1:
                firms_for_case_pred.append('in spark database')
            else:
                firms_for_case_pred.append('court')
        outupt_list.append(firms_for_case_pred)
    return outupt_list


merge_lists(spark_per_case['Ответчик link'])[:6]
spark_per_case['Ответчик link'][:6]


# Next try to apply Named entity recognition model to detect people and locations

from navec import Navec
from slovnet import NER
from ipymarkup import show_span_ascii_markup as show_markup

text = 'Кассационную  жалобу  Федерального  казенного  учреждения «Объединенное стратегическое командование Южного военного Округа» от 04.09.2013 № 3/12651 по делу № А06-8060/2012 возвратить заявителю. '


navec = Navec.load('slovnet/navec_news_v1_1B_250K_300d_100q.tar')
ner = NER.load('slovnet/slovnet_ner_news_v1.tar')
ner.navec(navec)

markup = ner(text)

#%%
only_firms_list[15]


ner_predictions = [ner(x) for x in only_firms_list]


ner_entities = []
for ner_markup in ner_predictions:
    ner_ent_list_for_one_obs = []
    for ner_span in ner_markup.spans:
        ner_ent_list_for_one_obs.append(ner_span.type)
    ner_entities.append(ner_ent_list_for_one_obs)


zip(ner_predictions, ner_entities)

persons_list = [firm_name for firm_name, ner_label in zip(ner_predictions, ner_entities) if "PER" in ner_label]
persons_list.__len__()

only_person_list = [firm_name for firm_name, ner_label in zip(only_firms_list, ner_entities) if ner_label == ["PER"]]
only_person_list.__len__()
only_person_list


#locations_list = [firm_name for firm_name, ner_label in zip(only_firms_list, ner_entities) if "LOC" in ner_label]
locations_list = [firm_name for firm_name, ner_label in zip(ner_predictions, ner_entities) if "LOC" in ner_label]

locations_list

ner_predictions[36]


ner_predictions[36].text[ner_predictions[36].spans[1].start:ner_predictions[36].spans[1].stop]
ner_predictions[36].text
ner_predictions[36].spans[0].type == "ORG"


extracted_locs_list = []

for ner_prediction in ner_predictions:
    locations_one_obs_list = []
    for one_ner_span in ner_prediction.spans:
        if one_ner_span.type == "LOC":
            start_span = one_ner_span.start
            end_span = one_ner_span.stop
            locations_one_obs_list.append(ner_prediction.text[start_span:end_span])
    extracted_locs_list.append(locations_one_obs_list)


extracted_locs_list_non_empty = [x for x in extracted_locs_list if x != []]


from razdel import tokenize

list(tokenize(extracted_locs_list_non_empty[0][0]))
tokens_of_all_cases = [list(tokenize(x)) for x in spark_per_ruling['Резолютивная часть'].tolist()]

tokenized_loc_names = []
for loc_name in extracted_locs_list_non_empty:
    loc_name_tokens_list = []
    for word in loc_name:
        loc_name_tokens_list.extend(list(tokenize(word)))
    tokenized_loc_names.append(loc_name_tokens_list)


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

stemmed_tokens_all_locs = []
for tokens_list in tokenized_loc_names:
    lemmatized_tokens = [stemmer.stem(word) for word in [_.text for _ in tokens_list]]
    stemmed_tokens_all_locs.append(lemmatized_tokens)


locations_list.__len__()

#%%
countries_in_russian = pd.read_csv('countries_in_russian.csv', sep=';')

countries_in_russian['short_name_stemmed'] = countries_in_russian['short_name'].apply(lambda x: stemmer.stem(x))
