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
# Remove firms with 'суд' in their name
spark_per_case['Истец link'] = spark_per_case['Истец link'].apply(lambda firms_in_list: [x for x in firms_in_list if not bool(re.search('\sсуд\s', str(x), flags=re.IGNORECASE))])
spark_per_case['Ответчик link'] = spark_per_case['Ответчик link'].apply(lambda firms_in_list: [x for x in firms_in_list if not bool(re.search('\sсуд\s', str(x), flags=re.IGNORECASE))])


def flatten_lists(the_list):
    if not isinstance(the_list, list):
        out_list = the_list
    elif len(the_list) == 0:
        out_list = np.nan
    elif len(the_list) == 1:
        out_list = the_list[0]
    else:
        out_list = the_list
    return out_list

spark_per_ruling = spark_per_case.explode('Резолютивная часть').reset_index()

1447/2178
spark_per_case.shape
spark_per_case['Категория'].apply(flatten_lists).value_counts()



import itertools

plaintiff_firms = list(itertools.chain.from_iterable(spark_per_case['Истец link'].tolist()))
def_firms = list(itertools.chain.from_iterable(spark_per_case['Ответчик link'].tolist()))
third_party_firms = list(itertools.chain.from_iterable(spark_per_case['Третьи лица link'].tolist()))

all_firms_list = plaintiff_firms + def_firms + third_party_firms
len(all_firms_list)


plaintiff_firms_w_spark = list(itertools.chain.from_iterable(spark_per_case['Истец'].tolist()))
def_firms_w_spark = list(itertools.chain.from_iterable(spark_per_case['Ответчик'].tolist()))
third_party_firms_w_spark = list(itertools.chain.from_iterable(spark_per_case['Третьи лица'].tolist()))

from collections import Counter

Counter([x in plaintiff_firms + def_firms for x in third_party_firms])
1108 + 338

all_firms_list_to_export = plaintiff_firms_w_spark + def_firms_w_spark
all_firms_list_to_export = [x for x in all_firms_list_to_export if x == x]
all_firms_list_to_export = list(dict.fromkeys(all_firms_list_to_export))
all_firms_list_to_export = [x for x in all_firms_list_to_export if not bool(re.search('\sсуд\s', x, flags=re.IGNORECASE))]
pd.DataFrame(all_firms_list_to_export, columns=['firm_names']).to_excel('firm_list.xlsx', index=False, encoding='utf-8')

filrm_list_v2 = pd.read_excel('firm_list_v2_vasily.xlsx')
exported_firms = filrm_list_v2.firm_names.tolist()

len(all_firms_list_to_export)
all_firms_list = plaintiff_firms + def_firms
spark_firms = [x for x in all_firms_list_to_export if x not in exported_firms]

pd.DataFrame(spark_firms, columns=['firm_names']).to_excel('firm_list_spark.xlsx', index=False, encoding='utf-8')

[x for x in exported_firms if x not in all_firms_list_to_export].__len__()

set(exported_firms).intersection(set(all_firms_list_to_export)).__len__()

len(exported_firms)

all_firms_list_to_export.__len__()
all_firms_list.__len__()

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
search_results_list = []
for query in queries_list:
    one_firm_results_list = []
    for link in search(query,  lang='en', safe='off', num=10, start=0, stop=10,
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
other_nation_domains = [ 'de', 'tr', 'it', 'uk', 'cz',
                        'sk', 'eu', 'bg', 'lu']

baltic_domains = ['ee', 'lv', 'lt']
other_cis_domains = ['kg', 'az', 'am', 'tj', 'uz', 'ro']

european_domains = pd.read_excel('domain_lists/european_domains.xlsx')['Domains'].str.replace('.', '').tolist()
list_of_selected_domains = baltic_domains + ['ru', 'ua', 'kz', 'by'] + other_cis_domains
european_domains = [x for x in european_domains if x not in list_of_selected_domains]

rest_of_world_domains = pd.read_csv('domain_lists/countries_tld.csv')['Name[7]'].str.replace('.', '').tolist()
list_of_selected_domains = baltic_domains + ['ru', 'ua', 'kz', 'by'] + other_cis_domains + european_domains
rest_of_world_domains = [x for x in rest_of_world_domains if x not in list_of_selected_domains]

internat_domains = ['com', 'org', 'info', 'biz', 'site', 'edu', 'gov', 'name', 'net', 'co']

tdl_list_total_counts = [sum(counter_obj.values()) for counter_obj in tdl_list_counts]
tdl_list_ru_counts = [counter_obj['ru'] + counter_obj['рф'] for counter_obj in tdl_list_counts]
tdl_list_ua_counts = [counter_obj['ua'] for counter_obj in tdl_list_counts]
tdl_list_kz_counts = [counter_obj['kz'] for counter_obj in tdl_list_counts]
tdl_list_by_counts = [counter_obj['by'] for counter_obj in tdl_list_counts]
tdl_list_european_counts = [reduce(lambda x,y: add_counts(x, y, count_obj=counter_obj), european_domains, 0) for counter_obj in tdl_list_counts]
tdl_list_rest_of_world_countries_counts = [reduce(lambda x,y: add_counts(x, y, count_obj=counter_obj), rest_of_world_domains, 0) for counter_obj in tdl_list_counts]
tdl_list_internat_counts = [reduce(lambda x,y: add_counts(x, y, count_obj=counter_obj), internat_domains, 0) for counter_obj in tdl_list_counts]
tdl_list_baltic_counts = [reduce(lambda x,y: add_counts(x, y, count_obj=counter_obj), baltic_domains, 0) for counter_obj in tdl_list_counts]
tdl_list_other_cis_counts = [reduce(lambda x,y: add_counts(x, y, count_obj=counter_obj), other_cis_domains, 0) for counter_obj in tdl_list_counts]


def prop(n, totals=10):
    return n/totals


def decision_rule(tdl_counts, p_thesh=0.75):
    internat_domains = ['com', 'org', 'info', 'biz', 'site', 'edu', 'gov', 'name']
    other_nation_domains = ['by', 'kz', 'ee', 'de', 'tr', 'it', 'lv', 'uk', 'cz',
                            'sk', 'eu', 'kg', 'uz', 'bg', 'az', 'lu']
    internat_counts = reduce(lambda x,y: add_counts(x, y, count_obj=tdl_counts), internat_domains, 0)
    other_nat_counts = reduce(lambda x,y: add_counts(x, y, count_obj=tdl_counts), other_nation_domains, 0)
    #p_thesh = 0.75 # threshold for classifying
    total_links = sum(tdl_counts.values())

    def prop(n, total=total_links):
        return n/total

    if total_links == 0:
        output = 'no_links'
    elif prop(tdl_counts['ua']) > p_thesh and tdl_counts['ua'] >= 2:
        output = 'ukrainian'
    elif prop(tdl_counts['by']) > p_thesh and tdl_counts['by'] >= 2:
        output = 'byelorussian'
    elif prop(tdl_counts['kz']) > p_thesh and tdl_counts['kz'] >= 2:
         output = 'kazakh'
    elif prop(tdl_counts['kg']) > p_thesh and tdl_counts['kg'] >= 2:
        output = 'kyrgyz'
    elif prop(tdl_counts['uz']) > p_thesh and tdl_counts['uz'] >= 2:
        output = 'uzbek'
    elif prop(tdl_counts['az']) > p_thesh and tdl_counts['az'] >= 2:
        output = 'azeri'
    elif prop(tdl_counts['am']) > p_thesh and tdl_counts['am'] >= 2:
        output = 'armenian'
    elif prop(tdl_counts['ro']) > p_thesh and tdl_counts['ro'] >= 2:
        output = 'romanain'
    elif prop(tdl_counts['tj']) > p_thesh and tdl_counts['tj'] >= 2:
        output = 'tajik'
    elif prop(other_nat_counts) > p_thesh and other_nat_counts >= 2:
        output = 'other_non_CIS'
    elif prop(tdl_counts['ru']) > p_thesh and tdl_counts['ru'] >= 2:
        output = 'russian'
    else:
        output = 'unclear_offshore'
    return output

classified_firms = [decision_rule(tdl_count, p_thesh=0.55) for tdl_count in tdl_list_counts]
cis_countries = ['kazakh', 'byelorussian', 'kyrgyz', 'uzbek', 'azeri', 'armenian',
                 'romanain', 'tajik']
classified_firms = ['CIS' if x in cis_countries else x for x in classified_firms]

Counter(classified_firms).keys()




links_df = pd.DataFrame({'firm_name_in_spark': only_firms_list ,'links': search_results_list,
                          'total_valid_links': tdl_list_total_counts,
                          'ru_domians': tdl_list_ru_counts,
                          'ua_domains': tdl_list_ua_counts,
                          'other_countries_domains': tdl_list_other_nat_counts,
                          'international_domains': tdl_list_internat_counts})

links_df.explode('links').to_excel('firm_links.xlsx')


def merge_lists(list_to_merge=spark_per_case['Истец link'], index_list=only_firms_list,
                fill_with=classified_firms, skip_spark=True):
    outupt_list = []
    for firms_for_case in list_to_merge:
        firms_for_case_pred = []
        for firm_name in firms_for_case:
            if firm_name in only_firms_list:
                firms_for_case_pred.append(fill_with[index_list.index(firm_name)])
            elif firm_name == 1 and not skip_spark:
                firms_for_case_pred.append('in_spark_database')
            elif firm_name == 1 and skip_spark:
                pass
            else:
                firms_for_case_pred.append('court/missing') # the name is a court not a firm
        outupt_list.append(firms_for_case_pred)
    return outupt_list



#spark_per_case['defendant_tdl_counts'] = merge_lists(spark_per_case['Ответчик link'], fill_with=tdl_list_counts)

def make_count_series(series_to_merge=spark_per_case['Ответчик link'],
                      fill_with=tdl_list_total_counts):
    output = merge_lists(series_to_merge, fill_with=fill_with)
    output = pd.Series(output, index=series_to_merge.index).apply(np.sum).astype(int)
    return output


spark_per_case['defendant_domain_total'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_total_counts)
spark_per_case['plaintiff_domain_total'] = make_count_series(spark_per_case['Истец link'], tdl_list_total_counts)

spark_per_case['defendant_ru'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_ru_counts)
spark_per_case['plaintiff_ru'] = make_count_series(spark_per_case['Истец link'], tdl_list_ru_counts)
spark_per_case['defendant_ua'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_ua_counts)
spark_per_case['plaintiff_ua'] = make_count_series(spark_per_case['Истец link'], tdl_list_ua_counts)
spark_per_case['defendant_by'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_by_counts)
spark_per_case['plaintiff_by'] = make_count_series(spark_per_case['Истец link'], tdl_list_by_counts)
spark_per_case['defendant_kz'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_kz_counts)
spark_per_case['plaintiff_kz'] = make_count_series(spark_per_case['Истец link'], tdl_list_kz_counts)
spark_per_case['defendant_baltic'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_baltic_counts)
spark_per_case['plaintiff_baltic'] = make_count_series(spark_per_case['Истец link'], tdl_list_baltic_counts)
spark_per_case['defendant_rest_of_europe'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_european_counts)
spark_per_case['plaintiff_rest_of_europe'] = make_count_series(spark_per_case['Истец link'], tdl_list_european_counts)
spark_per_case['defendant_international'] = make_count_series(spark_per_case['Ответчик link'], tdl_list_internat_counts)
spark_per_case['plaintiff_international'] = make_count_series(spark_per_case['Истец link'], tdl_list_internat_counts)


spark_per_case['number_of_defendants'] = spark_per_case['Ответчик link'].apply(len)
spark_per_case['number_of_plaintiffs'] = spark_per_case['Истец link'].apply(len)

spark_per_case['defend_in_spark'] = spark_per_case['Ответчик link'].apply(lambda x: Counter(x)[1])
spark_per_case['plaint_in_spark'] = spark_per_case['Истец link'].apply(lambda x: Counter(x)[1])

decision_preds_pre_sel.columns
decision_preds_pre_sel = pd.read_csv('classifying_decisions_logit_preds\pre_selection_preds.csv', index_col='№')
decision_preds_pre_sel = decision_preds_pre_sel.rename({"claim_not_sat_pred": "claim_not_sat_pred_pre_sel",
                                                        "claim_not_sat_pred_prob": "claim_not_sat_pred_prob_pre_sel"}, axis=1)

decision_preds_all = pd.read_csv('classifying_decisions_logit_preds/all_rulings_preds.csv', index_col='№')
decision_preds_all = decision_preds_all.drop(['last_ruling_text', 'resolution_label', 'labeled',
                                              'claim_not_sat_dummy', 'last_ruling_date', 'first_ruling_date'], axis=1)
decision_preds_all = decision_preds_all.rename({"claim_not_sat_pred": "claim_not_sat_pred_all_rul",
                                                "claim_not_sat_pred_prob": "claim_not_sat_pred_prob_all_rul"}, axis=1)

both_decision_preds = pd.concat([decision_preds_pre_sel, decision_preds_all], axis=1)


columns_to_drop = ['Номер дела', 'Мои списки', 'Категория', 'Истец', 'Ответчик',
                   'Третьи лица', 'Состояние', 'Исход дела', 'Дата иска', 'Дата решения',
                   'Сумма иска, RUB', 'Сумма по решению, RUB', 'Суд и судья',
                   'Представитель, Любая роль', 'Суть иска', 'Резолютивная часть',
                   'Истец link', 'Ответчик link', 'Третьи лица link']

spark_per_case.drop(columns_to_drop, axis=1, inplace=True)


def flatten_lists(the_list):
    if not isinstance(the_list, list):
        out_list = the_list
    elif len(the_list) == 0:
        out_list = np.nan
    elif len(the_list) == 1:
        out_list = the_list[0]
    else:
        out_list = ';'.join(the_list)
    return out_list

#spark_per_case = spark_per_case.applymap(flatten_lists)



spark_per_case = spark_per_case.join(both_decision_preds)

spark_per_case.last_ruling_text.isna().sum()

cols_to_int = ['labeled', 'claim_not_sat_dummy', 'claim_not_sat_pred_pre_sel',
               'claim_not_sat_pred_all_rul']
spark_per_case.labeled.astype(pd.Int32Dtype())
spark_per_case[cols_to_int] = spark_per_case[cols_to_int].apply(lambda x: x.astype(pd.Int32Dtype()))
spark_per_case.to_csv('firm_country_pred_v2.csv', index=True)



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
