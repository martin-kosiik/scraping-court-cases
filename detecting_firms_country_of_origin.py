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

print(bool(re.search('\sсуд\s', 'Экономический  Гомельской области', flags=re.IGNORECASE)))

only_firms_list = [x for x in all_firms_list if not bool(re.search('\sсуд\s', x, flags=re.IGNORECASE))]
len(only_firms_list)




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


[x for x in extracted_locs_list if x != []]



locations_list.__len__()

#%%
countries_in_russian = pd.read_csv('countries_in_russian.csv', sep=';')
