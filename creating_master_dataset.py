"""
This script creates final dataset for the analysis.
"""
import pandas as pd
import os
import numpy as np
import re

working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"
os.chdir(working_directory)

# from helper_functions import *
#
# del flatten_lists
#
# flatten_lists(['f'])

spark_cases = pd.read_excel('spark_cases_export.xlsx', sheet_name='report', header=1, skiprows=2)
spark_cases['№'] =spark_cases['№'].fillna(method='ffill').astype(int)
spark_per_case = spark_cases.groupby('№').agg(list)

# this removes nan from every list in the dataframe since nan is not equal to nan
spark_per_case = spark_per_case.applymap(lambda list_in_cell: [x for x in list_in_cell if x == x])

firms_country = pd.read_excel('firm_list_v2_vasily.xlsx')

# import itertools
# plaintiff_firms = list(itertools.chain.from_iterable(spark_per_case['Истец'].tolist()))
# def_firms = list(itertools.chain.from_iterable(spark_per_case['Ответчик'].tolist()))

firms_country.columns

firm_names_list = firms_country['firm_names'].tolist()
firm_country_list = firms_country['Country'].tolist()

def merge_lists(list_to_merge=spark_per_case['Истец link'], index_list=firm_names_list,
              fill_with=firm_country_list):
    outupt_list = []
    for firms_for_case in list_to_merge:
        firms_for_case_pred = []
        for firm_name in firms_for_case:
            if firm_name in index_list:
                firms_for_case_pred.append(fill_with[index_list.index(firm_name)])
            elif firm_name == 1:
                firms_for_case_pred.append('Россия')
            elif bool(re.search('\sсуд\s', str(firm_name), flags=re.IGNORECASE)):
                firms_for_case_pred.append('court') # the name is a court not a firm
            else:
                firms_for_case_pred.append('missing') # the name is missing
        outupt_list.append(firms_for_case_pred)
    return outupt_list


spark_per_case['plaintiff_country_list'] = merge_lists()
spark_per_case['plaintiff_country_list'].apply(set).apply(len).value_counts()
spark_per_case['defendant_country_list'] = merge_lists(spark_per_case['Ответчик link'])
spark_per_case['defendant_country_list'].apply(set).apply(len).value_counts()

spark_per_case['plaintiff_court_involved'] = spark_per_case['plaintiff_country_list'].apply(lambda x: 'court' in x).astype(int)
spark_per_case['defendant_court_involved'] = spark_per_case['defendant_country_list'].apply(lambda x: 'court' in x).astype(int)

assert spark_per_case['Истец link'][spark_per_case['plaintiff_country_list'].apply(lambda x: 'missing' in x)].values.tolist() == []
assert spark_per_case['Ответчик link'][spark_per_case['defendant_country_list'].apply(lambda x: 'missing' in x)].values.tolist() == []

spark_per_case[['plaintiff_country_1', 'plaintiff_country_2', 'plaintiff_country_3']] = \
                 spark_per_case['plaintiff_country_list'].apply(lambda x: set(x).difference({'court'})).apply(list).apply(pd.Series)
spark_per_case[['defendant_country_1', 'defendant_country_2']] = \
                 spark_per_case['defendant_country_list'].apply(lambda x: set(x).difference({'court'})).apply(list).apply(pd.Series)


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


spark_per_case = spark_per_case.applymap(flatten_lists)

# Select only closed cases (Завершено) - we don't want on-going cases
spark_per_case = spark_per_case.loc[spark_per_case['Состояние'] == 'Завершено']
spark_per_case.shape

only_foreign_arb = spark_per_case['Категория'] == 'Признание решений иностранных судов и арбитражных решений'
spark_per_case = spark_per_case[only_foreign_arb].copy()


final_preds = pd.read_csv(r'classifying_decisions_logit_preds\final_preds.csv', index_col='№')

cols_to_drop = ['Мои списки', 'Категория', 'Истец', 'Ответчик', 'Суд и судья',
                'Третьи лица', 'Состояние', 'Суть иска',
                'Истец link', 'Ответчик link', 'Третьи лица link', 'Представитель, Любая роль',
                'plaintiff_country_list', 'defendant_country_list']

spark_per_case = spark_per_case.drop(columns=cols_to_drop)
spark_per_case.columns


spark_per_case = spark_per_case.join(final_preds, how='left')


not_labeled_obs_mask = spark_per_case['Исход дела'].isna()
spark_per_case['not_labeled'] = not_labeled_obs_mask *1

decisions_1 = (spark_per_case['Исход дела'] == 'Иск не удовлетворен') * 1
decisions_1[not_labeled_obs_mask] = np.nan
decisions_1 = decisions_1.astype('Int64')
spark_per_case['not_sat_labeled'] = decisions_1


spark_per_case['not_sat_pred_pre_sel'] = spark_per_case['not_sat_pred_pre_sel'].astype('Int64')
spark_per_case['not_sat_pred_all_rul'] = spark_per_case['not_sat_pred_all_rul'].astype('Int64')

spark_per_case.to_csv('classifying_decisions_logit_preds/final_dataset.csv', index=True, encoding='utf-8')
spark_per_case.to_excel('classifying_decisions_logit_preds/final_dataset.xlsx', index=True, encoding='utf-8')





spark_per_case['Резолютивная часть'].isna().sum()
spark_per_case[['Категория', 'Исход дела', 'Резолютивная часть' ]].join(final_preds, how='left')['not_sat_pred_pre_sel'].isna().sum()
spark_per_case
spark_per_case[['Категория', 'Исход дела', 'Резолютивная часть' ]].join(final_preds, how='left')
