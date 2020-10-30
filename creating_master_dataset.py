"""
This script creates final dataset for the analysis.
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

firms_country = pd.read_excel('firm_list_v2_vasily.xlsx')

# import itertools
# plaintiff_firms = list(itertools.chain.from_iterable(spark_per_case['Истец'].tolist()))
# def_firms = list(itertools.chain.from_iterable(spark_per_case['Ответчик'].tolist()))

firms_country.columns

firm_names_list = firms_country['firm_names'].tolist()
firm_country_list = firms_country['Country'].tolist()

def merge_lists(list_to_merge=spark_per_case['Истец'], index_list=firm_names_list,
              fill_with=firm_country_list, skip_spark=False):
    outupt_list = []
    for firms_for_case in list_to_merge:
        firms_for_case_pred = []
        for firm_name in firms_for_case:
            if firm_name in index_list:
                firms_for_case_pred.append(fill_with[index_list.index(firm_name)])
            elif firm_name == 1 and not skip_spark:
                firms_for_case_pred.append('in_spark_database')
            elif firm_name == 1 and skip_spark:
                pass
            else:
                firms_for_case_pred.append('court/missing') # the name is a court not a firm
        outupt_list.append(firms_for_case_pred)
    return outupt_list

spark_per_case['Ответчик']

merge_lists()
merge_lists(spark_per_case['Ответчик'])
