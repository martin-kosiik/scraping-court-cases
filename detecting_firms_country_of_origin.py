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
