import pandas as pd
import os
import numpy as np
working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"
os.chdir(working_directory)

spark_cases = pd.read_excel('spark_cases_export.xlsx', sheet_name='report', header=1, skiprows=2)

spark_cases.columns

spark_cases['№'] =spark_cases['№'].fillna(method='ffill').astype(int)

#spark_cases = spark_cases.fillna(method='ffill', axis=1)


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


spark_per_case['Третьи лица'][1]

flatten_lists(spark_per_case['Третьи лица'][2])

spark_per_case = spark_per_case.applymap(flatten_lists)

spark_per_case



spark_per_case.shape
spark_per_case['Исход дела'].isna().sum()
spark_per_case['Категория'].isna().sum()

spark_per_case.apply(lambda x: x.isna().sum())


spark_per_case[['Номер дела', 'Категория', 'Исход дела']].groupby(['Категория', 'Исход дела']).agg('count')
