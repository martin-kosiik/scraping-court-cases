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

spark_per_ruling[(spark_per_ruling['Резолютивная часть'].isna())]

spark_per_ruling = spark_per_case.explode('Резолютивная часть').reset_index()
spark_per_ruling


spark_per_ruling['ruling_day'] = spark_per_ruling['Резолютивная часть'].str.extract('^.*(\d\d) *\. *\d\d *\. *\d{4}')
spark_per_ruling['ruling_month'] = spark_per_ruling['Резолютивная часть'].str.extract('^.*\d\d *\. *(\d\d) *\. *\d{4}')
spark_per_ruling['ruling_year'] = spark_per_ruling['Резолютивная часть'].str.extract('^.*\d\d *\. *\d\d *\. *(\d{4})')

#arbitrage_rulings_df['ruling_date'] = pd.to_datetime(arbitrage_rulings_df['ruling_date'], format='%Y%m%d')

spark_per_ruling = spark_per_ruling.dropna(subset=['Резолютивная часть'])

spark_per_ruling['Исход дела'].value_counts()

spark_per_ruling['ruling_day'].isna().sum()


spark_per_ruling[]

from razdel import tokenize
tokens = list(tokenize(spark_per_ruling['Резолютивная часть'][5]))
#print([_.text for _ in tokens])

tokens_of_all_cases = [list(tokenize(x)) for x in spark_per_ruling['Резолютивная часть'].tolist()]

#tokens_of_all_cases = [type(x) for x in spark_per_ruling['Резолютивная часть'].tolist()]
#tokens_of_all_cases = [x for x in spark_per_ruling['Резолютивная часть'].tolist() if type(x) is float]


spark_per_ruling['Резолютивная часть'].isna().sum()

type(spark_per_ruling['Резолютивная часть'].tolist()[0])
tokens

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

stemmed_tokens_all_cases = []

for tokens_list in tokens_of_all_cases:
    lemmatized_tokens = [stemmer.stem(word) for word in [_.text for _ in tokens_list]]
    stemmed_tokens_all_cases.append(lemmatized_tokens)
#print(l)

stemmed_tokens_all_cases[2]


from sklearn.feature_extraction.text import TfidfVectorizer

tokenized_list_of_sentences = [['this', 'is', 'one', 'basketball'], ['this', 'is', 'a', 'football']]

def identity_tokenizer(text):
    return text

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words=None, lowercase=False)
tfidf.fit_transform(stemmed_tokens_all_cases)

tfidf.get_feature_names()

clf = MultinomialNB().fit(X_train_tfidf, y_train)


#spark_per_ruling['Резолютивная часть'].apply(lambda x: )
