import pandas as pd
import os
import re
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



spark_per_case.shape

spark_per_case.apply(lambda x: x.isna().sum())

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

# extract the text of the last ruling in the case (which is in the first place in the dataset)
spark_per_case['the_last_ruling_text'] = spark_per_case['Резолютивная часть'].apply(lambda x: x[0] if x != [] else [])
spark_per_case['the_last_ruling_text'] = spark_per_case['the_last_ruling_text'].apply(flatten_lists)
spark_per_case['all_rulings_text'] = spark_per_case['Резолютивная часть'].apply(lambda x: '\n '.join(x))

# Ad-hoc pre-selection algorithm
#step 1: pick the latest ruling
# -if it's also the only ruling--proceed to step 2
# -otherwise, if it contains "оставить без изменения" (or or and?) "жалобу - без удовлетворения", drop the ruling and proceed to step 1
# -otherwise, if it does not contain ... proceed to step 2

def pre_sel_alg(rulings_list):
    if rulings_list == []:
        output_text = []
    elif len(rulings_list) == 1:
        output_text = rulings_list[0]
    else:
        for ruling_text in rulings_list:
            output_text = ruling_text
            no_change_phrase = '(?:жалобу\s+-*\s*без удовлетворения|оставить\sбез\sизменения)'
            contains_no_change_phrase = bool(re.search(no_change_phrase, ruling_text, flags=re.IGNORECASE))
            if not contains_no_change_phrase:
                break
    return output_text


spark_per_case['pre_sel_ruling_text'] = spark_per_case['Резолютивная часть'].apply(pre_sel_alg)



spark_per_case = spark_per_case.applymap(flatten_lists)



spark_per_case[['Номер дела', 'Категория', 'Исход дела']].groupby(['Категория', 'Исход дела']).agg('count')

spark_per_ruling = spark_per_case.explode('Резолютивная часть').reset_index()
spark_per_ruling.shape

spark_per_ruling['ruling_day'] = spark_per_ruling['Резолютивная часть'].str.extract('^(\d\d) *\. *\d\d *\. *\d{4}')
spark_per_ruling['ruling_month'] = spark_per_ruling['Резолютивная часть'].str.extract('^\d\d *\. *(\d\d) *\. *\d{4}')
spark_per_ruling['ruling_year'] = spark_per_ruling['Резолютивная часть'].str.extract('^\d\d *\. *\d\d *\. *(\d{4})')
spark_per_ruling['ruling_date'] = pd.to_datetime(dict(year=spark_per_ruling['ruling_year'],
                                                      month=spark_per_ruling['ruling_month'],
                                                      day=spark_per_ruling['ruling_day']))
spark_per_ruling.columns

spark_per_case['date_of_last_ruling'] = spark_per_ruling.groupby('№').first()['ruling_date']
spark_per_case['date_of_first_ruling'] = spark_per_ruling.groupby('№').last()['ruling_date']

spark_per_case.shape
spark_per_case['Состояние'].value_counts()

# Select only closed cases (Завершено) - we don't want on-going cases
spark_per_case = spark_per_case.loc[spark_per_case['Состояние'] == 'Завершено']

spark_per_case = spark_per_case.dropna(subset=['pre_sel_ruling_text'])
spark_per_case.shape


spark_per_case['the_last_ruling_text'].isna().sum()
spark_per_case['all_rulings_text'].isna().sum()


from razdel import tokenize
from nltk.stem.snowball import SnowballStemmer
#tokens = list(tokenize(spark_per_ruling['Резолютивная часть'][5]))
#print([_.text for _ in tokens])

def tokenize_and_stem(list_of_texts):
    tokens_of_all_cases = [list(tokenize(x)) for x in list_of_texts]
    stemmer = SnowballStemmer("russian")
    stemmed_tokens_all_cases = []
    for tokens_list in tokens_of_all_cases:
        lemmatized_tokens = [stemmer.stem(word) for word in [_.text for _ in tokens_list]]
        stemmed_tokens_all_cases.append(lemmatized_tokens)
    return stemmed_tokens_all_cases


stemmed_tokens_pre_sel_ruling = tokenize_and_stem(spark_per_case['pre_sel_ruling_text'].tolist())
stemmed_tokens_last_ruling = tokenize_and_stem(spark_per_case['the_last_ruling_text'].tolist())
stemmed_tokens_all_rulings = tokenize_and_stem(spark_per_case['all_rulings_text'].tolist())

not_labeled_obs_mask = spark_per_case['Исход дела'].isna()
y = (spark_per_case['Исход дела'][~not_labeled_obs_mask] == 'Иск не удовлетворен') * 1


#tokens_of_all_cases = [type(x) for x in spark_per_ruling['Резолютивная часть'].tolist()]
#tokens_of_all_cases = [x for x in spark_per_ruling['Резолютивная часть'].tolist() if type(x) is float]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV


class LogitReg:
    def __init__(self, stemmed_tokens, rulings_labels=spark_per_case['Исход дела'], rnd_state=42):
        self.stemmed_tokens = stemmed_tokens
        self.rulings_labels = rulings_labels
        self.rnd_state = rnd_state
        self.not_labeled_obs_mask = self.rulings_labels.isna()
        self.y = (self.rulings_labels[~self.not_labeled_obs_mask] == 'Иск не удовлетворен') * 1
        def identity_tokenizer(text):
            return text
        tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words=None, lowercase=False, min_df=5, max_df=0.9,
                                ngram_range=(1, 3))
        self.features = tfidf.fit_transform(self.stemmed_tokens).toarray()

    def fit_on_train_set(self, test_set_prop=0.1):
        X_labeled = self.features[~self.not_labeled_obs_mask]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_labeled,
                                                                                 self.y, test_size=test_set_prop, random_state=self.rnd_state)
        self.log_reg_model = LogisticRegressionCV(cv=5, random_state=self.rnd_state, penalty='l1', solver='liblinear').fit(self.X_train, self.y_train)

    def get_clas_report(self, use_data='test'):
        if use_data == 'train':
            print(classification_report(self.y_train, self.log_reg_model.predict(self.X_train)))
        elif use_data == 'test':
            print(classification_report(self.y_test, self.log_reg_model.predict(self.X_test)))
        else:
            print("Error: The 'use_data' argument should be either 'train' or 'test'.")

    def fit_on_whole_data(self):
        X_labeled = self.features[~self.not_labeled_obs_mask]
        self.log_reg_model = LogisticRegressionCV(cv=5, random_state=self.rnd_state, penalty='l1', solver='liblinear').fit(X_labeled, self.y)

    def get_preds_on_whole_data(self):
        return self.log_reg_model.predict(self.features)

    def get_proba_on_whole_data(self):
        return self.log_reg_model.predict_proba(self.features)

    def create_pred_df(self, ruling_texts=spark_per_case['the_last_ruling_text'],
                       last_ruling_date=spark_per_case['date_of_last_ruling'],
                       first_ruling_date=spark_per_case['date_of_first_ruling']):
        labeled_obs = (~self.not_labeled_obs_mask) * 1

        dict_data = {'last_ruling_text': ruling_texts,
                    'resolution_label': self.rulings_labels,
                    'labeled': labeled_obs,
                    'claim_not_sat_dummy': (self.rulings_labels == 'Иск не удовлетворен') * 1,
                    'claim_not_sat_pred': self.log_reg_model.predict(self.features),
                    'claim_not_sat_pred_prob': self.log_reg_model.predict_proba(self.features)[:, 1],
                    'last_ruling_date': last_ruling_date,
                    'first_ruling_date': first_ruling_date}

        return pd.DataFrame(dict_data)




logit_reg_last_rul = LogitReg(stemmed_tokens_last_ruling)
logit_reg_last_rul.fit_on_train_set(test_set_prop=0.15)
logit_reg_last_rul.get_clas_report()

logit_reg_pre_sel_rul = LogitReg(stemmed_tokens_pre_sel_ruling)
logit_reg_pre_sel_rul.fit_on_train_set(test_set_prop=0.15)
logit_reg_pre_sel_rul.get_clas_report()
logit_reg_pre_sel_rul.fit_on_whole_data()
np.unique(logit_reg_pre_sel_rul.get_preds_on_whole_data()[spark_per_case['Исход дела'].isna()],
          return_counts=True)
pre_sel_df = logit_reg_pre_sel_rul.create_pred_df(ruling_texts=spark_per_case['pre_sel_ruling_text'])
pre_sel_df.to_csv('classifying_decisions_logit_preds/pre_selection_preds.csv', index=True, encoding='utf-8')
pre_sel_df.to_excel('classifying_decisions_logit_preds/pre_selection_preds.xlsx', encoding='utf-8')

y_pred_pre_sel = logit_reg_pre_sel_rul.get_preds_on_whole_data()
# for no. 69 and 70 and 47 correct label should be 0
y_pred_pre_sel[spark_per_case.index == 69]
y_pred_pre_sel[spark_per_case.index == 70]
y_pred_pre_sel[spark_per_case.index == 47]

# for cases no. 15 and 19 correct label is 1
y_pred_pre_sel[spark_per_case.index == 15]
y_pred_pre_sel[spark_per_case.index == 19]



logit_reg_all_rul = LogitReg(stemmed_tokens_all_rulings)
logit_reg_all_rul.fit_on_train_set(test_set_prop=0.15)
logit_reg_all_rul.get_clas_report()
logit_reg_all_rul.fit_on_whole_data()
np.unique(logit_reg_all_rul.get_preds_on_whole_data()[spark_per_case['Исход дела'].isna()],
          return_counts=True)
all_rulings_df = logit_reg_all_rul.create_pred_df(ruling_texts=spark_per_case['all_rulings_text'])
all_rulings_df.to_csv('classifying_decisions_logit_preds/all_rulings_preds.csv', index=True, encoding='utf-8')
all_rulings_df.to_excel('classifying_decisions_logit_preds/all_rulings_preds.xlsx', encoding='utf-8')

y_pred_all = logit_reg_all_rul.get_preds_on_whole_data()
# for no. 69 and 70 and 47 correct label should be 0
y_pred_all[spark_per_case.index == 69]
y_pred_all[spark_per_case.index == 70]
y_pred_all[spark_per_case.index == 47]

# for cases no. 15 and 19 correct label is 1
y_pred_all[spark_per_case.index == 15]
y_pred_all[spark_per_case.index == 19]





y_pred = logistic_reg.predict(features)
y_pred_prob = logistic_reg.predict_proba(features)[:, 1] # probability of y == 1

np.unique(y_pred[not_labeled_obs_mask], return_counts=True)

# for no. 69 and 70 and 47 correct label should be 0
y_pred[spark_per_case.index == 69]
y_pred[spark_per_case.index == 70]
y_pred[spark_per_case.index == 47]


# for cases no. 15 and 19 correct label is 1
y_pred[spark_per_case.index == 15]
y_pred[spark_per_case.index == 19]




spark_cases_all = pd.read_excel('spark_cases_export.xlsx', sheet_name='report', header=1, skiprows=2)
spark_cases_all['№'] =spark_cases_all['№'].fillna(method='ffill').astype(int)
spark_per_case_all = spark_cases_all.groupby('№').agg(list)
spark_per_case_all = spark_per_case_all.applymap(lambda list_in_cell: [x for x in list_in_cell if x == x])

spark_per_case_all['Резолютивная часть'].apply(lambda x: x== []).sum()
spark_per_case_all[spark_per_case_all['Резолютивная часть'].apply(lambda x: x== [])]
## text from all rulimgs


pd.DataFrame(spark_per_case.index[spark_per_case['Резолютивная часть'].isna()]).to_csv('spark_cases_missing_ruling.csv', index = False)
#spark_per_case.index[spark_per_case['Резолютивная часть'].isna()].to_csv('spark_cases_missing_ruling.csv')

# Predict on the whole data
# Use only the last ruling
# exclude Обжалуется and

indices = np.argsort(logistic_reg.coef_)
feature_names = np.array(tfidf.get_feature_names())[indices]

# Words most indicative of the claim not being satisfied
feature_names[0][-5:]

# Words most indicative of the claim being fully or partially satisfied
feature_names[0][:5]

np.sort(logistic_reg.coef_)
