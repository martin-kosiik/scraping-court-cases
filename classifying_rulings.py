import pandas as pd
import os
import re
import numpy as np
working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"
os.chdir(working_directory)

spark_cases = pd.read_excel('spark_cases_export.xlsx', sheet_name='report', header=1, skiprows=2)
spark_cases['№'] =spark_cases['№'].fillna(method='ffill').astype(int)
spark_per_case = spark_cases.groupby('№').agg(list)

# this removes nan from every list in the dataframe since nan is not equal to nan
spark_per_case = spark_per_case.applymap(lambda list_in_cell: [x for x in list_in_cell if x == x])



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


#spark_per_case[['Номер дела', 'Категория', 'Исход дела']].groupby(['Категория', 'Исход дела']).agg('count')

spark_per_ruling = spark_per_case.explode('Резолютивная часть').reset_index()
spark_per_ruling.shape

spark_per_ruling['ruling_day'] = spark_per_ruling['Резолютивная часть'].str.extract('^(\d\d) *\. *\d\d *\. *\d{4}')
spark_per_ruling['ruling_month'] = spark_per_ruling['Резолютивная часть'].str.extract('^\d\d *\. *(\d\d) *\. *\d{4}')
spark_per_ruling['ruling_year'] = spark_per_ruling['Резолютивная часть'].str.extract('^\d\d *\. *\d\d *\. *(\d{4})')
spark_per_ruling['ruling_date'] = pd.to_datetime(dict(year=spark_per_ruling['ruling_year'],
                                                      month=spark_per_ruling['ruling_month'],
                                                      day=spark_per_ruling['ruling_day']))

spark_per_case['date_of_last_ruling'] = spark_per_ruling.groupby('№').first()['ruling_date']
spark_per_case['date_of_first_ruling'] = spark_per_ruling.groupby('№').last()['ruling_date']

# Select only closed cases (Завершено) - we don't want on-going cases
spark_per_case = spark_per_case.loc[spark_per_case['Состояние'] == 'Завершено']
# Drop cases with missing ruling text
spark_per_case = spark_per_case.dropna(subset=['pre_sel_ruling_text'])
# Select only arbitrage cases that involve decisions of foreign courts
only_foreign_arb = spark_per_case['Категория'] == 'Признание решений иностранных судов и арбитражных решений'
spark_per_case = spark_per_case[only_foreign_arb].copy()

assert spark_per_case['the_last_ruling_text'].isna().sum() == 0
assert spark_per_case['all_rulings_text'].isna().sum() == 0


from razdel import tokenize
from nltk.stem.snowball import SnowballStemmer

tokens_to_remove = [',', ')', '(', '.', '-']

def tokenize_and_stem(list_of_texts):
    tokens_of_all_cases = [list(tokenize(x)) for x in list_of_texts]
    stemmer = SnowballStemmer("russian")
    stemmed_tokens_all_cases = []
    for tokens_list in tokens_of_all_cases:
        lemmatized_tokens = [stemmer.stem(word) for word in [_.text for _ in tokens_list] if stemmer.stem(word) not in tokens_to_remove]
        stemmed_tokens_all_cases.append(lemmatized_tokens)
    return stemmed_tokens_all_cases


stemmed_tokens_pre_sel_ruling = tokenize_and_stem(spark_per_case['pre_sel_ruling_text'].tolist())
stemmed_tokens_last_ruling = tokenize_and_stem(spark_per_case['the_last_ruling_text'].tolist())
stemmed_tokens_all_rulings = tokenize_and_stem(spark_per_case['all_rulings_text'].tolist())

round2_labels = pd.read_stata('classifying_decisions_logit_preds/cleaned_round_2.dta', index_col='A')
round2_labels = round2_labels[only_foreign_arb].copy()
# Make sure indexes are alligned
assert (spark_per_case.index.array == round2_labels.index.array).all()
round2_labels['decision'] = round2_labels['decision'].astype('Int64')

round3_labels = pd.read_csv('classifying_decisions_logit_preds/different_preds_2.csv', index_col='case_number')
round3_labels.index = round3_labels.index.rename('A')

round2_labels['round_3_decisions'] = round3_labels.correct_decision

round2_labels.loc[~round2_labels['round_3_decisions'].isna(), 'decision'] = \
        round2_labels['round_3_decisions'][~round2_labels['round_3_decisions'].isna()]




round2_labels.decision.isna().sum()
(round2_labels.decision == 0).sum()
not_labeled_obs_mask = spark_per_case['Исход дела'].isna()
y = (spark_per_case['Исход дела'][~not_labeled_obs_mask] == 'Иск не удовлетворен') * 1
decisions_1 = (spark_per_case['Исход дела'] == 'Иск не удовлетворен') * 1
decisions_1[not_labeled_obs_mask] = np.nan
decisions_1 = decisions_1.astype('Int64')
not_labeled_obs_mask_2 = round2_labels.decision.isna()
y_2 =  round2_labels.decision[~not_labeled_obs_mask_2]

spark_per_case['the_last_ruling_text'].index.tolist().index(659)
#tokens_of_all_cases = [type(x) for x in spark_per_ruling['Резолютивная часть'].tolist()]
#tokens_of_all_cases = [x for x in spark_per_ruling['Резолютивная часть'].tolist() if type(x) is float]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV


class LogitReg:
    def __init__(self, stemmed_tokens, rulings_labels=decisions_1, rnd_state=42):
        self.stemmed_tokens = stemmed_tokens
        self.rulings_labels = rulings_labels
        self.rnd_state = rnd_state
        self.not_labeled_obs_mask = self.rulings_labels.isna()
        self.y = self.rulings_labels[~self.not_labeled_obs_mask].astype('int64')
        def identity_tokenizer(text):
            return text

        self.tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words=None, lowercase=False, min_df=5, max_df=0.9,
                                        ngram_range=(1, 3))
        self.features = self.tfidf.fit_transform(self.stemmed_tokens).toarray()

    def fit_on_train_set(self, test_set_prop=0.1, regul_type='l1', solver='liblinear', max_iter=1000, Cs=10, cv_type=5):
        X_labeled = self.features[~self.not_labeled_obs_mask]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_labeled, self.y, test_size=test_set_prop, random_state=self.rnd_state)
        self.log_reg_model = LogisticRegressionCV(cv=cv_type, random_state=self.rnd_state, Cs=Cs, penalty=regul_type, solver=solver, max_iter=max_iter).fit(self.X_train, self.y_train)

    def get_clas_report(self, use_data='test'):
        if use_data == 'train':
            print(classification_report(self.y_train, self.log_reg_model.predict(self.X_train)))
        elif use_data == 'test':
            print(classification_report(self.y_test, self.log_reg_model.predict(self.X_test)))
        else:
            print("Error: The 'use_data' argument should be either 'train' or 'test'.")

    def fit_on_whole_data(self, regul_type='l1', Cs=10, cv_type=5):
        X_labeled = self.features[~self.not_labeled_obs_mask]
        self.log_reg_model = LogisticRegressionCV(cv=cv_type, Cs=Cs, random_state=self.rnd_state, penalty=regul_type, solver='liblinear').fit(X_labeled, self.y)

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
                    #'claim_not_sat_dummy': (self.rulings_labels == 'Иск не удовлетворен') * 1,
                    'claim_not_sat_pred': self.log_reg_model.predict(self.features),
                    'claim_not_sat_pred_prob': self.log_reg_model.predict_proba(self.features)[:, 1],
                    'last_ruling_date': last_ruling_date,
                    'first_ruling_date': first_ruling_date}

        return pd.DataFrame(dict_data)

    def plot_coefs(self, mean_contr=0):
        import matplotlib.pyplot as plt
        if mean_contr == 1:
            std_coefs = np.mean(self.features, 0) *self.log_reg_model.coef_[0]
            axis_label = 'Mean contribution'
        else:
            std_coefs = self.log_reg_model.coef_[0]
            axis_label = 'Coefficients'

        non_zero_coefs = std_coefs[std_coefs != 0]
        non_zero_features = np.array(self.tfidf.get_feature_names())[(std_coefs!= 0)]

        non_zero_features = non_zero_features[np.argsort(non_zero_coefs)]
        non_zero_coefs = np.sort(non_zero_coefs)

        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=[7,9.5])

        ax.barh(np.arange(len(non_zero_coefs)), non_zero_coefs, align='center')
        ax.set_yticks(np.arange(len(non_zero_coefs)) )
        ax.set_yticklabels(non_zero_features, {'fontsize':8})
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(axis_label)
        #plt.show()
        return fig

# stemmed_tokens_pre_sel_ruling.shape
# spark_per_case['Категория'][1]
# # DELETE THIS AFTERWARDS
# only_for_arb = spark_per_case['Категория'] == 'Признание решений иностранных судов и арбитражных решений'
# logit_reg_pre_sel_rul = LogitReg(np.array(stemmed_tokens_pre_sel_ruling)[only_for_arb].tolist(),
#                              rulings_labels=decisions_1[only_for_arb])
# logit_reg_pre_sel_rul.fit_on_train_set(test_set_prop=0.19)
# logit_reg_pre_sel_rul.get_clas_report()
# logit_reg_pre_sel_rul.fit_on_whole_data()
# y_pred_pre_sel_2 = logit_reg_pre_sel_rul.get_preds_on_whole_data()
#
# logit_reg_all_rul = LogitReg(np.array(stemmed_tokens_all_rulings)[only_for_arb].tolist(),
#                              rulings_labels=decisions_1[only_for_arb])
# logit_reg_all_rul.fit_on_train_set(test_set_prop=0.17)
# logit_reg_all_rul.get_clas_report()
# logit_reg_all_rul.fit_on_whole_data()
# y_pred_all_rul_2 = logit_reg_all_rul.get_preds_on_whole_data()
# not_labeled_obs_mask_2 = round2_labels.decision[only_for_arb].isna()
#
# (y_pred_pre_sel_2 != y_pred_all_rul_2).sum()
# (y_pred_pre_sel_2[not_labeled_obs_mask_2] != y_pred_all_rul_2[not_labeled_obs_mask_2]).sum()
#




logit_reg_last_rul = LogitReg(stemmed_tokens_last_ruling)
logit_reg_last_rul.fit_on_train_set(test_set_prop=0.15)
logit_reg_last_rul.get_clas_report()

logit_reg_pre_sel_rul = LogitReg(stemmed_tokens_pre_sel_ruling)
logit_reg_pre_sel_rul.fit_on_train_set(test_set_prop=0.20)
logit_reg_pre_sel_rul.get_clas_report()
logit_reg_pre_sel_rul.fit_on_whole_data()
np.unique(logit_reg_pre_sel_rul.get_preds_on_whole_data()[spark_per_case['Исход дела'].isna()],
          return_counts=True)
pre_sel_df = logit_reg_pre_sel_rul.create_pred_df(ruling_texts=spark_per_case['pre_sel_ruling_text'])
pre_sel_df.to_csv('classifying_decisions_logit_preds/pre_selection_preds.csv', index=True, encoding='utf-8')
pre_sel_df.to_excel('classifying_decisions_logit_preds/pre_selection_preds.xlsx', encoding='utf-8')

logit_reg_pre_sel_rul.plot_coefs().savefig(r'classifying_decisions_logit_preds\pre_sel_alg_coefs_plot.pdf', bbox_inches = "tight")
logit_reg_pre_sel_rul.plot_coefs(mean_contr=1).savefig(r'classifying_decisions_logit_preds\pre_sel_alg_mean_contr_plot.pdf', bbox_inches = "tight")
y_pred_pre_sel = logit_reg_pre_sel_rul.get_preds_on_whole_data()

# 2nd round of labels
logit_reg_pre_sel_rul_2 = LogitReg(stemmed_tokens_pre_sel_ruling, round2_labels.decision)
#logit_reg_pre_sel_rul_2.fit_on_train_set(Cs=[3.59e-01, 2.78, 2.15e+01], test_set_prop=0.18, solver='liblinear')
Cs_list = [1, 10, 40, 100, 500]
from sklearn.model_selection import RepeatedKFold
regul_type = 'l1'

logit_reg_pre_sel_rul_2.fit_on_train_set(test_set_prop=0.20, Cs=Cs_list, cv_type=RepeatedKFold(), regul_type=regul_type)
logit_reg_pre_sel_rul_2.get_clas_report()
logit_reg_pre_sel_rul_2.fit_on_whole_data(regul_type=regul_type, Cs=Cs_list, cv_type=RepeatedKFold())
y_pred_pre_sel_2 = logit_reg_pre_sel_rul_2.get_preds_on_whole_data()
y_prob_pre_sel_2 = logit_reg_pre_sel_rul_2.get_proba_on_whole_data()[:, 1]


logit_reg_pre_sel_rul_2.tfidf.get_feature_names()

logit_reg_pre_sel_rul_2.log_reg_model.Cs
logit_reg_pre_sel_rul_2.log_reg_model.C_
(logit_reg_pre_sel_rul_2.log_reg_model.coef_[0] != 0).sum()
logit_reg_all_rul_2.log_reg_model.coef_[0].shape
logit_reg_pre_sel_rul_2.log_reg_model.scores_

logit_reg_all_rul_2.log_reg_model.coef_[0].sum()

logit_reg_all_rul_2.log_reg_model.coef_[0][:50]
logit_reg_all_rul_2.plot_coefs()





logit_reg_all_rul = LogitReg(stemmed_tokens_all_rulings)
logit_reg_all_rul.fit_on_train_set(test_set_prop=0.15)
logit_reg_all_rul.get_clas_report()
logit_reg_all_rul.fit_on_whole_data()
np.unique(logit_reg_all_rul.get_preds_on_whole_data()[spark_per_case['Исход дела'].isna()],
          return_counts=True)
all_rulings_df = logit_reg_all_rul.create_pred_df(ruling_texts=spark_per_case['all_rulings_text'])
all_rulings_df.to_csv('classifying_decisions_logit_preds/all_rulings_preds.csv', index=True, encoding='utf-8')
all_rulings_df.to_excel('classifying_decisions_logit_preds/all_rulings_preds.xlsx', encoding='utf-8')


logit_reg_all_rul.plot_coefs().savefig(r'classifying_decisions_logit_preds\all_rulings_coefs_plot.pdf', bbox_inches = "tight")
logit_reg_all_rul.plot_coefs(mean_contr=1).savefig(r'classifying_decisions_logit_preds\all_rulings_mean_contr_plot.pdf', bbox_inches = "tight")


y_pred_all = logit_reg_all_rul.get_preds_on_whole_data()


(y_pred_pre_sel != y_pred_all).sum()

logit_reg_all_rul_2 = LogitReg(stemmed_tokens_all_rulings, round2_labels.decision)
logit_reg_all_rul_2.fit_on_train_set(test_set_prop=0.20, Cs=Cs_list, cv_type=RepeatedKFold(), regul_type=regul_type)
#logit_reg_all_rul_2.fit_on_train_set(test_set_prop=0.20)

logit_reg_all_rul_2.get_clas_report()

(logit_reg_all_rul_2.log_reg_model.coef_ != 0).sum()
logit_reg_all_rul_2.plot_coefs()
logit_reg_all_rul_2.log_reg_model.C_
logit_reg_all_rul_2.fit_on_whole_data(Cs=Cs_list, cv_type=RepeatedKFold(), regul_type=regul_type)
logit_reg_all_rul_2.create_pred_df()
logit_reg_all_rul_2.log_reg_model.scores_

y_pred_all_2 = logit_reg_all_rul_2.get_preds_on_whole_data()

y_prob_all_2 = logit_reg_all_rul_2.get_proba_on_whole_data()[:, 1]
y_prob_all_2[not_labeled_obs_mask_2 & diff_preds] -0.5

# Where do the predictions of the different models disagree?
(y_pred_pre_sel_2 != y_pred_all_2).sum()
(y_pred_pre_sel_2[not_labeled_obs_mask_2] != y_pred_all_2[not_labeled_obs_mask_2]).sum()



final_preds = pd.DataFrame({'№': spark_per_case.index.values, 'not_sat_pred_pre_sel': y_pred_pre_sel_2,
                            'not_sat_prob_pre_sel': y_prob_pre_sel_2, 'not_sat_pred_all_rul': y_pred_all_2,
                            'not_sat_prob_all_rul': y_prob_all_2 })

final_preds.to_csv('classifying_decisions_logit_preds/final_preds_2.csv', index=False, encoding='utf-8')

diff_preds = y_pred_pre_sel_2 != y_pred_all_2
diff_preds.__len__()
not_labeled_obs_mask_2.sum()
disagree_preds = pd.DataFrame({'pre_sel_pred': y_pred_pre_sel_2, 'all_rulings_pred': y_pred_all_2,
                                'category': spark_per_case['Категория'],
                                'all_rulings_text': spark_per_case['Резолютивная часть'],
                                'last_ruling_text': spark_per_case['the_last_ruling_text']})
disagree_preds = disagree_preds[diff_preds & not_labeled_obs_mask_2]
disagree_preds.to_excel('classifying_decisions_logit_preds/different_preds_after_round_2.xlsx', index=True)


firms_country = pd.read_excel('firm_list_v2_vasily.xlsx')

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




spark_per_case

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
