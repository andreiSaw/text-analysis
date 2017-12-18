import pandas as pd
import numpy as np
from sklearn.externals import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class Solution:  # Class must have name "Solution"
    def __init__(self):
        self.tfidf = None
        self.vectorizer = None

    def train(self, training_corpus):

        train_df = pd.DataFrame.from_dict(training_corpus)
        Z = train_df

        Z.loc[:, 'q'] = pd.Series(np.random.randn(len(Z)), index=Z.index)
        for index, row in Z.iterrows():
            Z.at[index, 'q'] = len(Z.at[index, 'texts'])
            Z.at[index, 'text'] = ''.join(Z.at[index, 'texts'])

        for index, row in Z.iterrows():
            Z.at[index, 'text'] = Z.at[index, 'text'].lower().replace('.', ' ').replace(',', ' ').replace(';',
                                                                                                          ' ').replace(
                ':',
                ' ').replace(
                '!', ' ').replace('?', ' ').replace('\r', ' ').replace('\n', ' ')

        cleanup_nums = {"age": {"<=17": 1, "18-24": 2, "25-34": 3, "35-44": 4, ">=45": 5},
                        "education": {"low": 1, "middle": 2, "high": 3}}
        Z.replace(cleanup_nums, inplace=True)

        Z.rename(columns={'age': 'y'}, inplace=True)

        train_sels = ~np.isnan(Z.y)
        test_sels = np.isnan(Z.y)

        train_inds = [i for i, x in enumerate(train_sels) if x]
        test_inds = [i for i, x in enumerate(test_sels) if x]

        del train_sels, test_sels

        y_train = Z.y.values[train_inds]

        def extract_features_cv(train_texts, test_texts, ngrams_count, stp_wrds=None, mindf=2, maxdf=0.9):
            self.tfidf = CountVectorizer(ngram_range=(1, ngrams_count), stop_words=stp_wrds, min_df=mindf, max_df=maxdf)
            train_features = self.tfidf.fit_transform(train_texts)
            test_features = self.tfidf.transform(test_texts)
            return train_features, test_features

        tit_train_features_cv, tit_test_features_cv = extract_features_cv(Z["text"].values[train_inds],
                                                                          Z["text"].values[test_inds], 2, 'english', 5,
                                                                          0.9)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        text_train_f = self.vectorizer.fit_transform(Z["text"].values[train_inds])

        train_f = sp.hstack((text_train_f, tit_train_features_cv), format='csr')

        print("training LR\n")

        clf = LogisticRegression(C=1.0)
        clf.fit(train_f, y_train)

        print("dumped LR\n")

        joblib.dump(clf, 'm.pkl')

    def get_age(self, texts):
        model1 = joblib.load('m.pkl')

        df = pd.DataFrame({"q": len(texts), "text": "".join(texts), }, index=[0])

        tit_test_features_cv = self.tfidf.transform(df['text'].values)
        text_test_f = self.vectorizer.transform(df["text"].values)
        test_f = sp.hstack((text_test_f, tit_test_features_cv), format='csr')

        df_t = model1.predict(test_f)
        df__3 = pd.DataFrame({"age": df_t}, index=[0])

        cleanup_nums = {"age": {1.0: "<=17", 2.0: "18-24", 3.0: "25-34", 4.0: "35-44", 5.0: ">=45"}}
        df__3.replace(cleanup_nums, inplace=True)
        """Returns age for author of the input texts
             :param texts: list of texts for processing
             :return: age interval
             """

        return df__3.at[0, "age"]

    def get_education(self, texts):
        return "high"
