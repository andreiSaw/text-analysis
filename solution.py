# Template solution for Text Processing Course 2017
# @author Denis Turdakov (turdakov@ispras.ru)
import pandas as pd
import numpy as np
from matplotlib.mlab import find
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as RFC


# import os


class Solution:  # Class must have name "Solution"
    def __init__(self):
        """Constructor
        :return: None
        """

    def train(self, training_corpus):
        """Trainer of classifiers.
        This function runs before get_age and get_education. Then mentioned functions can use trained model.
        Save trained model and load it here if you want to save time for training.
        Training corpus is a list of json-objects with the following fields:
            id: internal author's id in the dataset [mandatory]
            texsts: list of author's texts (list of strings) [mandatory]
            age: author's age (string representing age interval) [optional]
            education: author's education level (string representing age interval) [optional]
        :param training_corpus: trainings corpus
        :return: None
        """
        # if not (os.path.exists("m.pkl")):
        train_df = pd.DataFrame.from_dict(training_corpus)
        np.random.seed(0)
        df_t = train_df
        df_t = df_t.sample(frac=1.0).reset_index(drop=True)
        cleanup_nums = {"age": {"<=17": 1, "18-24": 2, "25-34": 3, "35-44": 4, ">=45": 5},
                        "education": {"low": 1, "middle": 2, "high": 3}}
        df_t.replace(cleanup_nums, inplace=True)
        df_t.education = df_t.education.fillna(0)
        df_t.loc[:, 'f'] = pd.Series(np.random.randn(len(df_t)), index=df_t.index)
        for index, row in df_t.iterrows():
            df_t.at[index, 'f'] = len(df_t.at[index, 'texts'])
            df_t.at[index, 'text'] = ''.join(df_t.at[index, 'texts'])
        Z = df_t.drop(["id", "education"], axis=1)

        Z.rename(columns={'age': 'y'}, inplace=True)

        train_sels = ~np.isnan(Z.y)
        test_sels = np.isnan(Z.y)

        train_inds = find(train_sels)
        test_inds = find(test_sels)

        del train_sels, test_sels

        corpus = Z.text
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(corpus)

        Y = Z.y.values

        # Initialize the Random Forest or bagged tree based the model chosen
        rfc = RFC(n_estimators=100, oob_score=True, max_features="auto")
        print("Training %s" % ("Random Forest"))
        rfc = rfc.fit(X[train_inds], Y[train_inds])
        print("OOB Score =", rfc.oob_score_)

        joblib.dump(rfc, 'm.pkl')
        # else:
        #    print("exists")

    def get_age(self, texts):
        model1 = joblib.load('m.pkl')
        df = pd.DataFrame({"f": len(texts), "text": "".join(texts), }, index=[0])

        corpus = df.text
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(corpus)

        df_t = model1.predict(X)

        cleanup_nums = {"age": {1: "<=17", 2: "18-24", 3: "25-34", 4: "35-44", 5: ">=45"}}
        df_t.replace(cleanup_nums, inplace=True)
        """Returns age for author of the input texts
             :param texts: list of texts for processing
             :return: age interval
             """
        return df_t[0]
        # return "25-34"

    def get_education(self, texts):
        """Returns education of author of the input texts
        :param texts: list of texts for processing
        :return: education
        """
        return "high"
