# Template solution for Text Processing Course 2017
# @author Denis Turdakov (turdakov@ispras.ru)

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

    def get_age(self, texts):
        """Returns age for author of the input texts
        :param texts: list of texts for processing
        :return: age interval
        """
        return "25-34"

    def get_education(self, texts):
        """Returns education of author of the input texts
        :param texts: list of texts for processing
        :return: education
        """
        return "high"

