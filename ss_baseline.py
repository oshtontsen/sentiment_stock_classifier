from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

class BaselineClassifier():
    def __init__(self, n_estimators, ngram_range):
        self.countvector = self._get_countvectorizer(ngram_range)
        self.ss_classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')

    def _get_countvectorizer(self, ngram_range):
        return CountVectorizer(ngram_range=(ngram_range[0], ngram_range[1]))

def build_graph(n_estimators, ngram_range):
    return BaselineClassifier(n_estimators, ngram_range)

