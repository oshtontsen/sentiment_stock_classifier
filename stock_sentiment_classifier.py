import os
import ss_baseline
import ss_randomizer 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

class BaselineClassifier():
    def __init__(self, n_estimators):
        self.countvector = self._get_countvector()
        self.ss_classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')

    def _get_countvectorizer(self):
        return Countvectorizer(ngram_range(ngram_range[0], ngram_range[1]))

def build_graph(n_estimators):
    return BaselineClassifier(n_estimators)

def train_step():
    pass


def main():
    DATA_ROOT = 'Data.csv'
    N_GRAM_RANGE = (2, 2)
    N_ESTIMATORS = 200
    SPLIT_IDX = '2014-12-31'

    # Load data 
    if not os.path.exists(DATA_ROOT):
        raise OSError("[ERROR]: Data root directory does not exist.")
    
    # Build the randomizer and sampler
    data_randomizer = ss_randomizer.setup_randomforest_randomizer(DATA_ROOT, SPLIT_IDX) 

    # Fit data into count vectorizer
    model = build_graph(N_ESTIMATORS)
    X_train = model.countvector.fit_transform(data_randomizer.train_features)
    X_test = model.countvector.fit_transform(data_randomizer.test_features)

    # Train the classifier
    model.ss_classifier.fit(X_train, data_randomizer.train_labels)

    # Perform inference
    predictions = model.ss_classifier.predict(X_test)

    # Define the metrics object


if __name__ == "__main__":
    main()
