import os
import ss_baseline
import ss_randomizer 
import metrics

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

def build_metrics(predictions, test_labels):
    return Metrics(predictions, test_labels)


def main():
    DATA_ROOT = 'Data.csv'
    NGRAM_RANGE = (2, 2)
    N_ESTIMATORS = 200
    SPLIT_IDX = '2014-12-31'

    # Load data 
    if not os.path.exists(DATA_ROOT):
        raise OSError("[ERROR]: Data root directory does not exist.")
    
    # Build the randomizer and sampler
    data_randomizer = ss_randomizer.setup_randomforest_randomizer(DATA_ROOT, SPLIT_IDX) 

    # Fit data into count vectorizer
    model = build_graph(N_ESTIMATORS, NGRAM_RANGE)
    print("[INFO]: Successfully built new classifier graph")
    X_train = model.countvector.fit_transform(data_randomizer.train_features)
    X_test = model.countvector.transform(data_randomizer.test_features)

    # Train the classifier
    print("[INFO]: Beginning RandomForest training")
    model.ss_classifier.fit(X_train, data_randomizer.train_labels)
    print("[INFO]: Successfully completed RandomForest training")

    # Perform inference
    predictions = model.ss_classifier.predict(X_test)

    # Define the metrics object
    metrics = build_metrics(predictions, data_randomizer.test_labels)
    metrics.print_confusion_matrix()


if __name__ == "__main__":
    main()
