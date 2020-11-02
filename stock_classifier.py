import os
import pandas as pd

class Randomizer:
    def __init__(self, data_root: str, split_idx: str):
        """Partitions the data into training/test sets."""
        self.data_root = data_root
        self.split_idx = split_idx          

        # Define variables that have not been specified
        self.df = None
        self.features = None
        self.train = None
        self.test = None

    def preprocess_data(self):
        """
        Clean the dataframe prior to dividing into train and
        test sets. Doing so will ensure that the train and 
        test sets are in a consistent format for the classifier.
        """
        self.df = pd.read_csv(self.data_root, encoding="ISO-8859-1")
        print("[INFO]: Data successfully loaded as data frame.")

        # The first two columns of the dataset contains the labels
        # and the dates, while the other columns contains headlines.
        self.features = self.df.iloc[:, 2:27]

        # Replace all numbers and punctuation found in text with spaces.
        self.features.replace("^a-zA-Z", " ", regex=True, inplace=True)

        # Rename the dataset column headers 
        cols = [str(i) for i in range(25)]
        self.df.columns = cols

        # Lower-case all text in each dataset column
        for col in cols:
            self.df[col] = df[col].lower()

        # Concatenate all 25 top headlines for each date into a paragraph
        headlines = []
        for row in range(len(data.index)):
            headlines.append(' '.join(i for i in data.iloc[row, 0:25]))

    def simple_label_partition(self):
        """Partitions data by prespecified date."""
        # The training data will consist of data before 2014-12-31
        self.train = df[df['Date'] <= self.split_idx] 
        # The test data will consist of data after 2014-12-31
        self.test = df[df['Date'] > self.split_idx]


def main():
    DATA_ROOT = 'Data.csv'
    N_GRAM_RANGE = (2, 2)
    N_ESTIMATORS = 200
    SPLIT_IDX = '20141231'

    # Load data 
    if not os.path.exists(DATA_ROOT):
        raise OSError("[ERROR]: Data root directory does not exist.")
    
    # Build the randomizer and sampler
    data_randomizer = Randomizer(DATA_ROOT, SPLIT_IDX, N_GRAM_RANGE) 

    # Preprocess data
    # Fit data into count vectorizer


    # Define the metrics object

    # Train the classifier


if __name__ == "__main__":
    main()
