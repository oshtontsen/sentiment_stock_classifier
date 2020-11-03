
class Randomizer:
    def __init__(self, data_root: str, split_date: str):
        """Partitions the data into training/test sets."""
        # Define variables that have not been specified.
        self.df = None
        self.train = None
        self.test = None

        self.data_root = data_root
        self.split_date = split_date
        self.data_list = self._get_data_list()

    def _get_data_list(self, remove_nans=True):
        """
        Clean the dataframe prior to dividing into train and
        test sets. Doing so will ensure that the train and 
        test sets are in a consistent format for the classifier.
        """
        self.df = pd.read_csv(self.data_root, encoding="ISO-8859-1")
        print("[INFO]: Data successfully loaded as data frame.")

        # The first two columns of the dataset contains the labels
        # and the dates, while the other columns contains headlines.
        features = self.df.iloc[:, 2:27]

        # Replace all numbers and punctuation found in text with spaces.
        features.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

        # Rename the dataset column headers 
        cols = [str(i) for i in range(25)]
        features.columns = cols

        # Lower-case all text in each dataset column.
        for col in cols:
            features[col] = features[col].str.lower()

        # Concatenate all 25 top headlines for each date into a paragraph.
        # NOTE: A few of the headlines contain nan values.
        data_list = []
        if remove_nans:
            for row in range(len(features.index)):
                headline = "" 
                for word in features.iloc[row, 0:25]:
                    if not isinstance(word, str):
                        continue 
                    else:
                        headline += " " + word
                data_list.append(headline)
        else:
            for row in range(len(features.index)):
                # The data set contains nan's, so they must be converted 
                # into strings.
                # TODO: Try removing nan's from the dataset rather than 
                # converting them in strings.
                data_list.append(' '.join(str(i) for i in features.iloc[row, 0:25]))
        return data_list 

    def simple_label_partition(self):
        for i in range(len(self.df['Date'])):
            if self.df['Date'][i] == self.split_date:
                split_idx = i
                break

        """Partitions data by prespecified date."""
        # The training data will consist of data before 2014-12-31.
        self.train = self.data_list[:split_idx] 
        # The test data will consist of data after 2014-12-31.
        self.test = self.data_list[split_idx:]

def setup_randomforest_randomizer(data_root: str, split_idx: str):
    randomizer = Randomizer(data_root, split_idx)
    randomizer.simple_label_partition()
    return randomizer
