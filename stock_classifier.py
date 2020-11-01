import os
import pandas as pd


def main():
    DATA_ROOT = 'Data.csv'

    # Load data 
    if not os.path.exists(DATA_ROOT):
        raise OSError("[ERROR]: Data root directory does not exist.")
    
    df = pd.read_csv(DATA_ROOT, encoding="ISO-8859-1")
    print("[INFO]: Data successfully loaded as data frame.")
    # Preprocess data
    # Fit data into count vectorizer

if __name__ == "__main__":
    main()
