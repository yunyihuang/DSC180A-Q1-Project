import pandas as pd

def clean_buckets(bucket_num):
    """Helper function to clean buckets in to binary classes

    Args:
        bucket_num (string): bucket number in the raw data file

    Returns:
        int: 1 for Bucket 1 and 0 for Non-Bucket 1
    """
    if bucket_num == "1" or bucket_num == "1.0":
        return 1 # bucket1
    else:
        return 0 # non-bucket1


def get_data(filepath):
    """get dataframe from csv

    Args:
        filepath (string): filepath of the raw data file

    Returns:
        pd.DataFrame: df of the raw data file
    """
    raw_data = pd.read_csv(filepath)
    return raw_data


def clean_data(data):
    """Main cleaning function for the raw data file

    Args:
        data (pd.DataFrame): dataframe of the raw data file

    Returns:
        pd.DataFrame: df of the cleaned data 
    """
    # drop the duplicate data 
    data.drop_duplicates(inplace=True)

    # uniform data format
    data["term_partisanship"] = data["term_partisanship"].str.strip("{}")
    data["term_type"] = data["term_type"].str.strip("{}")
    data["term_state"] = data["term_state"].str.strip("{}")

    # feature engineering for data visualization only
    data['text_length'] = data['text'].apply(lambda x: len(x.split()))
    data['cleaned_buckets'] = data['Bucket'].apply(clean_buckets)
    data[['date','birth']] = data[['date','birth']].apply(pd.to_datetime)

    # filtering abnormal data
    data.reset_index(drop=True, inplace=True)
    return data

def load_data(filepath):
    """Function to load the cleaned data

    Args:
        filepath (string): filepath of the raw data file

    Returns:
        pd.DataFrame: df of the cleaned data 
    """
    raw_data = get_data(filepath)
    data = clean_data(raw_data)
    return data