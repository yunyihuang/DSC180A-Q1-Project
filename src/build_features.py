import numpy as np
import pandas as pd
from sklearn.utils import resample
from nltk.corpus import stopwords

def process_text(text):
    """Function to preprocess the text content of a tweet

    Args:
        text (str): the text content of a tweet

    Returns:
        str: processed text content
    """
    # if text is Nan
    if type(text)==np.float: 
        return ""
    
    # uniform format
    temp = text.lower().replace("\n","")
    # remove hyperlinks
    temp = ' '.join([x for x in temp.split() if not ('http' in x or 'www' in x)]).split()

    # filtering special characters
    special_chars = list("~!@#$%^&*()_+-={}[]\|<>?.,;:`")
    res = []
    for word in temp:
        for i in special_chars:
            word = word.replace(i,'')
        res.append(word)

    # remove stopwords
    sw = stopwords.words('english')
    res = [i for i in res if i not in sw]

    return " ".join(res)


def get_features(data, task, raw):
    """Function to get X and y for the corresponding task 

    Args:
        data (pd.DataFrame): cleaned and processed data
        task (int): 1 for relevance_bucket_classifier, 2 for sentiment_score_regressor

    Returns:
        pd.DataFrame: X, y for model training
    """
    # select China-related tweets only
    df = data[data['country']=='China']
    df['cleaned_text'] = df['text'].apply(process_text)

    # relevance_bucket_classifier
    if task == 1:
        df_bkt = df[['cleaned_text', 'cleaned_buckets']]
        df_bkt.drop_duplicates(inplace=True, ignore_index=True)
        # balance class
        df_bkt_A = df_bkt[df_bkt.cleaned_buckets==1]
        df_bkt_B = df_bkt[df_bkt.cleaned_buckets==0]

        # only replace the sample data when not using the raw data file
        if raw:
            replace = False
            N = 3000
        else:
            replace = True
            N = 10

        df_bkt_A_sample = resample(df_bkt_A, replace=replace, n_samples=N, random_state=0)
        df_bkt_B_sample = resample(df_bkt_B, replace=replace, n_samples=N, random_state=0)
        df_bkt_final = pd.concat([df_bkt_A_sample, df_bkt_B_sample])
        # split data
        X = df_bkt_final.cleaned_text.to_frame()
        y = df_bkt_final.cleaned_buckets.to_frame()
    
    # sentiment_score_regressor
    if task == 2:
        df_ss = df[['cleaned_text', 'SentimentScore']]
        df_ss.drop_duplicates(inplace=True, ignore_index=True)
        # extra cleaning
        df_ss.dropna(inplace=True)
        df_ss = df_ss[df_ss['SentimentScore']<=5]
        df_ss['SentimentScore'] = df_ss['SentimentScore'].astype(float)
        # split data
        X = df_ss.cleaned_text.to_frame()
        y = df_ss.SentimentScore.to_frame()
    
    return X, y