import os
import sys
import json
from src.make_dataset import *
from src.build_features import *
from src.relevance_bucket_classifier import *
from src.sentiment_score_regressor import *
import warnings

def main(targets):
    try:
        # check if running using raw data or test data
        if 'test' in targets:
            filepath = os.path.join('data/test', 'data.csv')
            raw = False
        elif 'raw' in targets:
            filepath = os.path.join('data/raw', 'SentimentLabeled_10112022.csv')
            raw = True
        print(filepath)
        # import the tuned parameters
        with open('config/param-A.json') as fh:
                paramA = json.load(fh)
        with open('config/param-B.json') as fh:
                paramB = json.load(fh)

        # run models and get the results
        res_task_1 = RB_classifier(filepath, paramA['svc'], paramA['rf'], raw=raw)
        res_task_2 = SS_regressor(filepath, paramB['gb'], paramB['en'], raw=raw)

        return [res_task_1, res_task_2]

    except Exception as e:
        print(e)
    

if __name__ == '__main__':
    targets = sys.argv[1]
    warnings.filterwarnings('ignore')
    res = main(targets)
    print(res)