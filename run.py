import os
import sys
import json
import pandas as pd 
from baselinemodel import build_model

if __name__ == '__main__':
    target = sys.argv[1]
    filepath = os.path.join(target, 'data.csv')
    df = pd.read_csv(filepath)

    with open('config/config.json') as configfile:
        params = json.load(configfile)

    build_model(df, **params)