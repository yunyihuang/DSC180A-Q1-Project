from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from src.make_dataset import *
from src.build_features import *

def SS_regressor(filepath, gb_params, en_params):
    """model for the second task

    Args:
        filepath (str): filepath of the raw data file
        gb_params (dict): tuned parameters for Gradient Boosting
        en_params (dict): tuned parameters for Elastic Net

    Returns:
        dict: dictionary of models' performance
    """
    data = load_data(filepath)
    X, y = get_features(data, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = [GradientBoostingRegressor(**gb_params), 
              LGBMRegressor(),
              ElasticNet(**en_params)]

    preprocessor = ColumnTransformer(
        transformers=[
            ("bog", CountVectorizer(), 'cleaned_text'),
            ("tfidf", TfidfVectorizer(), 'cleaned_text')]
    )

    res = {}
    for model in models:
        model_name = type(model).__name__
        
        pl = Pipeline([
                    ('preprocessor', preprocessor),
                    ('reg', model),
                ])
        
        pl.fit(X_train, y_train)
        preds = pl.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        model_res = {'Mean Squared Error':mse}
        res[model_name] = model_res

    return res