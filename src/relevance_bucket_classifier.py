from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.make_dataset import *
from src.build_features import *

def RB_classifier(filepath, svc_params, rf_params, raw):
    """model for the first task

    Args:
        filepath (str): filepath of the raw data file
        svc_params (dict): tuned parameters for SVC
        rf_params (dict): tuned parameters for Random Forest 

    Returns:
        dict: dictionary of models' performance
    """
    data = load_data(filepath)
    X, y = get_features(data, 1, raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = [SVC(**svc_params), 
              RandomForestClassifier(**rf_params)]

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
                    ('clf', OneVsRestClassifier(model, n_jobs=1)),
                ])
        
        pl.fit(X_train, y_train)
        preds = pl.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        matrix = confusion_matrix(y_test, preds)
        stats = precision_recall_fscore_support(y_test, preds,average='binary')

        model_res = {'Accuracy':accuracy,
                     'Confusion Matrix':matrix.tolist(),
                     'Precision':stats[0],
                     'Recall':stats[1],
                     'F1 score':stats[2]}
        res[model_name] = model_res

    return res