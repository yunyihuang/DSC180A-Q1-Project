# DSC180A Section A12 Quarter 1 Project
This is the repository for our DSC180A section's Quarter 1 Project, which consists of 2 machine learning models that can be used to predict the relevance and sentiment toward China of the tweets posted by the members of the U.S. Congress, given the tweet's text content.

## Main Content
- __config__: folder to store tuned parameters for the models
  - `param-A.json` - tuned parameter for task 1 model
  - `param-B.json` - tuned parameter for task 2 model
- __data__: folder to store data, including test data and other data. It is also used to store the results data for the author
  - test: folder to store the test data
  - raw: empty, folder to store the raw data
- __notebook__: folder to store the pre-development notebooks
  - `DSC180A_sketch.ipynb` - all the pre-development code for the tasks
- __src__: folder to store the files of obtaining the dataset, building the features, and the code for the 2 models
  - `make_dataset.py` - script to preprocess the raw data
  - `build_features.py` - script to build features for different tasks
  - `relevance_bucket_classifier.py` - code for the task 1 model
  - `sentiment_score_regressor.py`- code for the task 2 model

## Data Source
The data used in this project was provided by the staffs from the China Data Lab at UC San Diego. Click [here](https://drive.google.com/drive/folders/1VSYdGh12UNVNhfxbSeHRdANvHr5xF8Ea?usp=sharing) for data. If running the models with the raw data, please place the `SentimentLabeled_10112022.csv` in the folder `data/raw`.

## Important Files
- `Dockerfile`: contains the information for building the docker image
- `run.py`: the script to run the models. To run the models on test data, use the following command: 
  - `python3 run.py test`