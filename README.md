New Instructions:

conda activate py3
python3 scikit_classifiers.py -m MLP -d heart-statlog 

To fetch a repo from git:

Go to the repo - click on Fork
Then copy the HTTTP git link 
then do - git clone <link>
once you make changes in the repo locally, 

Go to git button on left
click on + of files you want to stage
then click on tick to commit those files
it'll ask you for commit message
then click on ... next to tick- click on push
then might ask you for username and password
enter and done!


# IFT6390 - Final Project

This project is about Interpretable Machine Learning in the Low-Data Regime.

## Authors 

- Isabelle Bouchard 
- Carolyne Pelletier

## Prerequisites

```
Python 3.5 +
```

## Setup

Install the Python dependencies in a virtual environment
```
python -m venv venv  # Create the virtual env
source venv/bin/activate  # Activate it
pip install -r requirements.txt  # Install the requirements
```


### Data
- Download the [Heart Disease dataset](https://datahub.io/machine-learning/heart-statlog#data) unzip it in `data/heart-statlog`.
- Download the [Cervical Cancer dataset](https://datahub.io/machine-learning/cervical-cancer#data) and unzip it in `data/cervical-cancer`. 


### Available Classifiers
- MLP
- Random Forest
- Logistic Regression


## Running the code

`scikit_classifiers.py` is a script that performs grid search on a for a 
classifier given a dataset. 
```
python scikit_classifiers.py -m <MODEL_NAME> -d <DATASET_NAME> --help 
```

The repo also include a few useful notebooks to help understand the data, 
visualize grid search and do feature selection.
- `data_exploration.ipynb`
- `grid_search.ipynb`
- `feature_selection.ipynb`

Those notebooks have been used to produce the figures shown in the report.


## Results 

Some raw results have been pushed in `HP_results.md`
