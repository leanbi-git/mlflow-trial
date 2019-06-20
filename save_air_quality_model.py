# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np

module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_models', 'code', 'air_quality')
# module_path = './my_models/code/air_quality'

print()
print('loading model code from folder: ', module_path)
print()

if module_path not in sys.path:
    sys.path.append(module_path)
from AirQuality import AirQuality
    
import mlflow

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_models', 'air_quality')
    
    print('saving model in folder: ', model_save_path)
    
    with mlflow.start_run():
        aq = AirQuality()

        mlflow.pyfunc.save_model(path=model_save_path, python_model=aq)
