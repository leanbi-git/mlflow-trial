# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np

# module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/my_modelsjj/air_quality')
module_path = './my_models/code/air_quality'

print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)
from AirQuality import AirQuality
    
import mlflow
import mlflow.sklearn


def eval_metrics(air_quality_data):
    n = air_quality_data.shape[0]
    return air_quality_data['AQ_good'].sum()/n, air_quality_data['AQ_acceptable'].sum()/n


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    measurements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "air_temperature_humidity.csv")
    data = pd.read_csv(measurements_path)

#     print(data.columns)
    
    with mlflow.start_run():
        aq = AirQuality()

        predicted_qualities = aq.predict(None, data)

        (percent_good, percent_acceptable) = eval_metrics(predicted_qualities)

        print("AirQuality model:")
        print("  percent_good: %s" % percent_good)
        print("  percent_acceptable: %s" % percent_acceptable)

#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("l1_ratio", l1_ratio)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)
#         mlflow.log_metric("mae", mae)
#         mlflow.sklearn.log_model(lr, "model")
