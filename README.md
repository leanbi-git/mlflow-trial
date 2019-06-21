# Example MLflow project
To try it:

Clone project:
* git clone git@github.com:leanbi-git/mlflow-trial.git

Save model locally in your path (automatically):
* mlflow run mlflow-trial/ -e 'save_air_quality_model'

Try the saved model on data from a CSV file:
* mlflow models predict -m ./mlflow-trial/my_models/saved_models/air_quality/ -i ./mlflow-trial/data/air_tempture_humidity.csv -t 'csv'

...or from a JSON file:
* mlflow models predict -m ./mlflow-trial/my_models/saved_models/air_quality/ -i ./mlflow-trial/data/air_tempture_humidity.json

Serve the saved model and let it listen to POST messages:
* mlflow models serve -m mlflow-trial/my_models/saved_models/air_quality/

Test from a new console the model running on localhost:5000:
* curl -d '{"columns":["temperature","humidity"],"index":[0,1,2],"data":[[18, 50],[19, 90],[12, 10]]}' -H 'Content-Type: application/json'  localhost:5000/invocations

