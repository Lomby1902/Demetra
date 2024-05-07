import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, PredictionErrorDisplay


path = 'SAMPLE_DATA_SET.xlsx'

# Read and load dataset
df = pd.read_excel(path, sheet_name=[0, 1])
X = df.get(0)
X = (X.iloc[:, 1:]).values
X = X.T
print(X.shape)

Y = df.get(1)
Y = (Y.iloc[:, :]).values
print(Y.shape)




# Ripartition in training and test and
X_train, X_test, Y_train, Y_test = train_test_split(X, Y[:, 0], test_size=0.3, random_state=42)
regr = LinearRegression()


# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred_regr_svd = regr.predict(X_test)



# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Demetra Test")

# Start an MLflow run
with mlflow.start_run():


    # Log the loss metric
    mlflow.log_metric("MSE", mean_squared_error(Y_test, Y_pred_regr_svd))


    # Infer the model signature
    signature = infer_signature(X_train, regr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=regr,
        artifact_path="Demetra",
        signature=signature,
        input_example=X_train,
        registered_model_name="Demetra Test",
    )
