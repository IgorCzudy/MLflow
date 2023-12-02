import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

if __name__ == '__main__':
    df = pd.read_csv('word-happiness-raport-2021.csv')
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Ladder score'], axis=1), df['Ladder score'], test_size=0.2, random_state=42)

    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    score = dt.score(X_train, y_train)
    print(f"Score: {score}")

    y_pred = dt.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("score", score)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(dt, "model")

    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")
