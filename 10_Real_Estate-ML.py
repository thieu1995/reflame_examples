#!/usr/bin/env python
# Created by "Thieu" at 18:17, 06/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from ucimlrepo import fetch_ucirepo
from concurrent import futures as parallel
from reflame import Data, FlnnRegressor
from utils.helper import save_metrics, save_model
from pathlib import Path


TEST_SIZE = 0.2
MAX_WORKERS = 6
LIST_METRICS = ("MAE", "RMSE", "MAPE", "NNSE", "R2")
DATA_NAME = "RealEstate"
PATH_SAVE = f"history/{DATA_NAME}"
Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

## Load data object
df = fetch_ucirepo(id=477)
X = df.data.features.values[:,1:]
y = df.data.targets.values
data = Data(X, y, name=DATA_NAME)

## Split train and test
data.split_train_test(test_size=TEST_SIZE, random_state=42, inplace=True, shuffle=True)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("minmax"))
data.y_test = scaler_y.transform(data.y_test)

regressors = {
    "SVR": SVR(kernel="rbf", C=0.75),
    "DTR": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RFR": RandomForestRegressor(max_depth=5, n_estimators=30, max_features=3, random_state=42),
    "GBR": GradientBoostingRegressor(n_estimators=75, learning_rate=0.5, max_depth=3, random_state=42),
    "MLP": MLPRegressor(alpha=1, max_iter=750, hidden_layer_sizes=(25,), activation="relu", random_state=42),
    "FLNN": FlnnRegressor(expand_name="laguerre", n_funcs=3, act_name="tanh",
                          obj_name="MSE", max_epochs=750, batch_size=32, optimizer="SGD", verbose=False)
}

def train_and_evaluate_model(name, model):
    ## Train the model
    model.fit(X=data.X_train, y=data.y_train)

    ## Test the model
    y_pred_train = model.predict(data.X_train)
    y_pred_test = model.predict(data.X_test)

    ## Save metrics
    save_metrics(problem="regression", y_true=data.y_train, y_pred=y_pred_train, list_metrics=LIST_METRICS,
                 save_path=PATH_SAVE, filename=f"{name}-train-metrics.csv")
    save_metrics(problem="regression", y_true=data.y_test, y_pred=y_pred_test, list_metrics=LIST_METRICS,
                 save_path=PATH_SAVE, filename=f"{name}-test-metrics.csv")

    ## Save model
    save_model(model=model, save_path=PATH_SAVE, filename=f"{name}-model.pkl")
    print(f"Completed processing for classifier {name}")


if __name__ == "__main__":
    with parallel.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each classifier for parallel processing
        futures = [executor.submit(train_and_evaluate_model, name, model) for name, model in regressors.items()]

        # Optionally, wait for all tasks to complete
        for future in futures:
            future.result()  # Blocking call to ensure each task completes and handle any exceptions

    print("All regressors have been processed.")
