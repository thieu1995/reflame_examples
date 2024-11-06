#!/usr/bin/env python
# Created by "Thieu" at 17:15, 06/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metacluster import get_dataset
from reflame import Data, FlnnClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from concurrent import futures as parallel
from utils.helper import save_metrics, save_model
from utils.visualizer import draw_confusion_matrix
from pathlib import Path


TEST_SIZE = 0.2
MAX_WORKERS = 6
LIST_METRICS = ("AS", "PS", "NPV", "RS", "F1S")
DATA_NAME = "Iris"
PATH_SAVE = f"history/{DATA_NAME}"
Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

## Load data object
df = get_dataset(DATA_NAME)
data = Data(df.X, df.y, name=df.name)

## Split train and test
data.split_train_test(test_size=TEST_SIZE, random_state=2, inplace=True, shuffle=True)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

classifiers = {
    "SVC": SVC(kernel="linear", C=0.1, random_state=42),
    "DTC": DecisionTreeClassifier(max_depth=4, random_state=42),
    "RFC": RandomForestClassifier(max_depth=4, n_estimators=15, max_features=1, random_state=42),
    "GBC": GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=42),
    "MLP": MLPClassifier(alpha=1, max_iter=750, hidden_layer_sizes=(10,), activation="relu", random_state=42),
    "FLNN": FlnnClassifier(expand_name="chebyshev", n_funcs=3, act_name="softmax",
                   obj_name="CEL", max_epochs=750, batch_size=8, optimizer="SGD", verbose=False)
}

def train_and_evaluate_model(name, model):
    ## Train the model
    model.fit(X=data.X_train, y=data.y_train)

    ## Test the model
    y_pred_train = model.predict(data.X_train)
    y_pred_test = model.predict(data.X_test)

    ## Save metrics
    save_metrics(problem="classification", y_true=data.y_train, y_pred=y_pred_train, list_metrics=LIST_METRICS,
                 save_path=PATH_SAVE, filename=f"{name}-train-metrics.csv")
    save_metrics(problem="classification", y_true=data.y_test, y_pred=y_pred_test, list_metrics=LIST_METRICS,
                 save_path=PATH_SAVE, filename=f"{name}-test-metrics.csv")

    ## Save confusion matrix
    draw_confusion_matrix(data.y_train, y_pred_train, title=f"Confusion Matrix of {name} on training set",
                          pathsave=f"{PATH_SAVE}/{name}-train-cm.png")
    draw_confusion_matrix(data.y_test, y_pred_test, title=f"Confusion Matrix of {name} on testing set",
                          pathsave=f"{PATH_SAVE}/{name}-test-cm.png")

    ## Save model
    save_model(model=model, save_path=PATH_SAVE, filename=f"{name}-model.pkl")
    print(f"Completed processing for classifier {name}")


if __name__ == "__main__":
    with parallel.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each classifier for parallel processing
        futures = [executor.submit(train_and_evaluate_model, name, model) for name, model in classifiers.items()]

        # Optionally, wait for all tasks to complete
        for future in futures:
            future.result()  # Blocking call to ensure each task completes and handle any exceptions

    print("All classifiers have been processed.")
