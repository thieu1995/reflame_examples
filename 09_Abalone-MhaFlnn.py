#!/usr/bin/env python
# Created by "Thieu" at 17:56, 06/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# https://archive.ics.uci.edu/dataset/1/abalone
from pathlib import Path
from concurrent import futures as parallel
from ucimlrepo import fetch_ucirepo
from reflame import Data, MhaFlnnRegressor


EPOCH = 750
POP_SIZE = 20
TEST_SIZE = 0.2
MAX_WORKERS = 6
LIST_METRICS = ("MAE", "RMSE", "MAPE", "NNSE", "R2")
DATA_NAME = "Abalone"
PATH_SAVE = f"history/{DATA_NAME}"
Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

## Load data object
df = fetch_ucirepo(id=1)
X = df.data.features.values[:, 1:]
y = df.data.targets.values
data = Data(X, y, name=DATA_NAME)

## Split train and test
data.split_train_test(test_size=TEST_SIZE, random_state=42, inplace=True, shuffle=True)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("minmax"))
data.y_test = scaler_y.transform(data.y_test)

# list_optimizers = ("OriginalAVOA", "OriginalARO", "OriginalRUN", "OriginalINFO", "OriginalTLO", "OriginalSHADE")
list_optimizers = ("OriginalTLO", "OriginalSHADE")
list_paras = [
    # {"name": "AVOA-FLNN", "epoch": EPOCH, "pop_size": POP_SIZE},
    # {"name": "ARO-FLNN", "epoch": EPOCH, "pop_size": POP_SIZE},
    # {"name": "RUN-FLNN", "epoch": EPOCH, "pop_size": POP_SIZE},
    # {"name": "INFO-FLNN", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "TLO-FLNN", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "SHADE-FLNN", "epoch": EPOCH, "pop_size": POP_SIZE},
]

def train_and_evaluate_optimizer(opt, paras):
    ## Create model
    model = MhaFlnnRegressor(expand_name="chebyshev", n_funcs=3, act_name="tanh", obj_name="MSE",
                             optimizer=opt, optimizer_paras=paras, verbose=False)

    ## Train the model
    model.fit(X=data.X_train, y=data.y_train)

    ## Test the model
    y_pred_train = model.predict(data.X_train)
    y_pred_test = model.predict(data.X_test)

    ## Save metrics
    model.save_metrics(data.y_train, y_pred_train, list_metrics=LIST_METRICS,
                       save_path=PATH_SAVE, filename=f"{model.optimizer.name}-train-metrics.csv")
    model.save_metrics(data.y_test, y_pred_test, list_metrics=LIST_METRICS,
                       save_path=PATH_SAVE, filename=f"{model.optimizer.name}-test-metrics.csv")

    ## Save loss train
    model.save_loss_train(save_path=PATH_SAVE, filename=f"{model.optimizer.name}-loss.csv")

    ## Save model
    model.save_model(save_path=PATH_SAVE, filename=f"{model.optimizer.name}-model.pkl")
    print(f"Completed processing for optimizer {opt}")


## Run in parallel using ProcessPoolExecutor
if __name__ == "__main__":
    with parallel.ProcessPoolExecutor(MAX_WORKERS) as executor:
        # Submit each optimizer configuration to be processed in parallel
        futures = [executor.submit(train_and_evaluate_optimizer, opt, list_paras[idx]) for idx, opt in enumerate(list_optimizers)]

        # Optionally: wait for all futures to complete (gathers all results)
        for future in futures:
            future.result()  # Blocking call, also allows capturing exceptions if any

    print(f"All optimizers have been processed for data: {DATA_NAME}")
