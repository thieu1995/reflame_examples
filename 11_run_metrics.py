#!/usr/bin/env python
# Created by "Thieu" at 08:26, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd


def get_metrics(data_name, pathfile, model_names):

    train_dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{pathfile}/{model_name}-train-metrics.csv")
        df["model"] = model_name
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)

    test_dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{pathfile}/{model_name}-test-metrics.csv")
        df["model"] = model_name
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df.to_csv(f"history/{data_name}-train.csv", index=False)
    test_df.to_csv(f"history/{data_name}-test.csv", index=False)


model_names = ["SVC", "DTC", "RFC", "GBC", "MLP", "FLNN",
               "AVOA-FLNN", "ARO-FLNN", "RUN-FLNN", "INFO-FLNN", "TLO-FLNN", "SHADE-FLNN"]

model_names_rr = ["SVR", "DTR", "RFR", "GBR", "MLP", "FLNN",
               "AVOA-FLNN", "ARO-FLNN", "RUN-FLNN", "INFO-FLNN", "TLO-FLNN", "SHADE-FLNN"]

get_metrics(data_name="BreastEW", pathfile="history/BreastEW", model_names=model_names)

get_metrics(data_name="Heart", pathfile="history/heart", model_names=model_names)

get_metrics(data_name="Iris", pathfile="history/Iris", model_names=model_names)

get_metrics(data_name="Wine", pathfile="history/Wine", model_names=model_names)

get_metrics(data_name="Banknote", pathfile="history/banknote", model_names=model_names)

get_metrics(data_name="MovieRevenue", pathfile="history/MovieRevenue", model_names=model_names_rr)

get_metrics(data_name="Concrete", pathfile="history/Concrete", model_names=model_names_rr)

get_metrics(data_name="Energy", pathfile="history/Energy", model_names=model_names_rr)

get_metrics(data_name="Abalone", pathfile="history/Abalone", model_names=model_names_rr)

get_metrics(data_name="RealEstate", pathfile="history/RealEstate", model_names=model_names_rr)
