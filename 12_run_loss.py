#!/usr/bin/env python
# Created by "Thieu" at 08:41, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import matplotlib.pyplot as plt


def draw_loss(data_name, pathfile, model_names, verbose=False):
    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{pathfile}/{model_name}-loss.csv")
        df['Model'] = model_name
        dfs.append(df)
    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs, ignore_index=True)
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(f"history/{data_name}-loss.csv", index=False)

    # Plot the loss for all models in a single figure
    plt.figure(figsize=(8, 6))
    for model_name, group in merged_df.groupby('Model'):
        plt.plot(group['epoch'], group['loss'], label=model_name)

    plt.xlabel('Epoch')
    plt.ylabel('Fitness value')
    plt.title(f"The fitness value of Metaheuristic-trained FLNN models on {data_name} data.")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"history/{data_name}-loss.png", bbox_inches='tight')
    if verbose:
        plt.show()


model_names = ["AVOA-FLNN", "ARO-FLNN", "RUN-FLNN", "INFO-FLNN", "TLO-FLNN", "SHADE-FLNN"]


draw_loss(data_name="BreastEW", pathfile="history/BreastEW", model_names=model_names)
draw_loss(data_name="Heart", pathfile="history/heart", model_names=model_names)
draw_loss(data_name="Iris", pathfile="history/Iris", model_names=model_names)
draw_loss(data_name="Wine", pathfile="history/Wine", model_names=model_names)
draw_loss(data_name="Banknote", pathfile="history/banknote", model_names=model_names)
draw_loss(data_name="MovieRevenue", pathfile="history/MovieRevenue", model_names=model_names)
draw_loss(data_name="Concrete", pathfile="history/Concrete", model_names=model_names)
draw_loss(data_name="Energy", pathfile="history/Energy", model_names=model_names)
draw_loss(data_name="Abalone", pathfile="history/Abalone", model_names=model_names)
draw_loss(data_name="RealEstate", pathfile="history/RealEstate", model_names=model_names)
