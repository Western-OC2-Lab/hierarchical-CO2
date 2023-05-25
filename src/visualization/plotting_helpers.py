import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error

def plot_correlations(list_correlations):

    fig, ax = plt.subplots(figsize = (15, 8), dpi = 300)
    markers = ["<", "x", "1", "", "|"]
    for idx, feature in enumerate(list_correlations):
        pd.DataFrame(list_correlations[feature]).plot(kind='density', ax = ax, fontsize = 21, 
                                     marker = markers[idx], markevery=15, mew=2)
    plt.xlim(-10)
    plt.ylim(0)
    ax.legend(labels = list(list_correlations.keys()),prop={'size': 15})

    plt.title("Distribution of Lag-time for each parameter using different thresholds", fontsize = 27)

    ax.set_ylabel("Density", fontsize = 23)
    ax.set_xlabel("Lag Index", fontsize = 23)

    plt.tight_layout()
    plt.savefig("distribution of correlations.png")
    plt.savefig("distribution of correlations.eps")


def plot_predictions(predictions, y_values, file_name, title):
    sns.set_theme()
    plt.figure(figsize=(30, 12))
    plt.plot(predictions, 'v', label='predictions', alpha=0.6)
    plt.plot(y_values, 'x', label='values', alpha=0.6)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.xlabel('Timestamp', fontsize=  27)
    plt.ylabel('CO2 Levels', fontsize=  27)
    plt.title(title, fontsize = 32)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

'''
    This function plots the predictions of models with respect to different threshold values. It takes 4 parameters:
    - predictions: the set of predictions of a specific model
    - y_values: the original set of values
    - filename: the directory where the images are saved
    - threshold_values: can be none, which means all the values are returned or a specific threshold whereby the absolute value is considered
'''
def defining_plotting_thresholds(predictions, y_values, file_name, threshold_values = None):
    set_predictions, set_original = None, None
    if threshold_values == None:
        mae = mean_absolute_error(y_values, predictions)

        title = "MAE = {}".format(str(round(mae, 2)))
        plot_predictions(predictions, y_values, file_name, title)

        return mae
    indices = np.where((y_values > threshold_values) | (y_values < -1*threshold_values))[0]
    if len(indices) > 1:
        set_predictions, set_original = predictions[indices], y_values[indices]
        
        mae = mean_absolute_error(set_original, set_predictions)

        title = "MAE = {}".format(str(round(mae, 2)))
        plot_predictions(set_predictions, set_original, file_name, title)

        return mae
    else:
        return "N/A"