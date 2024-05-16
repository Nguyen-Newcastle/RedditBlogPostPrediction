import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error


"""This utility function will take a list of Scikit learn models, training data and training labels, performing
cross validation and calculating metrics(MAE and RMSE) on these models and plotting bar chart to visualize the metrics of different models.
We also compare the models with the mean of the target labels to get an overall picture of the accuracy of trained models."""
def comparing_models_cross_validation_bar_plot(list_of_estimators, X, y, plot_title, cv=5, figsize=(15, 11),
                                               title_fontsize=22, legend_fontsize=16, yaxis_fontsize=18,
                                               xaxis_fontsize=18, annotation_fontsize=16, yticklabels_fontsize=16,
                                               legend_location="upper left", save_path = None):
    list_of_models_scores = {}

    for estimator in list_of_estimators:
        cross_val_score = cross_validate(estimator, X, y, cv=cv,
                                         scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'],
                                         n_jobs=-1)

        list_of_models_scores[f"{estimator[-1].__class__.__name__}"] = [
            -np.mean(cross_val_score["test_neg_mean_absolute_error"]),
            -np.mean(cross_val_score["test_neg_root_mean_squared_error"])]
    metrics = ['MAE', 'RMSE', "Target mean"]

    x = np.arange(len(metrics))  # the label locations
    width = 0.8 / (len(list_of_estimators) + 1)  # the width of the bars

    #Plotting the metrics for different models
    fig, ax = plt.subplots(figsize=figsize)
    for i, (key, values) in enumerate(list_of_models_scores.items()):

        rects = ax.bar(x[:-1] + i * width - width * len(list_of_estimators) / 2, values, width, label=key)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=annotation_fontsize)

    # Add the mean value of y_true as an additional metric
    mean_y_true = np.mean(y)
    # Add the bar for the mean of y_true
    mean_rect = ax.bar(x[-1] - width / 2, mean_y_true, width, label='Mean of target', color='red')
    ax.annotate(f'{mean_y_true:.2f}',
                    xy=(x[-1], mean_y_true),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=annotation_fontsize)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=yaxis_fontsize)
    ax.set_title(plot_title, fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=xaxis_fontsize)
    ax.legend(loc=legend_location, fontsize=legend_fontsize)

    # Set y-axis font size
    ax.tick_params(axis='y', labelsize=yticklabels_fontsize)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


"""This utility function will take a list of Scikit learn models, training data and training labels, predicting on the testing dataset
 and calculating metrics(MAE and RMSE) on these models and plotting bar chart to visualize the metrics of different models.
 We also compare the models with the mean of the target labels to get an overall picture of the accuracy of trained models."""
def comparing_models_test_bar_plot(list_of_estimators, X, y, plot_title, figsize=(15, 11),
                                               title_fontsize=22, legend_fontsize=16, yaxis_fontsize=18,
                                               xaxis_fontsize=18, annotation_fontsize=16, yticklabels_fontsize=16,
                                               legend_location="upper left", save_path = None):
    list_of_models_scores = {}


    for estimator in list_of_estimators:
        train_prediction = estimator.predict(X)
        mae = mean_absolute_error(y, train_prediction)
        rmse = mean_squared_error(y, train_prediction, squared=False)

        list_of_models_scores[f"{estimator[-1].__class__.__name__}"] = [mae, rmse]

    metrics = ['MAE', 'RMSE', 'Target mean']

    x = np.arange(len(metrics))  # the label locations
    width = 0.8 / (len(list_of_estimators) + 1)  # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)
    for i, (key, values) in enumerate(list_of_models_scores.items()):

        rects = ax.bar(x[:-1] + i * width - width * len(list_of_estimators) / 2, values, width, label=key)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=annotation_fontsize)

    # Add the mean value of y_true as an additional metric
    mean_y_true = np.mean(y)
    # Add the bar for the mean of y_true
    mean_rect = ax.bar(x[-1] - width / 2, mean_y_true, width, label='Mean of target', color='red')
    ax.annotate(f'{mean_y_true:.2f}',
                    xy=(x[-1], mean_y_true),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=annotation_fontsize)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=yaxis_fontsize)
    ax.set_title(plot_title, fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=xaxis_fontsize)
    ax.legend(loc=legend_location, fontsize=legend_fontsize)

    # Set y-axis font size
    ax.tick_params(axis='y', labelsize=yticklabels_fontsize)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


"""Function to save scikit-learn model into a directory"""
def save_model(model, model_filename):
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")


"""Function to load the scikit learn model"""
def load_model(model_filename):
    loaded_model = joblib.load(model_filename)
    return loaded_model


"""Function to save the visualiazation figure to a directory"""
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join("images", fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)