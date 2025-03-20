import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distribution(reference_data, current_data, feature):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(reference_data[feature], label="Reference Data")
    sns.kdeplot(current_data[feature], label="Current Data")
    plt.title(f"Distribution of {feature}")
    plt.legend()
    return plt