import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def visualize_correlation(features):

    # Visualize correlation among features
    correlation_matrix = np.corrcoef(features, rowvar=False)

    # Create a heatmap for visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=features.columns, yticklabels=features.columns)
    plt.title('Correlation Heatmap')
    plt.show()