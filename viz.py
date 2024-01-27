# visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

# Example usage
simulation_run_number = 1
output_folder = 'output'


def save_graph(simulation_run_number, graph_name, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate a folder name with the current date and simulation run number
    folder_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_simrun{simulation_run_number}"

    # Create the folder if it doesn't exist
    folder_path = os.path.join(output_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Specify the path to save the figure inside the generated folder
    graph_path = os.path.join(folder_path, f"{datetime.datetime.now().strftime('%Y%m%d')}_graph_{graph_name}.png")

    # Save the figure
    plt.savefig(graph_path)

def visualize_generated_data(generated_samples):


    """
    Visualize the generated data using histograms, box plots, and correlation matrix.
    """
    for i in range(4):
        plt.hist(generated_samples[:, i], bins=30, label=f'Station {i+1}', alpha=0.7)

    graph_name = 'Generated Yields Histogram'
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.title(graph_name)
    plt.legend()
    plt.show()
    # Save graph
    save_graph(simulation_run_number, graph_name, output_folder)

    graph_name = 'Generated Yields Box Plot'
    sns.boxplot(data=generated_samples)
    plt.xlabel('Station')
    plt.ylabel('Yield')
    plt.title(graph_name)
    plt.show()
    # Save graph
    save_graph(simulation_run_number, graph_name, output_folder)

    graph_name = 'Correlation Matrix of Generated Yields'
    correlation_matrix = np.corrcoef(generated_samples, rowvar=False)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=['Station 1', 'Station 2', 'Station 3', 'Station 4'], yticklabels=['Station 1', 'Station 2', 'Station 3', 'Station 4'])
    plt.title(graph_name)
    plt.show()
    # Save graph
    save_graph(simulation_run_number, graph_name, output_folder)

#%%
