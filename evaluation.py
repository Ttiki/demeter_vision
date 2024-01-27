import numpy as np

def evaluate_generated_samples(generated_samples, ground_truth_samples):
    """
    Evaluate generated samples using mean squared error.

    Parameters
    ----------
    generated_samples : ndarray
        Generated samples from your generative model.
    ground_truth_samples : ndarray
        Ground truth samples for comparison.

    Returns
    -------
    mse : float
        Mean squared error between generated and ground truth samples.
    """
    # Consider only the first 4 columns of generated samples for comparison
    generated_samples_subset = generated_samples[:, :4]

    mse = np.mean((ground_truth_samples - generated_samples_subset)**2)

    # Print the evaluation result
    print(f"Mean Squared Error: {mse}")
    print("Generated Samples:")
    print(generated_samples_subset[:5])  # Print the first 5 samples as an example

    return mse