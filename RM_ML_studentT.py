# @author: qi7876
# @date: 2024-11-15
# @brief: The machine learning script for the RM project.
# @branch: student t distribution

import scipy.io as sio
import numpy as np
import os
import torch
from torch.distributions import StudentT
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew, kurtosis

mat_data = sio.loadmat("origin.mat")  # NOTE: Path to .mat file.
os.chdir("./student_t_distribution/")  # NOTE: Path to store the results.

# Print available vectors in the .mat file
print("Variables in .mat file:", mat_data.keys())

# Iterate over all vectors in the .mat file
for var_name in mat_data:
    # Skip metadata entries
    if var_name.startswith("__"):
        continue

    data = mat_data[var_name].flatten()

    # Remove zero values from the vector
    data = data[data != 0]

    # NOTE: Improve accuracy.
    # Dynamic bin width using Freedman-Diaconis rule
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / (len(data) ** (1 / 3))

    # Define bins for histogram
    bins = np.arange(data.min(), data.max() + bin_width, bin_width)
    # Compute histogram
    frequencies, bin_edges = np.histogram(data, bins=bins)
    # Calculate mid-points of bins
    mid_points = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_count = frequencies.sum()
    probabilities = frequencies / total_count
    data_tensor = torch.from_numpy(mid_points).double()
    prob_tensor = torch.from_numpy(probabilities).double()

    # Create a directory for the vector
    os.makedirs(var_name, exist_ok=True)

    # Fit the distribution model to the binned data
    def negative_log_likelihood(params):
        df, loc, scale = params
        df = torch.clamp(df, 1e-2, 100.0)  # Degrees of freedom > 0
        scale = torch.clamp(scale, 1e-2, None)  # Scale > 0
        dist = StudentT(df=df, loc=loc, scale=scale)
        nll = -torch.sum(prob_tensor * dist.log_prob(data_tensor))
        return nll

    # NOTE: Improve accuracy.
    # Better parameter initialization
    df_param = torch.tensor(np.log(4.0), requires_grad=True)  # Start with df=4
    loc_param = torch.tensor(
        np.median(data), requires_grad=True
    )  # Use median instead of mean
    scale_param = torch.tensor(np.log(np.std(data)), requires_grad=True)

    # NOTE: Improve accuracy.
    # Add learning rate scheduler
    optimizer = torch.optim.Adam([df_param, loc_param, scale_param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, verbose=False
    )

    # Early stopping setup
    best_loss = float("inf")
    patience = 100
    patience_counter = 0
    best_params = None

    # Increased iterations with early stopping
    for epoch in range(5000):
        optimizer.zero_grad()
        df = torch.exp(df_param) + 1e-2
        scale = torch.exp(scale_param) + 1e-2
        loc = loc_param
        dist = StudentT(df=df, loc=loc, scale=scale)
        nll = -torch.sum(prob_tensor * dist.log_prob(data_tensor))

        # Early stopping check
        if nll.item() < best_loss:
            best_loss = nll.item()
            patience_counter = 0
            best_params = (df.item(), loc.item(), scale.item())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        nll.backward()
        optimizer.step()
        scheduler.step(nll)

    # Use best parameters found
    df_est, loc_est, scale_est = best_params

    # Distribution-agnostic metrics
    # Basic statistics
    sample_mean = np.mean(data)
    sample_std = np.std(data)
    sample_skew = skew(data)
    sample_kurt = kurtosis(data)

    # Create theoretical distribution first
    theoretical_dist = stats.t(df=df_est, loc=loc_est, scale=scale_est)

    # QQ plot data
    theoretical_quantiles = theoretical_dist.ppf(np.linspace(0.01, 0.99, len(data)))
    observed_quantiles = np.sort(data)

    # Calculate percentile-based metrics
    percentiles = np.percentile(data, [1, 5, 25, 50, 75, 95, 99])
    theo_percentiles = theoretical_dist.ppf([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    percentile_rmse = np.sqrt(np.mean((percentiles - theo_percentiles) ** 2))

    # Plot QQ plot
    plt.figure()
    plt.plot(theoretical_quantiles, observed_quantiles, "bo")
    plt.plot([data.min(), data.max()], [data.min(), data.max()], "r--")
    plt.title(f"{var_name} Q-Q Plot")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.savefig(os.path.join(var_name, f"{var_name}_qq_plot.png"))
    plt.close()

    # Calculate additional metrics
    # Generate theoretical CDF values for KS test
    ks_statistic, ks_pvalue = stats.kstest(data, theoretical_dist.cdf)

    # Calculate MSE
    pdf_observed = probabilities
    pdf_predicted = theoretical_dist.pdf(mid_points)
    mse = np.mean((pdf_observed - pdf_predicted) ** 2)

    # Calculate AIC and BIC
    n_params = 3  # df, loc, scale
    n_observations = len(data)
    aic = 2 * n_params - 2 * (-best_loss)
    bic = np.log(n_observations) * n_params - 2 * (-best_loss)

    # Plot histogram and fitted PDF
    x_values = np.linspace(data.min(), data.max(), 1000)
    dist = StudentT(df=df_est, loc=loc_est, scale=scale_est)
    pdf_values = np.exp(dist.log_prob(torch.from_numpy(x_values).double()).numpy())

    plt.figure()
    plt.hist(data, bins=50, density=True, alpha=0.6, label="Data Histogram")
    plt.plot(x_values, pdf_values, "r-", lw=2, label="Fitted PDF")
    plt.title(f"{var_name} Data and Fitted Student's t-Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(var_name, f"{var_name}_distribution.png"))
    plt.close()

    def get_metric_quality(metric_name, value):
        """Evaluate the quality of different metrics."""
        guidelines = {
            "p_value": {
                "good": 0.05,
                "description": "p-value > 0.05 suggests good fit",
            },
            "mse": {
                "good": 0.01,
                "moderate": 0.05,
                "description": "MSE < 0.01 excellent, < 0.05 good",
            },
            "percentile_rmse": {
                "good": 0.1,
                "moderate": 0.3,
                "description": "RMSE < 0.1 excellent, < 0.3 good",
            },
        }

        if metric_name not in guidelines:
            return "No guidelines available"

        guide = guidelines[metric_name]
        if "p_value" in metric_name.lower():
            return "GOOD" if value > guide["good"] else "POOR"
        elif metric_name in ["mse", "percentile_rmse"]:
            if value < guide["good"]:
                return "EXCELLENT"
            elif value < guide["moderate"]:
                return "GOOD"
            else:
                return "POOR"

        return "No quality assessment available"

    # Save metrics
    with open(os.path.join(var_name, f"{var_name}_metrics.txt"), "w") as f:
        f.write("Sample Statistics:\n")
        f.write(f"Mean: {sample_mean}\n")
        f.write(f"Standard Deviation: {sample_std}\n")

        f.write("Distribution Parameters:\n")
        f.write(f"Degrees of Freedom (df): {df_est}\n")
        f.write(f"Location (loc): {loc_est}\n")
        f.write(f"Scale (scale): {scale_est}\n\n")

        f.write("Goodness of Fit Tests:\n")
        f.write("Kolmogorov-Smirnov test:\n")
        f.write(f"  Statistic: {ks_statistic}\n")
        f.write(
            f'  p-value: {ks_pvalue} - Quality: {get_metric_quality("p_value", ks_pvalue)}\n'
        )

        f.write("Fit Quality Metrics:\n")
        f.write(
            f'Mean Squared Error: {mse} - Quality: {get_metric_quality("mse", mse)}\n'
        )
        f.write(f"AIC: {aic} (Lower is better.)\n")
        f.write(f"BIC: {bic} (Lower is better.)\n")
        f.write(
            f'Percentile RMSE: {percentile_rmse} - Quality: {get_metric_quality("percentile_rmse", percentile_rmse)}\n\n'
        )