import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_summary_data(file_paths):
    """Load and aggregate summary data from multiple CSV files."""
    data_frames = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data_frames.append(df)

    # Combine all data into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


def plot_radar_chart(df, output_file="model_comparison_radar.png"):
    """Generate a radar chart comparing Structural Accuracy, Functional Correctness, Consistency, and Avg Response Time."""
    # Define categories for the radar chart
    categories = [
        "Structural Accuracy",
        "Functional Correctness",
        "Consistency",
        "Avg Response Time",
    ]
    num_vars = len(categories)

    # Normalize Avg Response Time to the 0-1 scale for better comparison
    df["Normalized Response Time"] = (
        df["Avg Response Time"] - df["Avg Response Time"].min()
    ) / (df["Avg Response Time"].max() - df["Avg Response Time"].min())

    # Set seaborn style
    sns.set(style="whitegrid")

    # Prepare radar chart data
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Define angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Plot each model's data
    for i, row in df.iterrows():
        values = [
            row["Structural Accuracy"],
            row["Functional Correctness"],
            row["Consistency"],
            row["Normalized Response Time"],
        ]
        values += values[:1]  # Repeat the first value to close the circle

        ax.plot(angles, values, label=row["Model ID"], marker="o", linewidth=1.5)
        ax.fill(angles, values, alpha=0.25)

    # Add labels and title
    ax.set_title(
        "Model Comparison",
        size=16,
        weight="bold",
    )
    plt.xticks(angles[:-1], categories, size=10)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color="grey", size=8)
    plt.ylim(0, 1)

    # Customize gridlines for a clean look
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    # Add legend and save the figure
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


def plot_winner_line_chart(df, output_file="model_winner_line.png"):
    """Generate a clean and minimalistic line chart showing models ranked by weighted scores using seaborn."""
    # Normalize Avg Response Time to the 0-1 scale
    df["Normalized Response Time"] = (
        df["Avg Response Time"] - df["Avg Response Time"].min()
    ) / (df["Avg Response Time"].max() - df["Avg Response Time"].min())

    df["Score"] = (df["Structural Accuracy"] + df["Consistency"]) / 2

    # Sort the DataFrame by the calculated score (ascending, lower scores are better)
    df_sorted = df.sort_values(by="Score", ascending=True)

    # Set a seaborn style
    sns.set(style="whitegrid")  # Set a minimalistic style

    # Prepare line chart data
    plt.figure(figsize=(10, 6))

    # Plotting the sorted DataFrame as a line graph
    sns.lineplot(
        data=df_sorted,
        x="Model ID",
        y="Score",
        marker="o",
        linewidth=2,
        color="b",
        markersize=8,
    )

    # Add labels and title
    plt.title("Models ranked by overall performance", size=16, weight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model ID", fontsize=12)

    # Customize ticks
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Add annotations
    for i, txt in enumerate(df_sorted["Score"]):
        plt.text(
            df_sorted["Model ID"].iloc[i], txt, f"{txt:.2f}", ha="center", va="bottom"
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=700)


def main():
    # Define the directory where the summary files are located
    summary_dir = "scores"

    # Collect all summary CSV files
    summary_files = [
        os.path.join(summary_dir, f)
        for f in os.listdir(summary_dir)
        if f.endswith("_summary.csv")
    ]

    # Load and aggregate the summary data
    combined_df = load_summary_data(summary_files)

    # Plot the radar comparison graph
    plot_radar_chart(combined_df)

    # Plot the clean minimalistic line chart for the winner graph
    plot_winner_line_chart(combined_df)


if __name__ == "__main__":
    main()
