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
    """Generate a radar chart comparing Structural Accuracy, Functional Correctness, Consistency, and Adjusted Avg Response Time."""
    # Define categories for the radar chart
    categories = [
        "Structural Accuracy",
        "Functional Correctness",
        "Consistency",
        "Adjusted Response Time",  # Updated label
    ]
    num_vars = len(categories)

    # Normalize Avg Response Time to the 0-1 scale
    df["Normalized Response Time"] = (
        df["Avg Response Time"] - df["Avg Response Time"].min()
    ) / (df["Avg Response Time"].max() - df["Avg Response Time"].min())

    # Adjusted Normalized Response Time: Higher response time should result in a lower score
    df["Adjusted Response Time"] = df["Normalized Response Time"]

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
            row["Adjusted Response Time"],  # Use adjusted response time
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
    plt.savefig(output_file, dpi=700)


def plot_winner_line_chart(df, output_file="model_winner_line_comparison.png"):
    """Generate a clean and minimalistic line chart showing models ranked by weighted scores with and without normalized response time using seaborn."""
    # Normalize Avg Response Time to the 0-1 scale (Higher is worse, so we invert it)
    df["Normalized Response Time"] = (
        df["Avg Response Time"] - df["Avg Response Time"].min()
    ) / (df["Avg Response Time"].max() - df["Avg Response Time"].min())

    # Adjusted Normalized Response Time: Higher response time should result in a lower score
    df["Adjusted Response Time"] = 1 - df["Normalized Response Time"]

    # Calculate overall score without response time
    df["Score_Without_Time"] = (
        df["Structural Accuracy"] + df["Consistency"] + df["Functional Correctness"]
    ) / 3

    # Calculate overall score with adjusted response time
    df["Score_With_Time"] = (
        df["Structural Accuracy"]
        + df["Consistency"]
        + df["Functional Correctness"]
        + df["Adjusted Response Time"]
    ) / 4

    # Save the calculated scores to CSV
    df[["Model ID", "Score_Without_Time", "Score_With_Time"]].to_csv(
        "scores/model_scores.csv", index=False
    )

    # Sort the DataFrame by the calculated scores
    df_sorted_without_time = df.sort_values(by="Score_Without_Time", ascending=True)
    df_sorted_with_time = df.sort_values(by="Score_With_Time", ascending=True)

    # Set a seaborn style
    sns.set(style="whitegrid")  # Set a minimalistic style

    # Prepare line chart data
    plt.figure(figsize=(10, 6))

    # Plotting the sorted DataFrame as line graphs
    sns.lineplot(
        data=df_sorted_without_time,
        x="Model ID",
        y="Score_Without_Time",
        marker="o",
        linewidth=2,
        color="b",
        markersize=8,
        label="Without Normalized Time",
    )

    sns.lineplot(
        data=df_sorted_with_time,
        x="Model ID",
        y="Score_With_Time",
        marker="o",
        linewidth=2,
        color="r",
        markersize=8,
        label="With Adjusted Normalized Time",
    )

    # Add labels and title
    plt.title(
        "Models ranked by overall performance",
        size=16,
        weight="bold",
    )
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model ID", fontsize=12)

    # Customize ticks
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Add annotations for both lines
    for i, txt in enumerate(df_sorted_without_time["Score_Without_Time"]):
        plt.text(
            df_sorted_without_time["Model ID"].iloc[i],
            txt,
            f"{txt:.2f}",
            ha="center",
            va="bottom",
        )
    for i, txt in enumerate(df_sorted_with_time["Score_With_Time"]):
        plt.text(
            df_sorted_with_time["Model ID"].iloc[i],
            txt,
            f"{txt:.2f}",
            ha="center",
            va="top",
        )

    # Show legend
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=700)


def generate_winner_graph_from_csv(
    input_file="scores/model_scores.csv", output_file="winner_graph_from_csv.png"
):
    """Generate a clean and minimalistic line chart showing models ranked by weighted scores from model_scores.csv."""

    # Load the data from CSV
    df = pd.read_csv(input_file)

    # Sort the DataFrame by the calculated scores
    df_sorted_without_time = df.sort_values(by="Score_Without_Time", ascending=True)
    df_sorted_with_time = df.sort_values(by="Score_With_Time", ascending=True)

    # Set a seaborn style
    sns.set(style="whitegrid")  # Set a minimalistic style

    # Prepare line chart data
    plt.figure(figsize=(10, 6))

    # Plotting the sorted DataFrame as line graphs
    sns.lineplot(
        data=df_sorted_without_time,
        x="Model ID",
        y="Score_Without_Time",
        marker="o",
        linewidth=2,
        color="b",
        markersize=8,
        label="Without Normalized Time",
    )

    sns.lineplot(
        data=df_sorted_with_time,
        x="Model ID",
        y="Score_With_Time",
        marker="o",
        linewidth=2,
        color="r",
        markersize=8,
        label="With Adjusted Normalized Time",
    )

    # Add labels and title
    plt.title(
        "Models ranked by overall performance",
        size=16,
        weight="bold",
    )
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model ID", fontsize=12)

    # Customize ticks
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Add annotations for both lines
    for i, txt in enumerate(df_sorted_without_time["Score_Without_Time"]):
        plt.text(
            df_sorted_without_time["Model ID"].iloc[i],
            txt,
            f"{txt:.2f}",
            ha="center",
            va="bottom",
        )
    for i, txt in enumerate(df_sorted_with_time["Score_With_Time"]):
        plt.text(
            df_sorted_with_time["Model ID"].iloc[i],
            txt,
            f"{txt:.2f}",
            ha="center",
            va="top",
        )

    # Show legend
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=700)


def main():
    # Define the directory where the summary files are located
    summary_dir = "scores"

    # Collect all summary CSV files
    summary_files = [
        os.path.join(summary_dir, f)
        for f in os.listdir(summary_dir)
        if f.endswith("_summary.csv") and not f.startswith("mistral")
    ]

    # Load and aggregate the summary data
    combined_df = load_summary_data(summary_files)

    # Plot the radar comparison graph
    plot_radar_chart(combined_df)

    # Plot the clean minimalistic line chart for the winner graph
    plot_winner_line_chart(combined_df)

    # generate_winner_graph_from_csv() # uncomment to generate the winner graph from model_scores.csv


if __name__ == "__main__":
    main()
