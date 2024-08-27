import os
import pandas as pd
import matplotlib.pyplot as plt


def load_summary_data(file_paths):
    """Load and aggregate summary data from multiple CSV files."""
    data_frames = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data_frames.append(df)

    # Combine all data into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


def plot_comparison_graph(df, output_file="model_comparison_line.png"):
    """Generate a line chart comparing accuracy, consistency, and response time."""
    df.set_index("Model ID", inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    df.plot(kind="line", marker="o", ax=ax)

    ax.set_title("Model Comparison: Accuracy, Consistency, and Avg Response Time")
    ax.set_ylabel("Score (0-1 scale)")
    ax.set_xlabel("Model ID")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()  # Removed this line as it's unnecessary in a non-interactive environment


def plot_winner_graph(df, output_file="model_winner.png"):
    """Generate a line chart showing models ranked by weighted scores."""
    # Normalize Avg Response Time to the 0-1 scale
    df["Normalized Response Time"] = (
        df["Avg Response Time"] - df["Avg Response Time"].min()
    ) / (df["Avg Response Time"].max() - df["Avg Response Time"].min())

    # Calculate a weighted score: 2/5 for accuracy, 2/5 for consistency, and 1/5 for normalized response time
    df["Score"] = (
        (2 / 5) * df["Accuracy"]
        + (2 / 5) * df["Consistency"]
        + (1 / 5) * df["Normalized Response Time"]
    )

    # Sort the DataFrame by the calculated score (ascending, lower scores are better)
    df_sorted = df.sort_values(by="Score", ascending=True)

    # Plotting the sorted DataFrame as a line graph
    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted["Score"].plot(kind="line", marker="o", ax=ax, color="blue")

    ax.set_title("Models Ranked by Weighted Score (Lower is Better)")
    ax.set_ylabel("Weighted Score")
    ax.set_xlabel("Model ID")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()  # Removed this line as it's unnecessary in a non-interactive environment


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

    # Plot the comparison line graph
    plot_comparison_graph(combined_df)

    # Plot the winner graph
    plot_winner_graph(combined_df)


if __name__ == "__main__":
    main()
