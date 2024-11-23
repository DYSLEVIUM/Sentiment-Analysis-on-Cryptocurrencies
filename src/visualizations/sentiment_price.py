import matplotlib.pyplot as plt
import seaborn as sns
from .visualization import Visualization


class SentimentPricePlot(Visualization):
    def __init__(self, df, sentiment_col, price_col, date_col="date"):
        super().__init__()
        self.df = df
        self.sentiment_col = sentiment_col
        self.price_col = price_col
        self.date_col = date_col

    def plot(self):
        # Create dual-axis plot similar to your existing implementation
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=600)

        # Price axis
        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"{self.price_col} (USD)", color="blue")
        ax1.plot(
            self.df[self.date_col],
            self.df[self.price_col],
            color="blue",
            marker="o",
            label=f"{self.price_col}",
        )
        ax1.tick_params(axis="y", labelcolor="blue")

        # Sentiment axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Sentiment Score", color="red")
        ax2.plot(
            self.df[self.date_col],
            self.df[self.sentiment_col],
            color="red",
            marker="o",
            linestyle="--",
            label="Sentiment",
        )
        ax2.tick_params(axis="y", labelcolor="red")

        # Add title and legend
        plt.title(f"{self.price_col} Price vs Sentiment Over Time")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        return fig
