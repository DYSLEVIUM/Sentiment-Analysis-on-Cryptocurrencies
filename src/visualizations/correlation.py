import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .visualization import Visualization


class CorrelationPlot(Visualization):
    def __init__(
        self,
        corr_df: pd.DataFrame,
        title: str = "Correlation Plot",
        filename: str = "correlation_plot",
    ):
        super().__init__(filename)
        self.corr_df = corr_df
        self.title = title

    def plot(self):
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(self.corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(self.title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        return fig
