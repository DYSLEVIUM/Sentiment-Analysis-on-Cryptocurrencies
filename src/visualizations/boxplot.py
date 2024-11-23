import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .visualization import Visualization


class BoxPlot(Visualization):
    def __init__(
        self,
        boxplot_data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Box Plot",
        filename: str = "box_plot",
    ):
        self.boxplot_data = boxplot_data
        self.x = x
        self.y = y
        self.title = title

        super().__init__(filename)

    def _plot(self):
        fig = plt.figure(figsize=(12, 10))
        sns.boxplot(
            data=self.boxplot_data,
            x=self.x,
            y=self.y,
            legend="full",
        )
        plt.title(self.title)

        return fig
