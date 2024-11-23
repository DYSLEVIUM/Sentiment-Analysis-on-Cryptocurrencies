import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .visualization import Visualization


class BoxPlot(Visualization):
    def __init__(
        self,
        boxplot_data: pd.DataFrame,
        x_label: str = "X Label",
        y_label: str = "Y Label",
        title: str = "Box Plot",
        filename: str = "box_plot",
    ):
        super().__init__(filename)

        self.boxplot_data = boxplot_data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

    def plot(self):
        fig = plt.figure(figsize=(12, 10))
        sns.boxplot(
            data=self.boxplot_data,
            x=self.x_label,
            y=self.y_label,
            grid=False,
            legend="full",
        )
        plt.title(self.title)

        return fig
