import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class Visualization(ABC):
    filepath = "data/out/visualizations/"
    filename: str

    def __init__(self, filename: str):
        self.filename = filename

        plt.style.use("ggplot")
        sns.set_theme()

        self.fig = self._plot()

    @abstractmethod
    def _plot(self) -> plt.Figure:
        pass

    def save(self):
        self.fig.savefig(
            self.filepath + self.filename + ".svg",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(self.fig)
