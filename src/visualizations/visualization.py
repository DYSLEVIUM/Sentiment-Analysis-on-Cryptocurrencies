import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class Visualization(ABC):
    filepath = "data/out/visualizations2/"
    filename: str

    def __init__(self, filename: str):
        self.filename = filename

        plt.style.use("ggplot")
        sns.set_theme()

    @abstractmethod
    def plot(self) -> plt.Figure:
        pass

    def save(self):
        fig = self.plot()
        fig.savefig(
            self.filepath + self.filename + ".svg",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
