import matplotlib.pyplot as plt
import seaborn as sns
from .visualization import Visualization
import numpy as np


class ModelComparisonPlot(Visualization):
    def __init__(self, results_dict):
        """
        results_dict: Dictionary with model names as keys and metrics as values
        Example: {
            'BERT': {'accuracy': 0.85, 'f1': 0.84},
            'RoBERTa': {'accuracy': 0.87, 'f1': 0.86}
        }
        """
        super().__init__()
        self.results = results_dict

    def plot(self):
        metrics = list(next(iter(self.results.values())).keys())
        models = list(self.results.keys())

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35

        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric)

        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models)
        ax.legend()

        return fig
