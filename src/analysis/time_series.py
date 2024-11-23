from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.visualizations.sentiment_price import SentimentPricePlot
from src.visualizations.correlation import CorrelationPlot
import os

@dataclass
class TimeSeriesAnalysis:
    def __init__(self, df):
        self.df = df

        self.df['next_day_price'] = self.df['price'].shift(-1)
        self.df['two_day_price'] = self.df['price'].shift(-2)
        self.df = self.df.dropna(subset=['next_day_price', 'two_day_price'])
        
    def calculate_metrics(self):
        metrics = {}
        
        for price_col in ['next_day_price', 'two_day_price']:
            correlation, p_value = stats.pearsonr(
                self.df['sentiment'], 
                self.df[price_col]
            )
            metrics[f'correlation_{price_col}'] = correlation
            metrics[f'p_value_{price_col}'] = p_value
            
            # Spearman rank correlation
            spearman_corr, spearman_p = stats.spearmanr(
                self.df['sentiment'], 
                self.df[price_col]
            )
            metrics[f'spearman_correlation_{price_col}'] = spearman_corr
            metrics[f'spearman_p_value_{price_col}'] = spearman_p
        
        return metrics
    
    def calculate_average_sentiment_and_price(self):
        # Group by date and calculate average sentiment and next day's price
        grouped_df = (
            self.df.groupby('date')
            .agg({'sentiment': "mean", "next_day_price": "mean"})
            .reset_index()
        )
        return grouped_df
    
    def generate_visualizations(self, output_dir='visualizations'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Time series plot
        sentiment_price_plot = SentimentPricePlot(
            self.df, 
            'sentiment', 
            'price', 
            'date'
        )
        fig_time_series = sentiment_price_plot.plot()
        fig_time_series.savefig(f'{output_dir}/sentiment_price_time_series.svg')
        
        # Correlation plot
        correlation_plot = CorrelationPlot(
            self.df, 
            'sentiment', 
            'next_day_price'
        )
        fig_correlation = correlation_plot.plot()
        fig_correlation.savefig(f'{output_dir}/sentiment_price_correlation.svg')
        
        return {
            'time_series': fig_time_series,
            'correlation': fig_correlation
        }