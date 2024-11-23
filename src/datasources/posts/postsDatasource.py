from ..datasource import DataSource
import pandas as pd

class PostsDataSource(DataSource):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
