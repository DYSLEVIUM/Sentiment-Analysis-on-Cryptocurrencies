from ..postsDatasource import PostsDataSource
import pandas as pd

class TweetsDataSource(PostsDataSource):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
