from .tweetsDatasource import TweetsDataSource
from src.utils.csvReader import CSVReader

class TweetsCSVDataSource(TweetsDataSource):
    file_path: str

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = CSVReader.read(file_path, usecols=["date", "text"], lineterminator="\n")

        super().__init__(self.df)
    