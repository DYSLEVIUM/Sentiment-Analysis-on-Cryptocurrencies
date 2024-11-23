import pandas as pd

class CSVReader:
    @staticmethod
    def read(file_path: str, usecols: list[str] = None, lineterminator: str = "\n"):
        df = pd.read_csv(
            file_path,
            usecols=usecols,
            lineterminator=lineterminator,
            low_memory=False,
        )

        return df
