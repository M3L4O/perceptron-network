from dataclasses import dataclass
import pandas as pd


@dataclass
class Dataset:
    filename: str

    def load_from_xls(self, input_shape: int) -> tuple:
        df = pd.read_excel(self.filename, header=None)
        X = df.iloc[1:, :input_shape].values
        Y = df.iloc[1:, input_shape].values
        return X, Y


if __name__ == "__main__":
    dataset = Dataset(
        filename="/home/melao/Codes/Python/ANN/Treinamento_Perceptron.xls"
    )
    X, Y = dataset.load_from_xls(input_shape=3)
    print(len(X))
    print(len(Y))
