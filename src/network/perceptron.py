from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from src.activation.functions import bipolar_binary_step


@dataclass
class Perceptron:
    input_shape: int
    activation: callable = bipolar_binary_step
    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        self.weights = np.random.rand(self.input_shape + 1)
        self.bias = -1

    def __call__(self, x: np.ndarray) -> int:
        u = np.matmul(x, self.weights[1:]) + self.weights[0] * self.bias
        return self.activation(u)


    def train(self, X: np.ndarray, Y: np.ndarray, max_epochs: int = 100) -> None:
        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1}")
            has_error = False
            for x, y in tqdm(zip(X, Y)):
                prediction = self(x)
                error = y - prediction
                if error:
                    has_error = True
                    self.weights = (
                        self.weights
                        + self.learning_rate * error * np.append(self.bias, x)
                    )

            if not has_error:
                print(f"Rede convergiu na época {epoch}.")
                break

    def test(self, X: np.ndarray, Y: np.ndarray) -> list:
        tp, tn, fp, fn = 0, 0, 0, 0
        for x, y in zip(X, Y):
            prediction = self(x)
            if prediction > 0:
                if y > 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if y > 0:
                    fn += 1
                else:
                    tn += 1
        return tp, tn, fp, fn

    def save(self, filename: str) -> None:
        np.save(filename, self.weights, allow_pickle=True)

    def load(self, filename: str) -> None:
        self.weights = np.load(filename, allow_pickle=True)


if __name__ == "__main__":
    model = Perceptron(input_shape=2)
