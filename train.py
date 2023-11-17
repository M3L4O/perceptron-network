from src.network.perceptron import Perceptron
from src.data.dataset import Dataset
from argparse import ArgumentParser


def parser_args():
    args = ArgumentParser()
    args.add_argument("--dataset", "-d", type=str, required=True)
    args.add_argument("--max-epoch", "-e", type=int, required=True)
    args.add_argument("--input-shape", "-i", type=int, required=True)

    return args.parse_args()

def train():
    args = parser_args()
    dataset = Dataset(filename=args.dataset)
    X, Y = dataset.load_from_xls(input_shape=args.input_shape)
    model = Perceptron(input_shape=args.input_shape)
    model.train(X, Y, max_epochs=args.max_epoch)
    model.save("models/weights.npy")
    

if __name__ == "__main__":
    train()