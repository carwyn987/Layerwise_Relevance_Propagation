import argparse
from data import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog = 'mnist_base',
                    description = 'loads mnist, creates a classifier, trains classifier, and returns test results',
                    epilog = 'CSE 512 - Machine Learning, Carwyn Collinsworth, Gopi Sumanth, Akhil Arradi')

    parser.add_argument('--data_file_path', type=str,
                            help='mnist data file path.')
    parser.add_argument('--activation-name', type=str, choices=("relu", "tanh", "sigmoid"),
                            help='activation-name', default="sigmoid")
    parser.add_argument('--batch_size', type=int,
                            help='training batch size of data.', default=100)

    args = parser.parse_args()

    print("\ndata_file_path: ", args.data_file_path)
    print("activation-name: ", args.activation_name)
    print("batch-size: ", args.batch_size)

    print("\nLoading Data from MNIST")
    train_loader, test_loader = load_data(args.batch_size, args.data_file_path)
    print("Dataset loaded.")

    # Resources:
    # https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/