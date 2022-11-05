import os
import glob
import argparse
from src.data import *
from src.model import *
from src.train import *
from src.test import *

if __name__ == '__main__':

    # Parse Command Line Arguments

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
    parser.add_argument('--hidden_size', type=int,
                            help='hidden layer size of network.', default=500)
    parser.add_argument('--device', type=str, choices=("cuda", "cpu"),
                            help='model device')
    parser.add_argument('--loss', type=str, choices=("cross-entropy", "mse"),
                            help='loss function', default="cross-entropy")
    parser.add_argument('--optimizer', type=str, choices=("adam", "sgd"),
                            help='optimizer', default="adam")
    parser.add_argument('--learning-rate', type=float,
                            help='learning rate', default=0.001)
    parser.add_argument('--epochs', type=int,
                            help='number of epochs', default=10)
    parser.add_argument('--experiment-name', type=str,
                            help='Name of experiment folder.')

    args = parser.parse_args()
    print(args)

    # Experiment Save

    path = None
    if args.experiment_name:

        # Development Experiment Name:
        if "dev" in args.experiment_name:
            path = args.experiment_name + "/"
            try:
                os.makedirs(path)
            except FileExistsError:
                print("\nDirectory already exists, deleting contents.")
                files = glob.glob(path + "*")
                for f in files:
                    os.remove(f)
            
        else:
            # Compute name of directory
            path = args.experiment_name + "/"
            append_num = 1
            if os.path.exists(path):
                path = args.experiment_name + "_" + str(append_num) + "/"
            while os.path.exists(path):
                append_num += 1
                path = path[:-2] + str(append_num) + "/"

            os.makedirs(path)
            print("\nCreated experiment directory at ", path)

        # Save program arguments
        d = dict(vars(args))
        info = open(path + "info.txt", 'w')
        for key in d:
            info.write(key + ": " + str(d[key]) + "\n") if d[key] else info.write(key + ": " + "None" + "\n")
        info.close()

    # Load MNIST Data

    print("\nLoading Data from MNIST")
    train_loader, test_loader = load_data(args.batch_size, args.data_file_path)
    print("Dataset loaded.")

    # Define Network

    input_size = 28*28 # 784
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == None else args.device
    model = SimpleNetwork(input_size, args.hidden_size, args.activation_name).to(device)
    print("\n" + str(model) + " on " + device + "\n")

    criterion = model.get_loss_function(args.loss)
    optimizer = model.get_optimizer(args.optimizer, args.learning_rate)

    # Train Loop

    train(train_loader, model, device, criterion, optimizer, args.epochs, path)

    # Testing

    print(accuracy(test_loader, model, device))