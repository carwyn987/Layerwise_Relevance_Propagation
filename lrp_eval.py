import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.lrp import *
from src.model import *
from src.data import *

if __name__ == '__main__':

    # Parse Command Line Arguments

    parser = argparse.ArgumentParser(
                    prog = 'lrp_eval',
                    description = 'evaluates a saved model from an experiment using lrp',
                    epilog = 'CSE 512 - Machine Learning, Carwyn Collinsworth, Gopi Sumanth, Akhil Arradi')

    parser.add_argument('--experiment', type=str,
                    help='path to experiment folder, no need to include model name')
    parser.add_argument('--lrp-rule', type=str, choices=("lrp-0", "lrp-epsilon", "lrp-gamma", "lrp-composite"),
                            help='specify lrp rule to use', default="lrp-0")

    args = parser.parse_args()
    print(args)

    # Load args from info

    with open(args.experiment + "info.json") as f:
        info = json.loads(f.read())
    
    print("Info: ", info)
    input_size = 28*28 # 784

    # Create model

    experiment_dir = info['experiment_name'] + "/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LRP(SimpleNetwork(input_size, info['hidden_size'], info['activation_name'], experiment_dir), experiment_dir, device)

    # Load MNIST test data

    print("\nLoading Data from MNIST")
    train_loader, test_loader = load_data(info['batch_size'], None)
    print("Dataset loaded.")
    
    # Make a single prediction

    img = np.squeeze(next(iter(test_loader))[0][0])

    # Perform LRP

    lrp_img = model.get_lrp_image(img, args.lrp_rule)

    # Save Visualization

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle('Input image vs LRP visualization')
    axs[0].imshow(img)
    axs[1].imshow(lrp_img)
    fig.savefig(experiment_dir + "lrp_" + args.lrp_rule + ".png")