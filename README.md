# LRP

1 Environment

Conda environment creation steps and specifications:

1. Make sure conda is updated completely ( my version is 22.9.0 )
2. torch version 1.12.0 is about 2GB ( may be unnecessary if you already have torch version 1.9.0 or newer )
3. Versions must match - https://pypi.org/project/torchvision/ 

Commands:
1. conda create -n lrp_mnist python=3.10 
2. conda activate lrp_mnist
3. pip install -r requirements.txt

Example command line program execution:
  $ python mnist_base.py --epochs 2 --experiment-name dev_experiment --activation-name relu --model 'dev_experiment/'
  $ python lrp_eval.py --experiment "dev_experiment/"

2 Program Structure

Folders:
 1. src - contains all python files / modules with the exception of the user-called command line function mnist_base.py
 2. "experiment-name" - contains all train, test charts and data associated with a single call of the program with --experiment-name flag defined.

mnist_base.py - base file user calls to run code. It contains code that parses arguments.
 1. --activation-name : choose between "sigmoid", "tanh", "relu". Sets the activations of the model to this. Default is sigmoid.
 2. --data_file_path : sets the data file. Should be set to the mnist datafile.

3 Development Options
 - If the experiment name contains "dev", then new folders with "_1", "_2", ..., "_n" will not be appended. The current folders contents will be deleted.


4 Resources
  - https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html 
  - https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/ 
  - https://github.com/python-engineer/pytorchTutorial 