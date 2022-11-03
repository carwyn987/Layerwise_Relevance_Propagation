# LRP

1. Environment

Conda environment creation steps and specifications:

1. Make sure conda is updated completely ( my version is 22.9.0 )
2. torch version 1.12.0 is about 2GB ( may be unnecessary if you already have torch version 1.9.0 or newer )
3. Versions must match - https://pypi.org/project/torchvision/ 

Commands:
1. conda create -n lrp_mnist python=3.10 
2. conda activate lrp_mnist
3. pip install -r requirements.txt

1 Program Structure

mnist_base.py - base file user calls to run code. It contains code that parses arguments.
 1. --activation-name : choose between "sigmoid", "tanh", "relu". Sets the activations of the model to this. Default is sigmoid.
 2. --data_file_path : sets the data file. Should be set to the mnist datafile.