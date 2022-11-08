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
 1. "src/" - contains all python files / modules with the exception of the user-called command line function mnist_base.py
 2. "experiment-name" - contains all train, test charts and data associated with a single call of the program with --experiment-name flag defined.
 3. "dev_experiment_lrp-0/", created by running the following commands:
  - python mnist_base.py --epochs 5 --experiment-name dev_experiment_lrp-0 --activation-name sigmoid
  - python lrp_eval.py --experiment "dev_experiment_lrp-0/" --lrp-rule 0
 4. "dev_experiment_lrp-epsilon/", created by running the following commands:
  - python mnist_base.py --epochs 5 --experiment-name dev_experiment_lrp-epsilon --activation-name sigmoid
  - python lrp_eval.py --experiment "dev_experiment_lrp-epsilon/" --lrp-rule epsilon --epsilon 0.2
 5. "data/", contains all mnist data

Base Python Files:

mnist_base.py - base file user calls to run mnist training code. It contains code that parses the following optional arguments:
 1. --activation-name
 2. --data_file_path
 3. -batch_size
 4. --hidden_size
 5. --device
 6. --loss
 7. --optimizer
 8. --learning-rate
 9. --epochs
 10. --experiment-name
 11. --model
 Learn more by running with the help command.

lrp_eval.py - base file user calls to run lrp. It contains code that parses arguments.
 1. --experiment
 2. --lrp-rule
 3. --epsilon
Learn more by running with the help command.

3 Development Options
 - If the experiment name contains "dev", then new folders with "_1", "_2", ..., "_n" will not be appended. The current folders contents will be deleted.

4 Resources
  - https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html 
  - https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/ 
  - https://github.com/python-engineer/pytorchTutorial 
  - 