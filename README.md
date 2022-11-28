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
  $ python mnist_base.py --epochs 2 --experiment-name dev_experiment --activation-name relu 
  ###### --model 'dev_experiment/' ######
  $ python lrp_eval.py --experiment "dev_experiment/"
  $ python lrp_feature_removal_eval.py --experiment "dev_experiment_lrp-gamma/" --lrp-rule gamma

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
 6. "experiment_folder/feature_removal" contains an lrp image, and one classification with important features removed, one with unimportant features removed, and the original image. Each image is accompanied by a text file showing the classification confidences.

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
 4. --gamma
Learn more by running with the help command.

lrp_feature_removal_eval.py - evaluate prediction confidence when important and not important image sections are removed.
 1. --experiment
 2. --lrp-rule
 3. --epsilon
 4. --gamma

3 Development Options
 - If the experiment name contains "dev", then new folders with "_1", "_2", ..., "_n" will not be appended. The current folders contents will be deleted.

4 Resources
  - https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html 
  - https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/ 
  - https://github.com/python-engineer/pytorchTutorial 
  - Bach S, Binder A, Montavon G, Klauschen F, Müller K-R, Samek W (2015) On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation. PLoS ONE 10(7): e0130140. https://doi.org/10.1371/journal.pone.0130140 
  - Montavon, Grégoire & Binder, Alexander & Lapuschkin, Sebastian & Samek, Wojciech & Müller, Klaus-Robert. (2019). Layer-Wise Relevance Propagation: An Overview. 10.1007/978-3-030-28954-6_10. 
  - Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), 141–142.
  - Bach, S., Samek, W., Mueller, K.R., Binder, A. and Montavon, G., Fraunhofer Gesellschaft zur Forderung der Angewandten Forschung eV and Technische Universitaet Berlin, 2018. Relevance score assignment for artificial neural networks. U.S. Patent Application 15/710,455.

5 Next Steps
 - Make DNN model scalable and generalize code base to conform to whatever model is produced
 - Generalize code base to support CNN Models
 - Generalize data loading to support other image datasets
 - Evaluation with removal of important sections of images and test performance. Note that a separate evaluation script must be written for this.

6 Goals
 - Get LRP Rules Fixed / Verified by Sunday (11/13/22)
 - Generalize the model / dataset by 11/18/22
 - Evaluation (Testing different LRP rules + important image region removal vs unimportant removal)