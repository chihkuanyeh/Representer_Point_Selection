# Representer Point Selection for Explaining Deep Neural Networks

Code release for [Representer Point Selection for Explaining Deep Neural Networks](https://arxiv.org/abs/1811.09720) at NeurIPS 2018


## Instructions

Before running any code run the following script:

```
mkdir data output; cd influence-release-mod/scripts; ln -s ../../data data; ln -s ../../output output
```

This will create symbolic links to two directories `data` and `output`, which will be used by the influence function code inside the `influence-release-mod` folder.

`data` directory will store data files such as training/test data, training/test features, etc. that is used to compute the influence function values / representer values.

`output` directory will store the computed influence function values /representer values.

You can download the contents of the `data` and `output` directory used [here](https://drive.google.com/drive/folders/1sB8MjeFmh_8-1mpkaBJdO89h6e1ELLnu?usp=sharing)

Both directory contains information used by the notebooks in `experiments` folder, which can be used to replicate the figures in the paper.


To calculate the representer values as in the paper, run
```
python compute_representer_vals.py --dataset Cifar
python compute_representer_vals.py --dataset AwA

```
in python 3 (for python 2, remove  encoding = 'latin1' in load_data() )
