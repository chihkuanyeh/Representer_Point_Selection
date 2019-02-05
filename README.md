# Representer Point Selection for Explaining Deep Neural Networks

Code release for [Representer Point Selection for Explaining Deep Neural Networks](link.to.follow)


## Instructions

Before running any code run the following script:

```
mkdir data output; cd influence-release-mod/scripts; ln -s ../../data data; ln -s ../../output output
```

This will create symbolic links to two directories `data` and `output`, which will be used by the influence function code inside the `influence-release-mod` folder.

`data` directory will store data files such as training/test data, training/test features, etc. that is used to compute the influence function values / representer values.

`output` directory will store the computed influence function values /representer values.

You can download the contents of the `data` and `output` directory used [here](https://drive.google.com/drive/u/1/folders/1juHGib-4qo7kpNoVS2vq-jmD2rey-yhg)

To calculate the representer values as in the paper, run
```
python compute_representer_vals.py --dataset Cifar
python compute_representer_vals.py --dataset AwA

```


Both directory contains information used by the notebooks in `experiments` folder, which can be used to replicate the figures in the paper.
