# EgoWalk Dataset
API for the [EgoWalk dataset](https://huggingface.co/datasets/EgoWalk/trajectories).

#### News and updates
- [2025/05/31] Version 0.2.0: simplified dataset downloading, see updated examples.

## Installation
We support **Python 3.9+**. Clone the repo and use `pip` to install the library:
```shell
pip3 install --upgrade pip
git clone https://github.com/egowalk/egowalk-dataset.git
cd egowalk-dataset
pip3 install -e .
```

To check the examples, you may also need to install `Jupyter` and `matplotlib`.

## Overview
We provide several dataset wrappers to interact with the data:

* **Trajectory**. Can be useful for interacting with the individual trajectories from the dataset. See [trajectory_example.ipynb](./examples/trajectory_example.ipynb)

* **GNM**. Stands for the [General Navigation Model](https://general-navigation-models.github.io/). Dataset provides data in the observation-goal-action format, which can be used to train GNM-family models ([ViNT](https://arxiv.org/abs/2306.14846), [NoMaD](https://arxiv.org/abs/2310.07896), etc.), or in any other downstream tasks. Implementation is highly inspired by the [GNM implementation](https://github.com/robodhruv/drive-any-robot/blob/main/train/gnm_train/data/gnm_dataset.py). See [gnm_example.ipynb](./examples/gnm_example.ipynb).

* **GNM Language**. Extension of the GNM dataset for the language annotations modalitites. See [gnm_language_example.ipynb](examples/gnm_language_example.ipynb)
