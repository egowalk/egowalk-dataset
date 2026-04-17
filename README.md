# EgoWalk Dataset
API for the [EgoWalk dataset](https://huggingface.co/datasets/EgoWalk/trajectories).

#### News and updates
- [2026/04/17] Version 0.3.0: Language annotations have been significantly revised for the upcoming paper update. We now provide two distinct sets of annotations: *goal_boxes* (generated using an algorithm similar to the initial publication) and *end2end* (generated using end-to-end VLM prompting). Compared to the original version, these new annotations are much cleaner and more informative. Further details will be shared in the forthcoming publication update. The original annotations have been moved to the legacy directory and are no longer recommended for use.
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

* **GNM Language Legacy**. Legacy version of extension of the GNM dataset for the language annotations modalitites. See [gnm_language_example_legacy.ipynb](examples/gnm_language_example_legacy.ipynb)