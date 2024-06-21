# autoTS
Automatized pipeline for TS analysis


## Setup

1. Clone the repository
2. Install the requirements (Installing pytorch and tensorflow at the same time is not recommended, due to dependency conflicts. Install only the one you need.)
```bash
# requirements are needed for the pipeline and transformations to work
pip install -r requirements.txt

# install pytorch
pip install -r requirements_torch.txt

# install tensorflow
pip install -r requirements_tensorflow.txt
```

# Pipeline structure
The pipeline consists of 3 main steps: dataset parsing, data transformation and model training and evaluation. Each of these steps can be customized by adding new datasets, transformations and models. The output of the first step is a python list of tuples containing (X,y) pairs, where X is the input data and y is the target variable. For example a time series classification task would contain a list X of length N representing a time series of length N, and y containing the label for the time series. The output of the second step is a transformed version of the input data, in the same list format. The output of the third step is a trained model and a classification report of the model performance.
```python
# Example of a dataset
X1 = [0.5, 0.34, 7.6, ...]
y1 = 1

dataset = [(X1,y1), (X2,y2), ...]


```

## Adding datasets
TODO
## Adding transformations

TODO add tutorial


## Adding models

TODO add tutorial