# autoTS
Automatized pipeline for TS analysis

## Available datasets

- LQE dataset: [link]() TODO add link
- NILM datasets: [link](http://sensorlab.ijs.si/archive/energy-knowledge-graph/harmonized.tar.gz) (80 GB)

## Available models

- VGG16: [link](https://arxiv.org/abs/1409.1556) (pytorch implementation)
- InceptionTime: [link](https://arxiv.org/abs/1909.04939) (tensorflow implementation)

## Available transformations

- GADF
- GASF
- MTF
- RP




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
3. Setup the configuration in the [config.py](src/config.py) file
4. Make sure that the dataset is present in the data folder and the path to the dataset is specified in the [config.py](src/config.py) file
5. Run the pipeline with the following command:
```bash
    python run.py
```

# Pipeline structure
The pipeline consists of 3 main steps: dataset parsing, data transformation and model training and evaluation. Each of these steps can be customized by adding new datasets, transformations and models. The output of the first step is a python list of tuples containing (X,y) pairs, where X is the input data and y is the target variable. For example a time series classification task would contain a list X of length N representing a time series of length N, and y containing the label for the time series. The output of the second step is a transformed version of the input data, in the same list format. The output of the third step is a trained model and a classification report of the model performance.




## Adding datasets

In order to add a new dataet, you need to create a new parser in the parsers folder. The file should contain a function that returns a list of tuples (X,y) where X is the input data and y is the target variable. The function should have the following signature:

```python
def parse_datasetName(path_to_data:str):
    return [(X1,y1), (X2,y2), ...]
```

The file should be named DATASET_NAME_parser.py and the function should be called parse() (check existing parsers for examples), where DATASETNAME is the name of the dataset. The function should be able to parse the data from the path_to_data and return a list of tuples (X,y) where X is the input data and y is the target variable.


### Warning!!!!!!!!!!!!!!!!!!!!!!!!!!
If the parser is not named exactly as described, the pipeline will not be able to find it. For example for LQE dataset the parser should be named LQE_parser.py and the function should be called parse() and to use this dataset we set DATASET to 'LQE' in the config file.

Before running the pipeline the path to the new dataset should be specified in the config file.

### Example of a dataset
```python
X1 = [0.5, 0.34, 7.6, ...]
y1 = 1

dataset = [(X1,y1), (X2,y2), ...]
```

If your parser requires additional packages, you should add them with a locked version to the [requirements.txt](requirements.txt) file. 



## Adding transformations
To add a new transformation, you need to add it to the transformations folder. The file should contain a function that takes a list of inputs and returns a list of transformed inputs. The function should have the following signature:

```python
def transform(data:List[list]):
    return [transformed_data1, transformed_data2, ...]
```

Same as with the parsers, the file should be named TRANSFORMATION_NAME.py and the function should be called transform() check existing [transformations](transformations/GADF.py)  for examples, where TRANSFORMATION_NAME is the name of the transformation.

If your transformation requires additional packages, you should add them with a locked version to the [requirements.txt](requirements.txt) file.

## Adding models
To add a new model you need to add a wrapper for the model in the models folder. The file should contain a class that inherits from the [BaseModelWrapper](models/BaseModelWrapper.py) class. The class should implement the abstract methods specified in the BaseModelWrapper class. The file should be named MODEL_NAME_wrapper.py and the class should be named MODEL_NAME_wrapper, where MODEL_NAME is the name of the model as specified in the config file in the MODEL variable. Depending on the model used you should install the [requirements_tensorflow.txt](requirements_tensorflow.txt) or [requirements_torch.txt](requirements_torch.txt) file. If your model requires a different version of pytorch or tensorflow or some other dependencies, you should add a new `requirements_{MODEL}.txt` file with the locked version of the packages. To see an example of a model wrapper check the [VGG16_wrapper.py](models/VGG16_wrapper.py) file for a pytorch model and [InceptionTime_wrapper.py](models/InceptionTime_wrapper.py) for a tensorflow implementation.

If the model requires a specific input shape to function properly, you should add a check in the `train_test_split` method of the model wrapper to ensure that the input data has the correct shape. If the input data does not have the correct shape, the method should transform the data to the correct shape. An example can be seen in [VGG16_wrapper.py](models/VGG16_wrapper.py) with the `preprocess_vgg` function.


## Running pipeline
After installing the required dependencies and adding the necessary parsers, transformations,models and datasets you can run the pipeline.

To run the pipeline, you need to set the configuration in the [config.py](src/config.py) file. After the configuration is set, you can run the pipeline with the following command:
```bash
python run.py
```

The result of the pipeline should be a classification report of the model performance. A config.json file will be created in the results folder containing the configuration of the pipeline. The results of the pipeline will be saved in the results folder with the name of the dataset, transformation, model used and date. The results will contain the classification report of the model performance, a config.json file containing the specific configuration of the pipeline and a model file containing the trained model. The results folder will be created if it does not exist. 