import torch
from sklearn.metrics import classification_report
import pandas as pd
from torch.utils.data import DataLoader

def load_model(model_name):
    """
    Load a model based on the model name.

    Parameters
    ----------
    model_name : str
        Name of the model to load

    Returns 
    -------
    model instance
        
    """

    if model_name == 'vgg11':
        from torchvision.models import vgg11
        model = vgg11()
    else:
        raise ValueError('Model not found: {}'.format(model_name))
    return model    


def evaluate(epochs, lr, batch_size, model,data, folds, data_split):
    """
    Train a model on a dataset.

    Parameters
    ----------
    epochs : int
        Number of epochs to train the model
    lr : float
        Learning rate for the optimizer
    batch_size : int
        Batch size for the DataLoader
    model : model instance
        Model to train
    data : list
    
    folds : int

    data_split : float

    Returns 
    -------
    model instance
        
    """
    from datasets.LQE_dataset import TimeSeriesDataset

    # Create a dataset
    data = TimeSeriesDataset(data)



    # if folds is 1 then we split the data into train and test using data_split ratio
    if folds == 1:
        train_size = int(data_split * len(data))
        test_size = len(data) - train_size

        train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


        # train the model
        model = train(model, train_loader, epochs, lr)

        report = test(model, test_loader)
        # test the model


    else:
        # if folds is larger than 1 we use cross validation
        # we split the data into k folds
        # for each fold we train on k-1 folds and test on the remaining fold
        # we then average the results
        fold_size = len(data) // folds
        fold_reports = []
        for i in range(folds):
            # split data into train and test
            test_data = data[i*fold_size:(i+1)*fold_size]
            train_data = data[:i*fold_size] + data[(i+1)*fold_size:]
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

            # train the model
            model = train(model, train_loader, epochs, lr)

            # test the model
            fold_reports.append(test(model, test_loader))

        # average the results
        report = sum(fold_reports) / len(fold_reports)
        # print(f'Average accuracy: {avg_accuracy}')

    
    return report




def test(model, test_loader):
    """
    Test a model on a dataset.

    Parameters
    ----------
    model : model instance
        Model to test
    test_loader : DataLoader
        DataLoader for the test data

    Returns
    -------
    pd.DataFrame
        classification report from sklearn

    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Initialize lists to store true labels and predictions
    true_labels = []
    predictions = []

    # Disable gradient computation since we are only making predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to the same device as the model
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            # Compute the model output
            outputs = model(inputs)
            
            # Convert outputs to predictions (assuming outputs are raw logits)
            predicted_labels = outputs.argmax(dim=1)

            # Store true labels and predictions
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted_labels.cpu().numpy())

    # Generate a classification report
    report = classification_report(true_labels, predictions, output_dict=True)

    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).T

    return report_df






def train(model, train_loader, epochs, lr):
    """
    Train a model on a dataset.

    Parameters
    ----------
    model : model instance
        Model to train
    train_loader : DataLoader
        DataLoader for the training data
    epochs : int
        Number of epochs to train the model 
    lr : float
        Learning rate for the optimizer

    Returns
    -------
    model instance
        Trained model

    """

    import torch
    import torch.optim as optim
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return model