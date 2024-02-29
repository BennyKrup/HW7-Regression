# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import sklearn
from regression import (logreg, utils)
#from sklearn.preprocessing import StandardScaler


def test_prediction():
    # Load data
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Train model
    np.random.seed(42)
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.001, tol=0.00001, max_iter = 300, batch_size = 100)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()

    # Make predictions
    predictions_train = (log_model.make_prediction(X_train) >= 0.5).astype(int)
    predictions_val = (log_model.make_prediction(X_val) >= 0.5).astype(int)

    # Check data structures working correctly
    assert predictions_train.shape == y_train.shape, "Mismatch in train predictions and actual labels shape"
    assert predictions_val.shape == y_val.shape, "Mismatch in validation predictions and actual labels shape"

    # Check better than chance
    train_accuracy = sklearn.metrics.accuracy_score(y_train, predictions_train)
    val_accuracy = sklearn.metrics.accuracy_score(y_val, predictions_val)
    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {val_accuracy}")
    assert train_accuracy > 0.5, "Train accuracy not better than chance"
    assert val_accuracy > 0.5, "Validation accuracy not better than chance"

def test_loss_function():

		log_model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.001, tol=0.1, max_iter=1000, batch_size=264)

		# test case
		X_test = np.array([[1, 1], [1, -1]])
		y_true = np.array([1, 0])
		log_model.W = np.array([0.5, 1, 0.5])  # model weights

		# Make predictions
		y_pred = log_model.make_prediction(X_test)

		# Calculate loss using the model's loss function
		loss = log_model.loss_function(y_true, y_pred)

		# Calculate the expected loss manually
		expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

		# Assert that the calculated loss is close to the expected loss
		assert np.isclose(loss, expected_loss), f"Loss function did not return expected value. Got {loss}, expected {expected_loss}"


def test_gradient():
    log_model = logreg.LogisticRegressor(num_feats=2)
    X_test = np.array([[1, 1], [1, -1]])
    y_true = np.array([1, 0])
    log_model.W = np.array([0.5, 1, 0.5])  # model weights

    # Make predictions
    y_pred = log_model.make_prediction(X_test)

    # Calculate gradient using the model's function
    gradient = log_model.calculate_gradient(y_true, X_test)
    expected_gradient = np.dot(X_test.T, (y_pred - y_true)) / y_true.size #manual calculation

    # Assert that the calculated gradient is close to the expected gradient
    assert np.allclose(gradient, expected_gradient), f"Gradient calculation did not return expected values. Got {gradient}, expected {expected_gradient}"

def test_training():
    log_model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.1, tol=0.001, max_iter=10, batch_size=2)

    X_train = np.array([[1, 1], [1, -1], [1, 0], [1, 2]])
    y_train = np.array([1, 0, 0, 1])
    initial_weights = np.array([0.0, 0.5, -0.5])  # model weights
    log_model.W = initial_weights.copy()

    # Train the model
    log_model.train_model(X_train, y_train, X_train, y_train)  
    # Check that weights have been updated
    assert not np.array_equal(log_model.W, initial_weights), "Model weights did not change after training."

    # Check that training loss has decreased
    assert log_model.loss_hist_train[-1] < log_model.loss_hist_train[0], "Training loss did not decrease after training."





# %%
#change to current folder
import os
#/Users/beniaminkrupkin/Desktop/BMI203/HW7-Regression
os.chdir('/Users/beniaminkrupkin/Desktop/BMI203/HW7-Regression')

# %%
#test
test_prediction()
test_loss_function()
test_gradient()
test_training()
print("All tests passed!")

# %%

# %%
