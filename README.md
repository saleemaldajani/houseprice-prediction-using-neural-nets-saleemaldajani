[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DUGMT0Yz)
# PS3NeuralNetHousePrice


## Overview

This repository contains my final submission for the PS3 House Price Prediction assignment. The goal of the project is to predict the sale prices for 460 houses (provided in `test.csv`) using a model trained on 1000 houses (provided in `train.csv`). The final predictions are saved in a CSV file with the required format.

## Contents

- **HousePriceComparison_Advanced.ipynb**:  
  The main notebook that implements a house price prediction model using three different frameworks:
  - **Keras/TensorFlow**
  - **PyTorch**
  - **JAX (Flax & Optax)**
  
- **predictions.csv Files**:  
  - `predictions_keras_KL.csv`
  - `predictions_pytorch_KL.csv`
  - `predictions_jax_KL.csv`

## Methodology

### Data Preprocessing
- Missing values in both train and test datasets are filled (numeric columns with the median and categorical columns with the mode).
- The target variable (`SalePrice`) is log-transformed to stabilize the dynamic range.
- The top 3 features (by correlation with `SalePrice`) are selected.
- A random interaction feature (product of the top two features) is added to capture non-linear relationships.
- Features are scaled using `StandardScaler`.

### Model Architecture
- A unified neural network architecture is implemented in all three frameworks:
  - **Three hidden layers**: 256, 128, and 64 units respectively.
  - **ReLU activations** and **dropout** (rate = 0.2) are used.
  - A final linear output layer predicts the log-transformed SalePrice.

### Training Strategy
- **Two-Phase Training**:
  - **Phase 1 (10 epochs):** Train using MSE loss only (on log-transformed SalePrice) to learn a stable baseline.
  - **Phase 2 (40 epochs):** Continue training using a combined loss (MSE + KL divergence) with a KL weight (α = 1e-4) to better match the predicted distribution to the actual distribution.
- Learning rates have been tuned for each framework (Keras uses 1e-3; PyTorch and JAX use 1e-4) for stability.

### Evaluation & Visualization
- Separate training and validation loss curves are plotted for each framework.
- Separate histograms of the actual vs. predicted SalePrice distributions are generated.
- A combined histogram plot (with x-axis from 50,000 to 450,000 and y-axis from 0 to 2e-5) is produced for comparison.

## Results

- **Keras/TensorFlow RMSE**: (see output in notebook)
- **PyTorch RMSE**: (see output in notebook)
- **JAX RMSE**: (see output in notebook)

The predictions are saved in the CSV files listed above and follow the required format:
```
ID,SALEPRICE
1461,169000.1
1462,187724.1233
1463,175221
```

## How to Run

1. Clone this repository.
2. Open the `HousePriceComparison_Advanced.ipynb` notebook.
3. Update the file paths for `train.csv` and `test.csv` as needed.
4. Install the required packages:
   - TensorFlow
   - PyTorch
   - JAX, Flax, and Optax
   - scikit-learn
   - seaborn
5. Run all cells in the notebook to train the models, generate summary plots, and produce the predictions CSV files.

## Additional Notes

This implementation incorporates insights from the SAlDajani notebook, including enhanced feature engineering (with a random interaction term) and a two-phase training strategy. These improvements help lower the RMSE and better align the predicted distribution with the actual SalePrice distribution.

