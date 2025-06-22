Dataset
Uses the Kaggle House Prices dataset.

Usage
Install dependencies:

pip install -r requirements.txt

Run the model training script:

python housePricesModel.py


View metrics and predictions output in the console.

Project structure
housePricesModel.py — main training and evaluation script

train.csv — training dataset 



Notes
Missing values are handled via imputation inside a preprocessing pipeline.

Numeric features are scaled; categorical features are one-hot encoded.
