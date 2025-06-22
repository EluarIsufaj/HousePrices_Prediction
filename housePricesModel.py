import numpy as np
import pandas
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
ds = pandas.read_csv('train.csv')


num_cols = ds.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('MSSubClass')
num_cols.remove('SalePrice')
num_cols.remove('Id')
cat_cols = ds.select_dtypes(include=['object', 'category']).columns.tolist() + ['MSSubClass']


num_imputer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_imputer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_imputer, num_cols),
    ('cat', cat_imputer, cat_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearRegression())
])


X = ds.drop(columns=['SalePrice'])
y = ds['SalePrice']


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



testDf = pd.read_csv('test.csv')



pipeline.fit(X_train, y_train)


predictions = pipeline.predict(X_val)





plt.scatter(y_val, predictions, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted on Validation Set')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.show()




mae = mean_absolute_error(y_val, predictions)
mse = mean_squared_error(y_val, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, predictions)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")




