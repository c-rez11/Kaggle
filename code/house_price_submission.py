
# Housing prices in Ames, IA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/data/train.csv')

# brief look at the data
print(df.info())
print(df.head())

# Identify the categorical variables in the dataset
categorical_col = [
    df['MSSubClass'], df['MSZoning'], df['Street'], df['Alley'], df['LotShape'], 
    df['LandContour'], df['Utilities'], df['LotConfig'], df['LandSlope'],
    df['Neighborhood'], df['Condition1'], df['Condition2'], df['BldgType'],
    df['HouseStyle'], df['RoofStyle'], 
    df['RoofMatl'], df['Exterior1st'], df['Exterior2nd'], df['MasVnrType'],
    df['ExterQual'], df['ExterCond'], df['Foundation'], df['BsmtQual'],
    df['BsmtCond'], df['BsmtExposure'], df['BsmtFinType1'], df['BsmtFinType2'],
    df['Heating'], df['HeatingQC'], df['CentralAir'], df['Electrical'],
    df['KitchenQual'], df['Functional'], df['FireplaceQu'], df['GarageType'],
    df['GarageFinish'], df['GarageQual'], df['GarageCond'], df['PavedDrive'],
    df['PoolQC'], df['Fence'], df['MiscFeature'], df['SaleType'], df['SaleCondition']]

# turn the categorical variables into dummy variables
for col in categorical_col:
    dummies = pd.get_dummies(col,drop_first=True)
    dummies = dummies.add_prefix(col.name + '_')
    df = pd.concat([df,dummies],axis=1)
print(df.head())

# transpose the columns to align with our existing dataframe
col_df = pd.DataFrame(categorical_col)
col_transpose = pd.DataFrame.transpose(col_df)
print(col_transpose.head())

# drop the categorical columns; we just created dummy variable columns for them
for col in df:
    if col in col_transpose.columns:
        df.drop(col, axis=1, inplace=True)
print(df.head())     

# Note: This process of transforming the categorical data is a built-in feature of
    # many random forest models, but it's good practice here to do it manually. 

print(df.columns[df.isnull().any()]) # print columns with null values

# Decide what to do with null values

# Garage Year -- about 80 null values

df1 = pd.DataFrame(data=df)
select_columns = ['GarageYrBlt', 'GarageArea']
df2 = df1[select_columns]
print(df2.head())

garage_yr_null = df2.isnull().any(axis=1)
null_rows = df2[garage_yr_null]
print(null_rows) # garages don't exist for the null, so we can fill in something like 1900

sns.histplot(df['GarageYrBlt']) # visualize distribution
plt.show()
# 1900 is the earliest year in our data, so it makes more sense than inputting a zero

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(1900)

#MasVnrArea (Masonry Veneer Area) null values

df3 = pd.DataFrame(data=df)
select_columns = ['MasVnrType_None','MasVnrType_Stone', 'MasVnrType_BrkFace', 'MasVnrArea']
df4 = df3[select_columns]
print(df4.head())

masonry_null = df4.isnull().any(axis=1)
null_rows2 = df4[masonry_null]
print(null_rows2) # based on other similar datapoints, inputting zero makes sense here

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Final null values: Lot Frontage

df5 = pd.DataFrame(data=df)
select_columns = ['LotArea', 'LotFrontage']
df6 = df5[select_columns]
print(df6.head())

lot_null = df6.isnull().any(axis=1)
null_rows3 = df6[lot_null]
print(null_rows3) # again, based on the data, it seems we can input zeroes

df['LotFrontage'] = df['LotFrontage'].fillna(0)


# drop null values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# drop ID before we start training the model
df = df.drop('Id', axis=1)

#result_df = result_df.drop('SalePrice_squared', axis=1)
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn import datasets
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# First model: Linear Regression

y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred_lr = lm.predict(X_test)

# model evaluation
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

from sklearn import metrics
# The competition judges root mean squared error (RMSE), but it's good to see all 3
    # to understand how outlier affects our performance.
MAE_lr = metrics.mean_absolute_error(y_test, y_pred_lr)
MSE_lr = metrics.mean_squared_error(y_test, y_pred_lr)
RMSE_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))

# Calculate the residuals
residuals_lr = y_test - y_pred_lr

# Create a scatter plot of residuals
plt.scatter(y_pred_lr, residuals_lr)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/submission/LR_residuals.png')
plt.show()

# Extract the t-statistic values from the summary
t_stats = est2.tvalues

# We'll keep all variables for now because the other models will do a pretty good job
    # at de-valuing the insignificant variables. My other (unused) model uses only significant variables as comparison.


# Model 2: Random Forest

y = df['SalePrice'] # dependent variable
X_rf = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X_rf, y, test_size=0.30, random_state=42)
print(X_rf.shape[1])

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=500, random_state=42) # ideally, pick n_estimators based on RMSE performance
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

MAE_rf = metrics.mean_absolute_error(y_test, y_pred_rf)
MSE_rf = metrics.mean_squared_error(y_test, y_pred_rf)
RMSE_rf = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))

# Calculate the residuals
residuals_rf = y_test - y_pred_rf

# Create a scatter plot of residuals
plt.scatter(y_pred_rf, residuals_rf)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Random Forest: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/submission/RF_residuals.png')
plt.show()

# neural network

# scaling the data -- necessary for neural networks
from sklearn.preprocessing import MinMaxScaler
y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # we don't scale our test data because we wouldn't know this 
    # in a real-world application

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# Adding the layers. We have 258 variables, so our first layer creates a node for each variable
    # Then, each node's weight is determined and fed into 129 in the next layer.
    # This continues until we have 1 node left for the SalePrice output.
model.add(Dense(258,activation='relu'))
model.add(Dense(129,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1)) # final layer for our output

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), 
          batch_size=128,epochs=60)
# note on validation: it checks how well the model is performing at each epoch
#   WITHOUT affecting the actual weights of the model
# note on batch_size: the smaller the batch size, longer training will take

losses = pd.DataFrame(model.history.history)
losses.plot() # helps us see if we are overfitting the data. Determines how many epochs to add in the model above
plt.show()
#   if the validation loss increased as time went on, we know we'd have an overfitting problem

y_pred_nn = model.predict(X_test)

MAE_nn = metrics.mean_absolute_error(y_test, y_pred_nn)
MSE_nn = metrics.mean_squared_error(y_test, y_pred_nn)
RMSE_nn = np.sqrt(metrics.mean_squared_error(y_test, y_pred_nn))

# Calculate the residuals
residuals_nn = y_test.values - y_pred_nn.flatten()

# Create a scatter plot of residuals
plt.scatter(y_pred_nn, residuals_nn)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Neural Network: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/submission/NN_residuals.png')
plt.show()

# definition of MSE and RMSE: avg magnitude of errors in a set of predictions WITHOUT considering direction
# RMSE is better if you care about your model's performance against outliers,
    # as RMSE more heavily weights the error of outliers

import xgboost as xgb

y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Create the XGBoost regression model
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Fit the model to the training data
xgb_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred_xg = xgb_reg.predict(X_test)

MAE_xg = metrics.mean_absolute_error(y_test, y_pred_xg)
MSE_xg = metrics.mean_squared_error(y_test, y_pred_xg)
RMSE_xg = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xg))

# Calculate the residuals
residuals_xg = y_test - y_pred_xg

# Create a scatter plot of residuals
plt.scatter(y_pred_xg, residuals_xg)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("XG Boost: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/submission/XG_residuals.png')
plt.show()

# save the evaluation of residuals to a .txt file if we want more granular detail than a graph
data = {
    "Model": ["LINEAR REGRESSION", "RANDOM FOREST", "NEURAL NETWORK", "XG BOOST"],
    "MAE": [MAE_lr, MAE_rf, MAE_nn, MAE_xg],
    "MSE": [MSE_lr, MSE_rf, MSE_nn, MSE_xg],
    "RMSE": [RMSE_lr, RMSE_rf, RMSE_nn, RMSE_xg],
}

results = pd.DataFrame(data)

results.to_csv('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/submission/results.txt', sep='\t', index=False)

# Analysis

# While the linear regression does well on average (as seen by mean avg error (MAE)),
    # it doesn't handle outliers well (as seen by root mean squared error, (RMSE)).
    # Both the random forest and XG Boost models perform similarly, while the neural network
    # doesn't perform well--I need to investigate that. 
    # Because we're being evaluated on RMSE, we'll use the model that produces the lowest RMSE: the random forest model


# Kaggle submission

# Test data
df = pd.read_csv('C:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/data/test.csv')

# Preprocessing the data--I should set up a procedure to do this, but for now I've copied and pasted the code from above

categorical_col = [
    df['MSSubClass'], df['MSZoning'], df['Street'], df['Alley'], df['LotShape'], 
    df['LandContour'], df['Utilities'], df['LotConfig'], df['LandSlope'],
    df['Neighborhood'], df['Condition1'], df['Condition2'], df['BldgType'],
    df['HouseStyle'], df['RoofStyle'], 
    df['RoofMatl'], df['Exterior1st'], df['Exterior2nd'], df['MasVnrType'],
    df['ExterQual'], df['ExterCond'], df['Foundation'], df['BsmtQual'],
    df['BsmtCond'], df['BsmtExposure'], df['BsmtFinType1'], df['BsmtFinType2'],
    df['Heating'], df['HeatingQC'], df['CentralAir'], df['Electrical'],
    df['KitchenQual'], df['Functional'], df['FireplaceQu'], df['GarageType'],
    df['GarageFinish'], df['GarageQual'], df['GarageCond'], df['PavedDrive'],
    df['PoolQC'], df['Fence'], df['MiscFeature'], df['SaleType'], df['SaleCondition']]

# turn the categorical variables into dummy variables
for col in categorical_col:
    dummies = pd.get_dummies(col,drop_first=True)
    dummies = dummies.add_prefix(col.name + '_')
    df = pd.concat([df,dummies],axis=1)
#print(df.head())

# transpose the columns to align with our existing dataframe
col_df = pd.DataFrame(categorical_col)
col_transpose = pd.DataFrame.transpose(col_df)
#print(col_transpose.head())

# drop the categorical columns; we just created dummy variable columns for them
for col in df:
    if col in col_transpose.columns:
        df.drop(col, axis=1, inplace=True)

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(1900)

    # Applies to null values in the test data that we have to fill in. 
    # Creating a .csv to take a look at the data in Excel (given there were only 3 rows with
    # null values), I was able to see that inputting zeroes was not out of the ordinary
    # compared to other data in the data set. In the future, I should consider using a regression
    # to predict the value given houses with similar features.

null_var = ['MasVnrArea','LotFrontage', 'BsmtFinSF1','BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
            'GarageCars','GarageArea']
for col in null_var:
    df[col] = df[col].fillna(0)

# Error: Feature names seen at fit time, yet now missing:
    # This means that the model was trained on certain features that didn't appear in the test data,
    # which is usually a result of our categorical data split. Our get_dummies split drops
    # categories without data, so this error occurs because there was no observations of
    # "Condition2_RRAe" (for example) in the test data, but there WAS in the train data.
fit_time = ['Condition2_RRAe', 'Condition2_RRAn','Condition2_RRNn',
            'Electrical_Mix', 'Exterior1st_ImStucc', 'Exterior1st_Stone',
            'Exterior2nd_Other', 'GarageQual_Fa', 'Heating_GasA',
            'Heating_OthW', 'HouseStyle_2.5Fin', 'MiscFeature_TenC',
            'PoolQC_Fa', 'RoofMatl_CompShg', 'RoofMatl_Membran',
            'RoofMatl_Metal', 'RoofMatl_Roll', 'Utilities_NoSeWa']
for col in fit_time:
    df[col] = 0

# Error: Feature names unseen at fit time
    # This is the opposite error from above. Tplt.showhe get_dummies categorical split dropped 
    # categories with no observations. The way the data shook out, 'MSSubClass_150' did not have
    # observations in the train data, but did for the test data. 
df = df.drop('MSSubClass_150', axis=1)

X_test_final = df.drop('Id', axis=1)

X_test_final = X_test_final[X_rf.columns] # rearrange columns to match the fitting process from the rf model

# # The below code is commented out, but it was used to identify the null values in the .fillna(0) we did a few rows above

# print(X_test_final.columns[X_test_final.isnull().any()]) # print columns with null values

# # Filter rows with at least one missing value
# non_zero_rows = X_test_final[X_test_final.isna().any(axis=1)]

# # Display the counts
# print(non_zero_rows) # only 3 rows, and all can be zeroed out

y_pred_rf_final = rf_model.predict(X_test_final) # run the random forest model on our test data

# Convert results to dataframe
y_pred_rf_final_df = pd.DataFrame({'SalePrice': y_pred_rf_final}) 

# Concatenate df and y_pred_rf_final_df
df_concatenated = pd.concat([df, y_pred_rf_final_df], axis=1)

# final submission for Kaggle competition
final_columns = ['Id','SalePrice']
submission = df_concatenated[final_columns]
submission.to_csv('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/submission/submission.csv', index=False)

# results: As of the date of submission (9/27/23), it placed in roughly the 50th percentile. Not bad for a first submission.
