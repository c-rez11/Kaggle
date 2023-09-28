# This model wasn't used, but contains useful code that helped get me to the submission page.
    # This model also removes insignificant variables based on the initial OLS regression.
    # I found that the RMSE was actually worse, plus my methodology was a bit convoluted,
    # so i stuck with the more straightforward approach in my submission file.

# Housing prices in Ames, IA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/data/train.csv')

print(df.info())
print(df.head())
print("MasVnrType HERE")
print("")
print(df['MasVnrType'].value_counts())
print("")
# Feature Engineering
#sns.histplot(df['LotConfig']) # visualize distribution
#sns.displot(df['SalePrice'])
#sns.scatterplot(data=df, x=df['LotFrontage'], y=df['SalePrice'])
#sns.boxplot(x='BldgType', y='SalePrice',data=df)
#sns.violinplot(x='LotShape', y='SalePrice',data=df)

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

#print(df.info(verbose=True,show_counts=True))

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

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(1900)

#sns.histplot(df['GarageYrBlt']) # visualize distribution
#plt.show()

#MasVnrArea null values

df3 = pd.DataFrame(data=df)
select_columns = ['MasVnrType_None','MasVnrType_Stone', 'MasVnrType_BrkFace', 'MasVnrArea']
df4 = df3[select_columns]
print(df4.head())

masonry_null = df4.isnull().any(axis=1)
null_rows2 = df4[masonry_null]
print(null_rows2) # there are only 8 null rows, so let's drop them

#print(df.info(verbose=True,show_counts=True))

# Final N/A is Lot Frontage

df5 = pd.DataFrame(data=df)
select_columns = ['LotArea', 'LotFrontage']
df6 = df5[select_columns]
print(df6.head())

lot_null = df6.isnull().any(axis=1)
null_rows3 = df6[lot_null]
print(null_rows3) # we should fill in zeroes for this 

#sns.scatterplot(data=null_rows2)
#sns.histplot(df['LotFrontage']) # visualize distribution

# There doesn't seem to be a strong case that I can input zeroes for LotFrontage,
    # so I will drop the rows that contain these null values 
    # UNTIL I can confirm that LotFrontage is not significant.

    # Then, I will drop LotFrontage and add back that extra data


# drop null values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
#print(df.info(verbose=True,show_counts=True))

# remove outliers
# from scipy import stats 
# df = df[(np.abs(stats.zscore(df['SalePrice'])) < 3)]

# drop various variables
df = df.drop('Id', axis=1)
#df = df.drop('OverallQual', axis=1)

# random forest to help narrow down the most important features
from sklearn.model_selection import train_test_split

y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

top_n = 10  # Replace with the number of top features you want to display
top_features = feature_importances.head(top_n)

print("Top", top_n, "Features:")
print(top_features)

#linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# model evaluation
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)

from sklearn import datasets
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

# Linear Regression

y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred_lr = lm.predict(X_test)

# model evaluation
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)

X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

# print("OUTLIER ROW HERE")
# print("")

# # Create a DataFrame with test data and predictions
# test_data_with_predictions = X_test.copy()
# test_data_with_predictions['Predicted'] = y_pred_lr

# # Identify rows where the predicted value is less than zero
# negative_predictions = test_data_with_predictions[test_data_with_predictions['Predicted'] < 0]

# # Print the rows with negative predictions
# print(negative_predictions)

# Extract the t-statistic values from the summary
t_stats = est2.tvalues

# Filter the coefficients DataFrame based on t-statistic absolute value > 2
significant_coeffs = coeff_df[abs(t_stats) > 2]

# Print the coefficients with significant t-statistics
print("SIGNIFICANT COEFFICIENTS BELOW")
print(significant_coeffs.describe())

# remove insignificant variables

# transpose the columns to align with our existing dataframe
col_transpose_sig = pd.DataFrame.transpose(significant_coeffs)
col_transpose_sig['SalePrice'] = [20] # so that SalePrice isn't removed with the below filter
print(significant_coeffs.head())

for col in df:
    if col not in col_transpose_sig.columns:
        df.drop(col, axis=1, inplace=True)
print(df.head())  

# Linear Regression

y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred_lr = lm.predict(X_test)

# model evaluation
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
#print(coeff_df)

X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

# do it all again

# Extract the t-statistic values from the summary
t_stats = est2.tvalues

# Filter the coefficients DataFrame based on t-statistic absolute value > 2
significant_coeffs = coeff_df[abs(t_stats) > 2]

# Print the coefficients with significant t-statistics
print("SIGNIFICANT COEFFICIENTS BELOW")
print(significant_coeffs.describe())

# remove insignificant variables

# transpose the columns to align with our existing dataframe
col_transpose_sig = pd.DataFrame.transpose(significant_coeffs)
col_transpose_sig['SalePrice'] = [20] # so that SalePrice isn't removed with the below filter
print(significant_coeffs.head())

for col in df:
    if col not in col_transpose_sig.columns:
        df.drop(col, axis=1, inplace=True)
print(df.head())  

# Linear Regression

y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred_lr = lm.predict(X_test)

# model evaluation
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
#print(coeff_df)

X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

from sklearn import metrics
MAE_lr = metrics.mean_absolute_error(y_test, y_pred_lr)
MSE_lr = metrics.mean_squared_error(y_test, y_pred_lr)
RMSE_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
exp_var_lr = explained_variance_score(y_test, y_pred_lr)

# Calculate the residuals
residuals_lr = y_test - y_pred_lr

# Create a scatter plot of residuals
plt.scatter(y_pred_lr, residuals_lr)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/unused/LR_residuals.png')
plt.show()

# now random forest

y = df['SalePrice'] # dependent variable
X = df.drop('SalePrice',axis=1) # explanatory variables, minus dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
num_of_var = X.shape[1]
print(num_of_var)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

MAE_rf = metrics.mean_absolute_error(y_test, y_pred_rf)
MSE_rf = metrics.mean_squared_error(y_test, y_pred_rf)
RMSE_rf = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
exp_var_rf = explained_variance_score(y_test, y_pred_rf)

# Calculate the residuals
residuals_rf = y_test - y_pred_rf

# Create a scatter plot of residuals
plt.scatter(y_pred_rf, residuals_rf)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Random Forest: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/unused/RF_residuals.png')
plt.show()

# neural network

# scaling the data
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
# adding the layers
model.add(Dense(num_of_var,activation='relu'))
model.add(Dense(27,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1)) # final layer for our output

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), 
          batch_size=128,epochs=120)
# note on validation: it checks how well the model is performing at each epoch
#   WITHOUT affecting the actual weights of the model
# note on batch_size: the smaller the batch size, longer training will take

losses = pd.DataFrame(model.history.history)
losses.plot() # as we can see in the graph, the loss matches validation loss, meaning we're not overfitting
plt.show()
#   if the validation loss increased as time went on, we know we'd have an overfitting problem

y_pred_nn = model.predict(X_test)

MAE_nn = metrics.mean_absolute_error(y_test, y_pred_nn)

MSE_nn = metrics.mean_squared_error(y_test, y_pred_nn)
RMSE_nn = np.sqrt(metrics.mean_squared_error(y_test, y_pred_nn))
exp_var_nn = explained_variance_score(y_test, y_pred_nn)

# Calculate the residuals
residuals_nn = y_test.values - y_pred_nn.flatten()

# Create a scatter plot of residuals
plt.scatter(y_pred_nn, residuals_nn)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Neural Network: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/unused/NN_residuals.png')
plt.show()

# definition of MSE and RMSE: avg magnitude of errors in a set of predictions WITHOUT considering direction
# RMSE is better if you care about your model's performance against outliers,
    # as RMSE more heavily weights the error of outliers

#print(df['SalePrice'].describe()) # with an average price of ~$500k, our avg error is $100k
# Calculate mean and standard deviation
avg_price = df['SalePrice'].mean()
std_dev = df['SalePrice'].std()

# Print the message with dynamic values
print(f"WITH AN AVG PRICE OF ${avg_price:.0f}, OUR AVG ERROR IS ${std_dev:.0f}")

print("EXPLAINED VARIANCE SCORE")
print(explained_variance_score(y_test, y_pred_nn)) # how much of the data is explained by our model

#plt.scatter(y_test, y_pred_nn) 
#plt.plot(y_test,y_test,'r') # gets a fit line of our predictions
#plt.show()
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
exp_var_xg = explained_variance_score(y_test, y_pred_xg)

# Calculate the residuals
residuals_xg = y_test - y_pred_xg

# Create a scatter plot of residuals
plt.scatter(y_pred_xg, residuals_xg)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("XG Boost: Residuals vs. Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.savefig('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/unused/XG_residuals.png')
plt.show()

# Calculate and print the Mean Squared Error (MSE)
#mse = mean_squared_error(y_test, y_pred)
#print("Mean Squared Error:", mse)


# Visualize feature importance
#xgb.plot_importance(xgb_reg, importance_type='weight')  # 'weight', 'gain', or 'cover'

# Get feature importance as a dictionary
feature_importance = xgb_reg.get_booster().get_score(importance_type='weight')  # 'weight', 'gain', or 'cover'

# Sort the feature importances by their values in descending order
sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Print the top 10 most important features
print("Top 10 Most Important Features:")
for feature, importance in sorted_feature_importance[:10]:
    print(f"Feature: {feature}, Importance: {importance}")


print("LINEAR REGRESSION")
print('MAE:', MAE_lr)
print('MSE:', MSE_lr)
print('RMSE:', RMSE_lr)
print("EXPLAINED VARIANCE SCORE:", exp_var_lr)
print("")

print("RANDOM FOREST")
print('MAE:', MAE_rf)
print('MSE:', MSE_rf)
print('RMSE:', RMSE_rf)
print("EXPLAINED VARIANCE SCORE:", exp_var_rf)
print("")

print("NEURAL NETWORK")
print('MAE:', MAE_nn)
print('MSE:', MSE_nn)
print('RMSE:', RMSE_nn)
print("EXPLAINED VARIANCE SCORE:", exp_var_nn)
print("")

print("XG BOOST")
print('MAE:', MAE_xg)
print('MSE:', MSE_xg)
print('RMSE:', RMSE_xg)
print("EXPLAINED VARIANCE SCORE:", exp_var_xg)

data = {
    "Model": ["LINEAR REGRESSION", "RANDOM FOREST", "NEURAL NETWORK", "XG BOOST"],
    "MAE": [MAE_lr, MAE_rf, MAE_nn, MAE_xg],
    "MSE": [MSE_lr, MSE_rf, MSE_nn, MSE_xg],
    "RMSE": [RMSE_lr, RMSE_rf, RMSE_nn, RMSE_xg],
}

# Create a DataFrame from the dictionary
results = pd.DataFrame(data)

results.to_csv('c:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/output/unused/results.txt', sep='\t', index=False)