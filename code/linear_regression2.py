# Housing prices in Ames, IA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/User/Desktop/python/c-rez11/Kaggle/Kaggle/data/train.csv')

print(df.info())
print(df.head())

# Feature Engineering
#sns.histplot(df['MSZoning']) # visualize distribution
#sns.displot(df['LotFrontage'])
sns.scatterplot(data=df, x=df['LotFrontage'], y=df['SalePrice'])
#sns.boxplot(x='BldgType', y='SalePrice',data=df)
#sns.violinplot(x='LotShape', y='SalePrice',data=df)

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


for col in categorical_col:
    dummies = pd.get_dummies(col,drop_first=True)
    df = pd.concat([df,dummies],axis=1)
    #df.drop(col,axis=1,inplace=True)
#for column in df:

print(df.head())

col_df = pd.DataFrame(categorical_col)
col_transpose = pd.DataFrame.transpose(col_df)
print(col_transpose.head())

for col in df:
    if col in col_transpose.columns:
        df.drop(col, axis=1, inplace=True)
#df.drop('Id', axis=1, inplace=True)
print(df.head())        

# # Principle Component Analysis -- determine which variables best explain the variance
# from sklearn.preprocessing import StandardScaler # scale the data
# scaler = StandardScaler()
# scaler.fit(df)
# scaled_data = scaler.transform(df)

# # from sklearn.decomposition import PCA 
# pca = PCA(n_components=1) #one component, maybe change that later
# pca.fit(scaled_data)
# x_pca = pca.transform(scaled_data)
# scaled_data.shape

# df_comp = pd.DataFrame(pca.components_,column)
print(df.info(verbose=True,show_counts=True))

print(df.columns[df.isnull().any()]) # print columns with null values



plt.show()