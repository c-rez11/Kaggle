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
#sns.scatterplot(data=df, x=df['LotArea'], y=df['SalePrice'])
sns.boxplot(x='BldgType', y='SalePrice',data=df)
#sns.violinplot(x='LotShape', y='SalePrice',data=df)

build_type = pd.get_dummies(df['MSSubClass'],drop_first=True) # value 20 is excluded
df = df.drop('Id',axis=1)
df = pd.concat([df,build_type],axis=1)
df.drop(['MSSubClass'],axis=1,inplace=True)

print(df['MSZoning'].value_counts())
zoning = pd.get_dummies(df['MSZoning'],drop_first=True) # value C is excluded
df = pd.concat([df,zoning],axis=1)
df.drop(['MSZoning'],axis=1,inplace=True)
print(df.head())

# LotFrontage -- two outliers to consider dropping
# LotArea -- a few outliers as well
# Street
print(df['Street'].value_counts()) # with so few gravel roads, consider dropping
# Alley
print(df['Alley'].value_counts()) # with so few gravel roads, consider dropping
alley = pd.get_dummies(df['Alley'],drop_first=False) #since N/A means no alley, we should keep both categories
df = pd.concat([df,alley],axis=1)
df.drop(['Alley'],axis=1,inplace=True)
print(df.head())
# LotShape -- I'm really not sure how to interpret 'moderately irregular'
    # versus 'irregular' so this might end up getting thrown out
print(df['LotShape'].value_counts()) 
lot_shape = pd.get_dummies(df['LotShape'],drop_first=True) 
df = pd.concat([df,lot_shape],axis=1)
df.drop(['LotShape'],axis=1,inplace=True)
# LandContour
print(df['LandContour'].value_counts()) 
contour = pd.get_dummies(df['LandContour'],drop_first=True) 
df = pd.concat([df,contour],axis=1)
df.drop(['LandContour'],axis=1,inplace=True)

# Utilities
print(df['Utilities'].value_counts()) # with only 1 observation that isn't
    # all public utilities, we will drop this variable
df.drop(['Utilities'],axis=1,inplace=True)
print(df.head())

# LotConfig
print(df['LotConfig'].value_counts()) # only cul-de-sac seems relevant
# consider dropping everything except dummy for cul-de-sac
print(df['LotConfig'].value_counts()) 
lot_config = pd.get_dummies(df['LotConfig'],drop_first=True) 
df = pd.concat([df,lot_config],axis=1)
df.drop(['LotConfig'],axis=1,inplace=True)
print(df.head())

# LandSlope
print(df['LandSlope'].value_counts()) # moderate and severe appear the same
# see if means are different
print(df.groupby('LandSlope').mean()['SalePrice']) # mod to sev not significantly different
# create boolean column that is either gentle slope or not gentle
df['SevereSlope'] = ((df['LandSlope']=='Mod') | (df['LandSlope']=='Sev')).astype(int)

print(df['SevereSlope'].value_counts())
df.drop(['LandSlope'],axis=1,inplace=True)
print(df.head())

# Neighborhood
print(df['Neighborhood'].value_counts()) # lots of dummies to create
neighborhood = pd.get_dummies(df['Neighborhood'],drop_first=True) 
df = pd.concat([df,neighborhood],axis=1)
df.drop(['Neighborhood'],axis=1,inplace=True)
print(df.head())

# Condition1: Proximity to certain features
print(df['Condition1'].value_counts()) # lots of dummies to create
condition = pd.get_dummies(df['Condition1'],drop_first=True) 
df = pd.concat([df,condition],axis=1)
df.drop(['Condition1'],axis=1,inplace=True)
print(df.head())

# Condition2: Proximity to certain features if more than 1 attribute
print(df['Condition2'].value_counts())
condition2 = pd.get_dummies(df['Condition2'],drop_first=True) 
df = pd.concat([df,condition2],axis=1)
df.drop(['Condition2'],axis=1,inplace=True)
print(df.head())

categorical_col = [
    df['MSSubClass'], df['MSZoning'], df['Street']]

for col in categorical_col








plt.show()