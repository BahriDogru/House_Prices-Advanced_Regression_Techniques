import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

##################################
# Görev 1: Keşifçi Veri Analiz
##################################

########################################################
########################################################
# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
# Data set Link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

test = pd.read_csv("datasets/House Prices - Advanced Regression Techniques/test.csv")
train = pd.read_csv("datasets/House Prices - Advanced Regression Techniques/train.csv")


frames = [train, test]
df = pd.concat(frames, ignore_index=True)

def check_data(dataframe):
    print("########################## HEAD ##########################")
    print(dataframe.head())
    print("########################## INFO ##########################")
    print(dataframe.info())
    print("########################## SHAPE ##########################")
    print(dataframe.shape)
    print("########################## ISNULL(?) ##########################")
    print(dataframe.isnull().sum())
    print("########################## DESCRIBE ##########################")
    print(dataframe.describe().T)
    print("####################################################")
check_data(df)

########################################################
########################################################
# Adım 2:  Numerik ve kategorik değişkenleri yakalayınız
df.info()

def grab_col_names(dataframe, cat_th=16, car_th=25):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


########################################################
########################################################
# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)



########################################################
########################################################
# Adım 4:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.25,  0.50,  0.75,  0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    num_summary(df, col)


########################################################
########################################################
####  Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "SalePrice", cat_cols)


########################################################
########################################################
#  Adım 6: Aykırı gözlem var mı inceleyiniz

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def check_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outliers(df,col))


########################################################
########################################################
# Adım 7: Eksik gözlem var mı inceleyiniz.

df.isna().sum().sort_values(ascending=False).head(20)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1,keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df)



##################################
# Görev 2: Feature Engineering
##################################

#  Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız

# Outliers
def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if check_outliers(df,col):
        replace_with_threshold(df, col)

df_level1 = df.copy()

# Missing Values
df.isna().sum().sort_values(ascending=False).head(20)

df["PoolQC"].dtypes
df["PoolQC"].value_counts()
df["PoolQC"].fillna("Unknown", inplace=True)

df["MiscFeature"].dtypes
df["MiscFeature"].value_counts()
df["MiscFeature"].fillna("Unknown", inplace=True)

df["Alley"].dtypes
df["Alley"].value_counts()
df["Alley"].fillna("No_alley_access", inplace=True)

df["Fence"].dtypes
df["Fence"].value_counts()
df["Fence"].fillna("No_Fence", inplace=True)

df["MasVnrType"].dtypes
df["MasVnrType"].value_counts()
df["MasVnrType"].fillna("Unknown", inplace=True)

df["FireplaceQu"].value_counts()
df["FireplaceQu"].fillna("No_Fireplace", inplace=True)

df["LotFrontage"].describe()
df["LotFrontage"].fillna(df["LotFrontage"].median(), inplace=True)


Garage_missing_variables = ["GarageCond", "GarageFinish", "GarageQual", "GarageType"]
missing_rows = df[Garage_missing_variables].isnull().all(axis=1)
for variable in Garage_missing_variables:
    df.loc[missing_rows, variable] = "Unknown"

df["GarageCond"].fillna(df["GarageCond"].mode()[0], inplace=True)

df["GarageYrBlt"].dtypes
df["GarageYrBlt"].fillna("Unknown", inplace=True)


Bsmt_missing_variables = ["BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual","BsmtCond"]
missing_rows_bsmt = df[Bsmt_missing_variables].isnull().all(axis=1)
for variable in Bsmt_missing_variables:
    df.loc[missing_rows_bsmt, variable] = "Unknown"


df["MasVnrArea"].describe()
df["MasVnrArea"].value_counts()
df["MasVnrType"].value_counts()
df.groupby("MasVnrType")["MasVnrArea"].mean()
df.loc[df["MasVnrArea"].isna(), "MasVnrArea"] = df.groupby("MasVnrType")["MasVnrArea"].transform("mean")


columns_to_fill = [
    'MSZoning', 'BsmtExposure', 'BsmtCond', 'GarageQual', 'BsmtQual','PoolQC', 'BsmtFullBath', 'BsmtHalfBath', 'Utilities',
    'Functional', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2','GarageFinish','BsmtFinType2','Electrical',
    'BsmtFinSF1', 'KitchenQual', 'Exterior2nd', 'Exterior1st', 'GarageCars', 'SaleType'
]

for var in columns_to_fill:
    if df[var].dtypes == "O":
        df[var].fillna(df[var].mode()[0], inplace=True)
    else:
        df[var].fillna(df[var].median(), inplace=True)



df.isna().sum().sort_values(ascending=False).head(30)

df_without_missing = df
########################################################
########################################################
#  Adım 2: Yeni değişkenler oluşturunuz

df_corr = df[num_cols].corr()
plt.figure(figsize=(15,12))
sns.heatmap(df_corr, cmap="RdBu")
plt.title("Correlation Heatmap")
plt.show()

#MSSubClass=The building class
df["MSSubClass"].dtypes
df["MSSubClass"] = df["MSSubClass"].astype(str)


#Year and month sold are transformed into categorical features.
df['YrSold'].value_counts()
df['MoSold'].value_counts()
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)


# Adding total sqfootage feature
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

df["NEW_PoolStatus"] = np.where((df["PoolArea"] == 0) & (df["PoolQC"] == "Unknown"), 0, 1)

df["NEW_Has_Garage"] = np.where((df["GarageCars"] == 0) &
                               (df["GarageArea"] == 0) &
                               (df["GarageFinish"] == "Unknown") &
                               (df["GarageType"] == "Unknown") &
                               (df["GarageQual"] == "Unknown") &
                               (df["GarageCond"] == "Unknown") &
                               (df["GarageYrBlt"] == "Unknown"), 0, 1)

df["NEW_Has_Basement"] = np.where((df["TotalBsmtSF"] == 0) &
                                 (df["BsmtUnfSF"] == 0) &
                                 (df["BsmtFinSF1"] == 0) &
                                 (df["BsmtFinSF2"] == 0) &
                                 (df["BsmtFinType1"] == "Unknown") &
                                 (df["BsmtCond"] == "Unknown") &
                                 (df["BsmtQual"] == "Unknown") &
                                 (df["BsmtExposure"] == "Unknown") &
                                 (df["BsmtFinType2"] == "Unknown"), 0, 1)


df["YearBuilt"].min()
df["YearBuilt"].max()
df["YearBuilt"].describe()
bins=[1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
labels = [f"{bins[i]}_{bins[i+1]}" for i in range(len(bins)-1)]

df["NEW_Build_Yr_Range"] = pd.cut(df["YearBuilt"].dropna(), bins=bins, labels=labels)


df["YearRemodAdd"].min()
df["YearRemodAdd"].max()
df["YearRemodAdd"].describe()
bins2=[1950, 1960, 1970, 1980, 1990, 2000, 2010]
labels2 = [f"{bins2[i]}_{bins2[i+1]}" for i in range(len(bins2)-1)]

df["NEW_YearRemodAdd_Range"] = pd.cut(df["YearRemodAdd"].dropna(), bins=bins2, labels=labels2)



df["NEW_OverallScore"] = df["OverallQual"] * df["OverallCond"]

current_year = df["YearBuilt"].max() + 1
df["NEW_HouseAge"] = current_year - df["YearBuilt"]


df["NEW_RestorationAge"] = current_year - df["YearRemodAdd"]


df_with_new_feature = df

########################################################
########################################################
# Adım 3:  Rare Encoder uygulayınız


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.02)

df_with_rare_encoder = df


########################################################
########################################################
#  Adım 4:  Encoding işlemlerini gerçekleştiriniz

cat_cols, cat_but_car, num_cols = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_columns = [col for col in train.columns if (df[col].nunique() == 2) and df[col].dtypes == "O"]

for col in binary_columns:
    label_encoder(df,col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in cat_cols if col not in binary_columns]
df = one_hot_encoder(df, ohe_cols)

df.shape
df.head()

df_with_encode = df


########################################################
########################################################
#  Adım 5: Scalling


mms = MinMaxScaler()
numerical_columns = [col for col in df.columns if df[col].dtypes in [np.float64,np.int32, np.int64] and col not in ["SalePrice", "Id"]]
df[numerical_columns] = mms.fit_transform(df[numerical_columns])

df.head()


##################################
# Görev 3: Model Kurma
##################################
#  Adım 1:  Train ve Test verisini ayırınız

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]


y = np.log1p(train_df[["SalePrice"]])
X = train_df.drop(["Id", "SalePrice"], axis=1)


models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=15)),
    ('KNN', KNeighborsRegressor()),
    ('XGBoost', XGBRegressor(random_state=15)),
    ('LightGBM', LGBMRegressor(random_state=15, verbose=-1)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=15)),
    ('CatBoost', CatBoostRegressor(verbose=False, random_state=15))
]

print("Model Performance Results (5-Fold Cross Validation)")
for model_name, model in models:
    rmse = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"{model_name} - RMSE:{rmse: .4f}")

# Model Performance Results (5-Fold Cross Validation)
# Linear Regression - RMSE: 38897999195.1867
# Random Forest - RMSE: 0.1379
# KNN - RMSE: 0.1810
# XGBoost - RMSE: 0.1407
# LightGBM - RMSE: 0.1311
# Gradient Boosting - RMSE: 0.1286
# CatBoost - RMSE: 0.1214

################################################
# Model Selection and Hyperparameters Optimization
###############################################

# CatBoost modelini devam ediyoruz.
catboost_model = CatBoostRegressor(verbose=False)

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.20, random_state=15)

param_grid = {
    'iterations': [400, 500, 700],  # Iterasyon sayısı
    'learning_rate': [0.07, 0.05,],  # Öğrenme oranı
    'depth': [4, 6, 7],  # Ağaç derinliği
    'l2_leaf_reg': [1, 3, 5],  # L2 düzenleme
    'border_count': [50, 64],  # Sınır sayısı
}

grid_search = GridSearchCV(estimator=catboost_model,
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi negatif MSE skoru:", grid_search.best_score_)
# {'border_count': 64, 'depth': 4, 'iterations': 700, 'l2_leaf_reg': 5, 'learning_rate': 0.05}

final_model = catboost_model.set_params(**grid_search.best_params_, random_state=15).fit(X_train,y_train)

y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = final_model.score(X_test, y_test)

print(f"RMSE:{rmse: .4f}, R2:{r2: .4f}")
# RMSE: 0.1150, R2: 0.9154


y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

comparison = pd.DataFrame({
    "Real SalePrice": y_test_real.values.flatten(),
    "Predict SalePrice": y_pred_real.flatten()
})
print(comparison.head(10))


#################################################
# Submission
#################################################

X_test_submission = test_df.drop(["Id", "SalePrice"], axis=1)
test_predictions = final_model.predict(X_test_submission)


test_predictions = np.expm1(test_predictions)

submission = pd.DataFrame({
    "Id": test["Id"].astype(int),
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)