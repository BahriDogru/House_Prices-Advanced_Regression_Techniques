# House Prices Prediction - Kaggle Competition

This project aims to predict house prices based on various features using machine learning models. The dataset comes from the **Kaggle "House Prices - Advanced Regression Techniques" competition**, where the goal is to predict the sale prices of houses based on various attributes.

## Project Overview

The objective of this project is to analyze and predict house prices using regression models. The dataset includes multiple features such as the size of the house, the quality of the neighborhood, and other factors that influence house prices. The project leverages several techniques, including data cleaning, feature engineering, and model selection, to create a predictive model with high accuracy.

### Dataset

The dataset used in this project is the **House Prices - Advanced Regression Techniques** competition dataset on Kaggle. You can find more information and access the dataset using the following link:

[House Prices - Advanced Regression Techniques Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

### Project Steps

1. **Data Preprocessing:**
   - Loaded the dataset and explored it to understand the structure.
   - Handled missing values by either imputing or filling with appropriate values.
   - Identified and dealt with outliers using the Interquartile Range (IQR) method.
   - Transformed categorical variables into numeric using encoding techniques.
   - Scaled numerical features using MinMaxScaler to standardize them.

2. **Exploratory Data Analysis (EDA):**
   - Analyzed the distribution of both numerical and categorical features.
   - Investigated correlations between features and the target variable (SalePrice).
   - Created new features (e.g., total square footage, age of the house, etc.) for better model performance.

3. **Feature Engineering:**
   - Created new binary features like `NEW_PoolStatus`, `NEW_Has_Garage`, and `NEW_Has_Basement`.
   - Grouped the year of construction and remodel into categorical bins for better model interpretation.

4. **Model Building:**
   - Split the data into training and test sets.
   - Built and evaluated multiple models, including Linear Regression, Random Forest, K-Nearest Neighbors, XGBoost, LightGBM, Gradient Boosting, and CatBoost.
   - Compared model performances using **RMSE (Root Mean Squared Error)** as the evaluation metric.
   - The best performing model was **CatBoost** with a final RMSE score of **0.1150** and an R² score of **0.9154** on the test set.

5. **Hyperparameter Tuning:**
   - Used **GridSearchCV** to optimize the hyperparameters of the **CatBoost** model.
   - The best parameters found were:
     - `iterations`: 400
     - `learning_rate`: 0.05
     - `depth`: 6
     - `l2_leaf_reg`: 3
     - `border_count`: 64

6. **Model Evaluation and Submission:**
   - Predicted house prices on the test set.
   - Converted predictions back from the log scale using `np.expm1`.
   - Prepared the submission file in the required format.

### Model Results

- **Best RMSE Score**: 0.1150
- **R² Score**: 0.9154 (91.54% of variance explained)
- The final model predicted house prices with high accuracy, yielding an RMSE of 0.1150, indicating good generalization on unseen data.

### Usage

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/BahriDogru/House_Prices-Advanced_Regression_Techniques.git
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Run the house_prices_prediction.ipynb notebook to see the results and retrain the models.
   
### Links
Kaggle Competition & Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

My Kaggle Notebook: [Link to your Kaggle Notebook] (https://www.kaggle.com/code/bahridgr/predicting-house-prices-with-regression-techniques)

## Conclusion

This project demonstrates the power of machine learning in predicting house prices. The CatBoost model provided the best performance, and through careful data preprocessing, feature engineering, and model tuning, we were able to achieve a highly accurate model. This work is a good starting point for tackling regression problems in the real estate industry.