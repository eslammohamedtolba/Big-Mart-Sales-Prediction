# Big-Mart-Sales-Prediction

This project is a simple sales prediction model for Big Mart outlets.
It uses a dataset and a machine learning approach to predict sales, achieving an accuracy of 35%. The dataset is included in the code.

## Prerequisites
Before running the code, make sure you have the following prerequisites installed:
- Python 3.x
- Required Python libraries: pandas, scikit-learn, matplotlib, seaborn, xgboost
- Jupyter Notebook or an IDE of your choice

## Overview
The Big Mart Sales Prediction Model is a Python-based project that predicts sales for Big Mart outlets. 
The model uses the XGBoost regressor and various preprocessing techniques to improve the accuracy of the predictions. 
The code includes data loading, data cleaning, data visualization, model training, and evaluation.
The main steps in the code include:
1. **Data Loading**: The dataset is loaded using pandas from the provided "Train.csv" file.
2. **Data Cleaning**: Data cleaning is performed to handle missing values in both numerical and categorical columns.
   Numerical missing values are imputed with the mean, while categorical missing values are imputed with the mode.
4. **Data Visualization**: Several data visualization plots are created to better understand the data.
   This includes distribution plots for features like Item_Outlet_Sales, Item_MRP, Item_Weight, and Item_Visibility.
   Count plots are generated for Item_Type, Outlet_Location_Type, Outlet_Establishment_Year, and Item_Fat_Content.
5. **Data Preprocessing**: Label encoding is applied to convert textual columns into numeric columns.
6. **Model Training**: The XGBoost regressor is used to build the sales prediction model. The dataset is split into training and testing sets.
7. **Model Evaluation**: The model is used to make predictions on the training and testing data. The R-squared (R2) score is used to evaluate the accuracy of the model.


## contribution
If you have suggestions to improve the model's accuracy or any questions, please feel free to contribute. Your contributions are welcome! 


  
