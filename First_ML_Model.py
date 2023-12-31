## RECAP
import pandas as pd
import os

# change current working directory

os.chdir (r"C:\Users\he209945\OneDrive - WA Health\Desktop\VSCode_workspaces")
print(os.getcwd())
# Path of the file to read
iowa_file_path = './ML_journey/data/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)
## RECAP 

#print all the columns
home_data.columns

#set target variable, which corresponds to the sales price
y = home_data['SalePrice']


#create a DataFrame called X holding the predictive features
# Create the list of features below
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','SalePrice']
                 
# Select data corresponding to features in feature_names
X = home_data[feature_names]

## Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
X.head()

# Create a DecisionTreeRegressor and save it iowa_model. 
# import sklearn
import sklearn
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
iowa_model = DecisionTreeRegressor(random_state=1)

#Fit model
iowa_model.fit(X,y)

# Make predictions with the model's predict command using X as the data. 
# Save the results to a variable called predictions

predictions = iowa_model.predict(X)
print (predictions)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

#EXTRA create a new table with predictions

#add predictions to a new column in the data frame
# first create a new data frame called results to store the predictions
results = pd.DataFrame (predictions, columns=['prediction'])
# Merge the results with the original DataFrame

home_data_predictions = pd.concat([X,results],axis=1)
home_data_predictions.head()

# export results to csv. note the purpose is to predict the sale price of 
# houses that has not been sold yet

home_data_predictions.to_csv('./ML_Journey/output.csv')