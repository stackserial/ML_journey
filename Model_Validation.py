
import pandas as pd
import os

# ----------------continue from the previous module code ----------------------------
# change current working directory

os.chdir (r"C:\Users\he209945\OneDrive - WA Health\Desktop\VSCode_workspaces")
print(os.getcwd())
# Path of the file to read
iowa_file_path = './ML_journey/data/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)
## RECAP 

#set target variable, which corresponds to the sales price
y = home_data['SalePrice']

#create a DataFrame called X holding the predictive features
# Create the list of features below
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
                 
# Select data corresponding to features in feature_names
X = home_data[feature_names]



# Create a DecisionTreeRegressor and save it iowa_model. 
# import sklearn
import sklearn
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
iowa_model = DecisionTreeRegressor()

#Fit model
iowa_model.fit(X,y)

print ("First in-smaple predictions:" , iowa_model.predict(X.head()))
print ("Actual target values for those homes:" , y.head().tolist())

#-------------------- Code start --------------------------------------------------------

# Measure the performance of your model, so you can test and compare alternatives.




# 1 Split your data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

# 2 Define / spefcify the  model
iowa_model = DecisionTreeRegressor(random_state=1)
#Fit model
iowa_model.fit(train_X, train_y)

# 3 Make Predictions with Validation data
val_predictions = iowa_model.predict(val_X)


# print the top few validation predictions
print(iowa_model.predict(val_X.head()))
# print the top few actual prices from validation data
print(y.head().tolist())


# 4 Calculate MAE

# calculate MAE (mean absolute error)
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error (val_y, val_predictions)

print(val_mae)