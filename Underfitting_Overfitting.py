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
iowa_model = DecisionTreeRegressor(random_state=1)

#Fit model
iowa_model.fit(X,y)

print ("First in-smaple predictions:" , iowa_model.predict(X.head()))
print ("Actual target values for those homes:" , y.head().tolist())

# split data into training and validation data, for both features and target
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation MAE: {:,.0f}".format(val_mae))
#------------------------------code start---------------------------------

# max leaf nodes to control overfitting vs underfitting
# define dunction to help comapre MAE score from different values for max_leaf_nodes

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = 100

#optimal number is the lowest MAE from the results. In this case 100 is the optimal number of leaves

# step 2 - Fit model using all the data

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

import numpy as np

final_model.fit(X,y)
print ("Best tree size in-smaple predictions:" , np.floor(final_model.predict(X.head())))
print ("First in-smaple predictions:" , iowa_model.predict(X.head()))
print ("Actual target values for those homes:" , y.head().tolist())