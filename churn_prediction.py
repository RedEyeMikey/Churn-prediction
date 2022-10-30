
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,r2_score,log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import dataloader
#import data
data_df= pd.read_csv('train1.csv')
data_df.head()

data_df.shape

data_df.columns.values

#data_df=data_df.drop(labels='Id',axis=1)

data_df['Response'].value_counts()

#churn plot
sns.countplot(x=data_df['Response'])

data_df['Modified_Response']  = data_df['Response'].apply(lambda x : 0 if x<=7 and x>=0 else (1 if x==8 else -1))
sns.countplot(x= data_df['Modified_Response'])

data_df['Modified_Response'].value_counts()

# I just checked correlated feature with greater than .8 here 
corr = data_df.corr()
corr_greater_than_80 = corr[corr>=.8]
corr_greater_than_80

plt.figure(figsize=(12,8))
sns.heatmap(corr_greater_than_80, cmap="Reds");

#no need for changes as they don't get affected by correlation much because of their non parametric nature

# Dropping old response columns
data_df.drop('Response',axis = 1, inplace=True)

missing_val_count_by_column = data_df.isna().sum()/len(data_df)

print(missing_val_count_by_column[missing_val_count_by_column > 0.4].sort_values(ascending=False))

#drop data contain >=40% of null data
data_df = data_df.dropna(thresh=data_df.shape[0]*0.4,how='all',axis=1)

#drop product_info_2 does not give out important information 
#data_df.drop('Product_Info_2',axis=1,inplace=True)

#fill out missing value
data_df= data_df.fillna(data_df.mean())

data_np=data_df.to_numpy(dtype=float)
print(data_np)

#split data and train test split
X= data_np[:,:-1]
Y=data_np[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=1)

np.shape(X_test)

np.shape(Y_test)

def check_scores(model, X_train, X_test):
  # Making predictions on train and test data

  train_class_preds = model.predict(X_train)
  test_class_preds = model.predict(X_test)

  # Get the probabilities on train and test
  train_preds = model.predict_proba(X_train)[:,1]
  test_preds = model.predict_proba(X_test)[:,1]


  # Calculating accuracy on train and test
  train_accuracy = accuracy_score(Y_train,train_class_preds)
  test_accuracy = accuracy_score(Y_test,test_class_preds)

  print("The accuracy on train dataset is", train_accuracy)
  print("The accuracy on test dataset is", test_accuracy)
  print()
  # Get the confusion matrices for train and test
  train_cm = confusion_matrix(Y_train,train_class_preds)
  test_cm = confusion_matrix(Y_test,test_class_preds )

  print('Train confusion matrix:')
  print( train_cm)
  print()
  print('Test confusion matrix:')
  print(test_cm)
  print()

  # Get the roc_auc score for train and test dataset
  train_auc = roc_auc_score(Y_train,train_preds)
  test_auc = roc_auc_score(Y_test,test_preds)

  print('ROC on train data:', train_auc)
  print('ROC on test data:', test_auc)
  
  # Fscore, precision and recall on test data
  f1 = f1_score(Y_test, test_class_preds)
  precision = precision_score(Y_test, test_class_preds)
  recall = recall_score(Y_test, test_class_preds) 
  
  
  #R2 score on train and test data
  train_log = log_loss(Y_train,train_preds)
  test_log = log_loss(Y_test, test_preds)

  print()
  print('Train log loss:', train_log)
  print('Test log loss:', test_log)
  print()
  print("F score is:",f1 )
  print("Precision is:",precision)
  print("Recall is:", recall)
  list=[train_auc,test_auc,train_accuracy,test_accuracy,train_log,test_log,f1,precision,recall]
  return list

#grid search
def grid_search(model, parameters, X_train, Y_train):
      #Doing a grid
  grid = GridSearchCV(estimator=model,
                       param_grid = parameters,
                       cv = 2, verbose=3, scoring='roc_auc')
  #Fitting the grid 
  grid.fit(X_train,Y_train)
  print()
  print()
  # Best model found using grid search
  optimal_model = grid.best_estimator_
  print('Best parameters are: ')
  print( grid.best_params_)

  return optimal_model

#Multi layer perceptron (Neural network)
mlp=MLPClassifier(random_state=0).fit(X_train, Y_train)
X_test_1d=mlp.predict(X_test)

mlp.get_params

print(classification_report(Y_test,X_test_1d))
accuracy_score(X_test_1d,Y_test)

list_mlp=check_scores(mlp,X_train,X_test)

#logistic regression
log=LogisticRegression(random_state=0).fit(X_train,Y_train)
log_X_test_1d=log.predict(X_test)

print(classification_report(Y_test,log_X_test_1d))
accuracy_score(log_X_test_1d,Y_test)

list_log=check_scores(log,X_train,X_test)

#Random Forest (default)
forest= RandomForestClassifier(random_state=0).fit(X_train,Y_train)
predict_forest=log.predict(X_test)
print(classification_report(Y_test,predict_forest))
accuracy_score(predict_forest,Y_test)

list_forest=check_scores(forest,X_train,X_test)

#Random Forest (finetune/optimal)
param_grid = { 
    'n_estimators': [200, 500],
    'max_depth' : [15,20],
}
grid_forest=grid_search(RandomForestClassifier(),param_grid,X_train,Y_train)
grid_forest_predict=grid_forest.predict(X_test)

print(classification_report(Y_test, grid_forest_predict))
accuracy_score(Y_test,grid_forest_predict)

list_forest_finetune=check_scores(grid_forest,X_train,X_test)

#XGBoost model
xgb=XGBClassifier(random_state=0).fit(X_train,Y_train)
xgb_predict=xgb.predict(X_test)
print(classification_report(Y_test, xgb_predict))
accuracy_score(Y_test,xgb_predict)

xgb.get_params()

list_xgb=check_scores(xgb,X_train,X_test)

#XGBoost (finetune/optimal)
xgb_parameters = {'max_depth': [1,3,5], 'n_estimators': [2,5,10], 'learning_rate': [.01 , .1, .5]}
xgb_optimal = grid_search(XGBClassifier(), xgb_parameters,X_train,Y_train)

xgb_optimal_predict=xgb_optimal.predict(X_test)
print(classification_report(Y_test, xgb_optimal_predict))
accuracy_score(Y_test,xgb_optimal_predict)

xgb_optimal.get_params()

list_xgb_finetune=check_scores(xgb_optimal,X_train,X_test)

#Stacking Model
stack_classifier= StackingClassifier(classifiers =[xgb,log,grid_forest,mlp],meta_classifier=RandomForestClassifier(),use_probas=True, use_features_in_secondary=True)
stack_model=stack_classifier.fit(X_train,Y_train)

stack_predict=stack_model.predict(X_test)
accuracy_score(Y_test,stack_predict)
print(classification_report(Y_test,stack_predict))

list_stacking=check_scores(stack_model,X_train,X_test)

from pycaret.classification import *
s = setup(data_df, target = 'Modified_Response',use_gpu=True)

best = compare_models()

print(best)

list_pycaret=check_scores(best,X_train,X_test)

plot_model(best, plot = 'class_report')

best.get_all_params()

predict_model(best)

df=pd.DataFrame([list_mlp,list_log,list_forest,list_forest_finetune,list_xgb,list_xgb_finetune,list_stacking,list_pycaret])
df