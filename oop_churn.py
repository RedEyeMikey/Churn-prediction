#bài này nếu mn có góp ý thì cứ nhắn cho em 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,r2_score,log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def check_scores(model, X_train, X_test,Y_train,Y_test):
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
  
    
class Churn():
        
    def __init__(self, link,model):
        self.link = link
        self.model= model
        try:
            df=pd.read_csv(filepath_or_buffer=self.link)
        except FileNotFoundError:
            self.link=input("nhap link lai:")
        self.dataframe = df
        
    def show_data(self):
        print(self.dataframe)
        
    def binary_classification(self):
        self.dataframe['Modified_Response']  = self.dataframe['Response'].apply(lambda x : 0 if x<=7 and x>=0 else (1 if x==8 else -1))
        self.dataframe.drop('Response',axis = 1, inplace=True)
    #chỉnh sửa theo dữ liệu của bài toán 
        
    def remove_over_hight_corr (self):
        corr = self.dataframe.corr()
        corr_greater_than_80 = corr[corr>=.8]
        corr_greater_than_80
        self.dataframe= self.dataframe.fillna(self.dataframe.mean())
    
    def split_data(self):
        self.data_np=self.dataframe.to_numpy(dtype=float)
        X= self.data_np[:,:-1]
        Y=self.data_np[:,-1]
        #add this only if your data is 1 value above 0
        # for i in range(len(Y)):
        #     Y[i]=Y[i]-1
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=.25, random_state=1)    
    def edit_train_output(self):
        self.data_np
    def switch_case(self):
        match self.model:
            case 1:
                xgb=XGBClassifier(random_state=0).fit(self.X_train,self.Y_train)
                xgb_predict=xgb.predict(self.X_test)
                print(classification_report(self.Y_test, xgb_predict))
                accuracy_score(self.Y_test,xgb_predict)
                xgb.get_params()
                check_scores(xgb,self.X_train,self.X_test,self.Y_train,self.Y_test)
                
                
            case 2:
                forest= RandomForestClassifier(random_state=0).fit(self.X_train,self.Y_train)
                predict_forest=forest.predict(self.X_test)
                print(classification_report(self.Y_test,predict_forest))
                accuracy_score(predict_forest,self.Y_test)
                check_scores(forest,self.X_train,self.X_test,self.Y_train,self.Y_test)
                
            case 3:
                mlp=MLPClassifier(random_state=0).fit(self.X_train, self.Y_train)
                mlp_predict=mlp.predict(self.X_test)
                print(classification_report(self.Y_test,mlp_predict))
                accuracy_score(mlp_predict,self.Y_test)
                check_scores(mlp,self.X_train,self.X_test,self.Y_train,self.Y_test)
                
            case 4: 
                log=LogisticRegression(random_state=0).fit(self.X_train,self.Y_train)
                predict_log=log.predict(self.X_test)
                print(classification_report(self.Y_test,predict_log))
                accuracy_score(predict_log,self.Y_test)
                check_scores(log,self.X_train,self.X_test,self.Y_train,self.Y_test)
                
            
def import_data():
    print('-------------Churn Prediction Problem (ver 1.0)-------------')
    print('Thông báo trước: file train data cần để giá trị y (output) ở cột cuối cùng')
    link=str(input('nhập địa chỉ (Các dấu cách của địa chỉ cần để là kí tự "/" : '))
    model=-1
    while model<=0 or model>=4:
        model=int(input('Model có sẵn: \n1. XGBoost \n2. random forest\n3. MLP\n4. logistic regession\nnhập model: '))
    return link, model


link,model=import_data()    
test=Churn(link,model)
test.binary_classification()
test.remove_over_hight_corr()
test.split_data()
test.switch_case()