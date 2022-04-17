# %% [markdown]
# Importing libs

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:07:38.683826Z","iopub.execute_input":"2022-04-17T02:07:38.684765Z","iopub.status.idle":"2022-04-17T02:07:39.885866Z","shell.execute_reply.started":"2022-04-17T02:07:38.684639Z","shell.execute_reply":"2022-04-17T02:07:39.885154Z"}}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:08:25.702782Z","iopub.execute_input":"2022-04-17T02:08:25.703059Z","iopub.status.idle":"2022-04-17T02:08:25.738972Z","shell.execute_reply.started":"2022-04-17T02:08:25.703031Z","shell.execute_reply":"2022-04-17T02:08:25.738101Z"}}
## load dataset
df=pd.read_csv(r'../input/pima-indians-diabetes-database/diabetes.csv')
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:09:56.013476Z","iopub.execute_input":"2022-04-17T02:09:56.013813Z","iopub.status.idle":"2022-04-17T02:09:56.027352Z","shell.execute_reply.started":"2022-04-17T02:09:56.013779Z","shell.execute_reply":"2022-04-17T02:09:56.026759Z"}}
## statistic of dataset
diabetes=df[df['Outcome']==1].shape
non_diabetes=df[df['Outcome']==0].shape
print("Has Diabeties {}, Does not have Diabetes {}".format(diabetes,non_diabetes))

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:11:47.492870Z","iopub.execute_input":"2022-04-17T02:11:47.493160Z","iopub.status.idle":"2022-04-17T02:11:48.762641Z","shell.execute_reply.started":"2022-04-17T02:11:47.493130Z","shell.execute_reply":"2022-04-17T02:11:48.762069Z"}}
# plot bars
histo = df.hist(figsize = (10,10))

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:12:38.497033Z","iopub.execute_input":"2022-04-17T02:12:38.497701Z","iopub.status.idle":"2022-04-17T02:12:38.508229Z","shell.execute_reply.started":"2022-04-17T02:12:38.497661Z","shell.execute_reply":"2022-04-17T02:12:38.507457Z"}}
## count of zero entries 
bp_zeros=df[df['BloodPressure']==0].shape
bmi_zeros=df[df['BMI']==0].shape
insulin_zeros=df[df['Insulin']==0].shape
glucose_zeros=df[df['Glucose']==0].shape
skin_zeros=df[df['SkinThickness']==0].shape
print('Zero Counts of BP :{},BMI : {} , Insulin : {} , Glucose : {} , skin : {} '
      .format(bp_zeros,bmi_zeros,insulin_zeros,glucose_zeros,skin_zeros))

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:16:14.051343Z","iopub.execute_input":"2022-04-17T02:16:14.052095Z","iopub.status.idle":"2022-04-17T02:16:14.064540Z","shell.execute_reply.started":"2022-04-17T02:16:14.052059Z","shell.execute_reply":"2022-04-17T02:16:14.063651Z"}}


#### pre-process Insulin & SkinThickness 
df_copy=df.copy(deep=True)
## replace 0 with NaN to fill out easily
df_copy[['Insulin','SkinThickness']]=df_copy[['Insulin','SkinThickness']].replace(0,np.NaN)
## replace null value with median
df_copy['Insulin'].fillna(df_copy['Insulin'].median(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(),inplace=True)
df_copy=df_copy[(df_copy['BloodPressure']!=0) & (df_copy['BMI']!=0) & (df_copy['Glucose']!=0)]



# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:16:42.056474Z","iopub.execute_input":"2022-04-17T02:16:42.056957Z","iopub.status.idle":"2022-04-17T02:16:42.066371Z","shell.execute_reply.started":"2022-04-17T02:16:42.056927Z","shell.execute_reply":"2022-04-17T02:16:42.065501Z"}}
## split dataset on outcome from 70% to 30%
X=df_copy.drop('Outcome',axis=1)
y=df_copy['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:18:15.863613Z","iopub.execute_input":"2022-04-17T02:18:15.864043Z","iopub.status.idle":"2022-04-17T02:18:16.193795Z","shell.execute_reply.started":"2022-04-17T02:18:15.864010Z","shell.execute_reply":"2022-04-17T02:18:16.193148Z"}}
naive=GaussianNB()
naive.fit(X_train,y_train)
predicted_naive=naive.predict(X_test)
cm_naive=metrics.confusion_matrix(y_test,predicted_naive)
accuracy_naive=metrics.accuracy_score(y_test,predicted_naive)


plot_confusion_matrix(naive, X_test, y_test)  
plt.show() 



# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:20:56.824757Z","iopub.execute_input":"2022-04-17T02:20:56.825042Z","iopub.status.idle":"2022-04-17T02:20:56.833160Z","shell.execute_reply.started":"2022-04-17T02:20:56.825014Z","shell.execute_reply":"2022-04-17T02:20:56.832184Z"}}
print("Confusion Matrix : \n {}  \n Accuracy : \n {} ".format(cm_naive,accuracy_naive))
# Accuracy : 0.7752293577981652
average_precision = average_precision_score(y_test, predicted_naive)
print(average_precision)

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:21:32.576493Z","iopub.execute_input":"2022-04-17T02:21:32.577137Z","iopub.status.idle":"2022-04-17T02:21:32.802439Z","shell.execute_reply.started":"2022-04-17T02:21:32.577103Z","shell.execute_reply":"2022-04-17T02:21:32.801566Z"}}


#  KNN Implementation 
knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
predicted_knn=knn.predict(X_test)
cm_knn=metrics.confusion_matrix(y_test,predicted_knn)
accuracy_knn=metrics.accuracy_score(y_test,predicted_knn)

plot_confusion_matrix(knn, X_test, y_test)  
plt.show() 

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:21:47.940546Z","iopub.execute_input":"2022-04-17T02:21:47.940871Z","iopub.status.idle":"2022-04-17T02:21:47.948512Z","shell.execute_reply.started":"2022-04-17T02:21:47.940841Z","shell.execute_reply":"2022-04-17T02:21:47.947674Z"}}
print("Confusion Matrix : \n {}  \n Accuracy :  {} ".format(cm_knn,accuracy_knn))
# k=3 (accuracy  0.7385321100917431) k=5 ( Accuracy :0.7522935779816514 ) 
# k=11 (Accuracy :  0.7706422018348624) k=13 ( Accuracy :  0.7614678899082569 )
average_precision = average_precision_score(y_test, predicted_knn)
print(average_precision)

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:22:27.409798Z","iopub.execute_input":"2022-04-17T02:22:27.410812Z","iopub.status.idle":"2022-04-17T02:22:27.427722Z","shell.execute_reply.started":"2022-04-17T02:22:27.410760Z","shell.execute_reply":"2022-04-17T02:22:27.426867Z"}}
# decision tree
d_tree=tree.DecisionTreeClassifier()
d_tree.fit(X_train,y_train)
predicted_tree=d_tree.predict(X_test)
accuracy_tree=metrics.accuracy_score(y_test,predicted_tree)
cm_dt=metrics.confusion_matrix(y_test,predicted_tree)


print("Confusion Matrix : \n {}  \n Accuracy :  {} ".format(cm_dt,accuracy_tree))

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:22:52.031577Z","iopub.execute_input":"2022-04-17T02:22:52.032329Z","iopub.status.idle":"2022-04-17T02:22:52.038225Z","shell.execute_reply.started":"2022-04-17T02:22:52.032290Z","shell.execute_reply":"2022-04-17T02:22:52.037440Z"}}


average_precision = average_precision_score(y_test, predicted_tree)
print(average_precision)



# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:23:34.565201Z","iopub.execute_input":"2022-04-17T02:23:34.565950Z","iopub.status.idle":"2022-04-17T02:23:34.611039Z","shell.execute_reply.started":"2022-04-17T02:23:34.565909Z","shell.execute_reply":"2022-04-17T02:23:34.610350Z"}}
# Logistic regression
logisticRegr= LogisticRegression()
logisticRegr.fit(X_train,y_train)
predict_lg = logisticRegr.predict(X_test)
accuarcy_lg=metrics.accuracy_score(y_test,predict_lg)
cm_lg=metrics.confusion_matrix(y_test,predict_lg)

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:24:10.425300Z","iopub.execute_input":"2022-04-17T02:24:10.426099Z","iopub.status.idle":"2022-04-17T02:24:10.434227Z","shell.execute_reply.started":"2022-04-17T02:24:10.426042Z","shell.execute_reply":"2022-04-17T02:24:10.433197Z"}}
print("Confusion Matrix : \n {}  \n Accuracy :  {} ".format(cm_lg,accuarcy_lg))
#0.7752293577981652
average_precision = average_precision_score(y_test, predict_lg)
print(average_precision)

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:24:42.040171Z","iopub.execute_input":"2022-04-17T02:24:42.040439Z","iopub.status.idle":"2022-04-17T02:24:42.065552Z","shell.execute_reply.started":"2022-04-17T02:24:42.040400Z","shell.execute_reply":"2022-04-17T02:24:42.064738Z"}}


# LDA
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
predict_lda=lda.predict(X_test)
accuracy_lda=metrics.accuracy_score(y_test,predict_lda)
cm_lda=metrics.confusion_matrix(y_test,predict_lda)

print("Confusion Matrix : \n {}  \n Accuracy :  {} ".format(cm_lda,accuracy_lda))

# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:24:55.477429Z","iopub.execute_input":"2022-04-17T02:24:55.478178Z","iopub.status.idle":"2022-04-17T02:24:55.483505Z","shell.execute_reply.started":"2022-04-17T02:24:55.478142Z","shell.execute_reply":"2022-04-17T02:24:55.482923Z"}}


average_precision = average_precision_score(y_test, predict_lda)
print(average_precision)




# %% [code] {"execution":{"iopub.status.busy":"2022-04-17T02:26:21.153913Z","iopub.execute_input":"2022-04-17T02:26:21.154714Z","iopub.status.idle":"2022-04-17T02:26:21.553809Z","shell.execute_reply.started":"2022-04-17T02:26:21.154673Z","shell.execute_reply":"2022-04-17T02:26:21.552791Z"}}
# Neural networks
nn=MLPClassifier()
nn.fit(X_train,y_train)
predict_nn=nn.predict(X_test)
accuracy_nn=metrics.accuracy_score(y_test,predict_nn)
cm_nn=metrics.confusion_matrix(y_test,predict_nn)

print("Neural networks : \n {}  \n Accuracy :  {} ".format(cm_nn,accuracy_nn))

# %% [code]
