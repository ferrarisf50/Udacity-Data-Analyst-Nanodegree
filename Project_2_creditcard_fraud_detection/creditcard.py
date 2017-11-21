# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:09:59 2017

@author: tangp05
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:25:10 2017

@author: ferra
"""

#%%
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score

userid=os.getlogin()
os.chdir('c:\\Users\\'+userid+'\\desktop')
#%%
#remove Time v8 v13 v15 v20 v22 v23 v24 v25 v26 v27 v28 Amount?

creditcard=pd.read_csv('creditcard.csv')
#1. average cv split ap=77+/-0.10    timeseries cv split ap=0.75+/-0.05
#df1 =creditcard.drop_duplicates()
df2= creditcard.drop(['Time','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8','Amount'], axis =1)

#2. average ap=72.6
#df2= creditcard.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)

#3 average ap=0.70+/-0.11
#df2= creditcard
#%%   create an hour variable
import datetime


df2['Hour']=df2['Time'].apply(lambda x : datetime.datetime.fromtimestamp(x).hour)
Hour= pd.get_dummies(df2['Hour'],prefix='Hour')
df2 = pd.concat([df2,Hour],axis=1)
df2 = df2.drop(["Hour","Time"], axis=1)

#%%

# neural network model 0.74+/-0.06


#%%
#df1 =creditcard.drop_duplicates()

X=df2.iloc[:,df2.columns != 'Class'].as_matrix()
y=df2['Class'].as_matrix()



#%% scaled

#split1=int(len(df2.index)*0.8)
#
#X_train , y_train =X[:split1,], y[:split1,]
#X_test, y_test=X[split1:,],y[split1:,]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
np.sum(y_test)
#%%
from sklearn.model_selection import GridSearchCV

layers=(100,100,100,100)
learning_rate_init=0.001
alpha=0.0001

#clf = MLPClassifier(solver='adam',activation='tanh',hidden_layer_sizes=layers, learning_rate='adaptive',learning_rate_init=lr_init,alpha=alpha,random_state=2)
clf = MLPClassifier(solver='adam',activation='tanh',
                    hidden_layer_sizes=layers, learning_rate='adaptive',
                    learning_rate_init=learning_rate_init,alpha=alpha,random_state=2,
                    max_iter=500,verbose=10)

#gs = GridSearchCV(clf,
#                  param_grid={'alpha': [0.0003,0.0001,0.00003]},
#                  scoring='average_precision', cv=cv)
#gs.fit(X_train, y_train)
#results = gs.cv_results_

#print('Grid best parameter is :',gs.best_params_)
#print('Grid best score is :',gs.best_score_)

#%% Time series cross validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import StratifiedKFold

scaler = StandardScaler() 
fold=5
tscv = TimeSeriesSplit(n_splits=fold)

cv=StratifiedKFold(n_splits=fold,random_state=1234)

pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])

#scores=cross_val_score(pipeline,X_train,y_train,cv=cv.split(X_train,y_train),
#                       scoring='average_precision',verbose=10)
#print(scores)
#print(np.mean(scores))

#%%
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=200,max_depth=2, random_state=0)



#%%
#%%
from sklearn.metrics import f1_score, fbeta_score,precision_recall_curve,average_precision_score


scaler = StandardScaler() 
scaler.fit(X_train) 
X_train_scaled = scaler.transform(X_train)  
X_test_scaled = scaler.transform(X_test)

y_score=clf.predict_proba(X_test_scaled)[:,1]


probas_ = clf.fit(X_train_scaled, y_train).predict_proba(X_test_scaled)[:, 1]
#precision, recall, thresholds = precision_recall_curve(y_test, probas_)
average_precision = average_precision_score(y_test, probas_)




predict=(probas_>0.06).astype(int)
            
f1score=f1_score(y_test,predict)   
f2score=fbeta_score(y_test,predict,beta=2)

print("Ap= ",average_precision)        
print("f1 score: ",f1score)
print("f2 score: ",f2score)
print(metrics.classification_report(y_test,predict))
print(metrics.confusion_matrix(y_test,predict))


  

#%%
metrics.f1_score(y_test, y_score, average=None)
#%%
y_expect=clf.predict(X_test)
print(metrics.classification_report(y_test,y_expect))
print(metrics.confusion_matrix(y_test,y_expect))
cm1=metrics.confusion_matrix(y_test,y_expect)
#%%
from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)





#scores=cross_val_score(clf,X_train_scaled,y_train,cv=5,scoring=ftwo_scorer)
#print(scores)
#print(" %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#%% multiple metrics
from sklearn.model_selection import cross_validate
scorings ='average_precision'

scores=cross_validate(clf,X_train_scaled,y_train,cv=5,scoring=scorings,return_train_score=True)
print(scores)




#%% Time series cross validate with roc plot
#from scipy import interp
#import matplotlib.pyplot as plt
#from sklearn.metrics import f1_score
#from sklearn.metrics import fbeta_score
#
#fold=5
#tscv = TimeSeriesSplit(n_splits=fold)
#
#tprs = []
#aucs = []
#mean_fpr = np.linspace(0, 1, 100)
#
#i = 0
#for train, test in cv.split(X, y):
#    scaler = StandardScaler() 
#    scaler.fit(X[train]) 
#    X[train] = scaler.transform(X[train])  
#    X[test] = scaler.transform(X[test])  
#    
#    probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
#    # Compute ROC curve and area the curve
#    fpr, tpr, thresholds = roc_curve(y[test], probas_)
#    tprs.append(interp(mean_fpr, fpr, tpr))
#    tprs[-1][0] = 0.0
#    roc_auc = auc(fpr, tpr)
#    
#    f1score_max=0
#    f2score_max=0
#    
#    for k in range(thresholds.shape[0]):
#        proba_temp=(probas_>thresholds[k]).astype(int)
#                
#        f1score=f1_score(y[test],proba_temp)   
#        f2score=fbeta_score(y[test],proba_temp,beta=2)
#         
#        if f1score>f1score_max:
#            f1score_max=f1score
#            best_threshold_f1=thresholds[k]
#        if f2score>f2score_max:
#            f2score_max=f2score
#            best_threshold_f2=thresholds[k]
#            
#    proba_temp=[0 if probas_[j]<=best_threshold_f1 else 1 for j in range(probas_.shape[0])]
#    
#    print("Fold=%d max F1 score" % i)
#    print(metrics.classification_report(y[test],proba_temp))
#    print(metrics.confusion_matrix(y[test],proba_temp))
#    
#    proba_temp=[0 if probas_[j]<=best_threshold_f2 else 1 for j in range(probas_.shape[0])]
#    
#    print("Fold=%d max F2 score" % i)
#    print(metrics.classification_report(y[test],proba_temp))
#    print(metrics.confusion_matrix(y[test],proba_temp))   
#    
#    aucs.append(roc_auc)
#    plt.plot(fpr, tpr, lw=1, alpha=0.3,
#             label=r'fold %d AUC=%0.2f F1=%0.2f F2=%0.2f' % (i, roc_auc,f1score_max,f2score_max))
#    
#    
#    
#    
#    i += 1
#    
#    
#plt.plot([0, 3], [0, 3], linestyle='--', lw=2, color='r',
#         label='Luck', alpha=.8)
#
#mean_tpr = np.mean(tprs, axis=0)
#mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
#std_auc = np.std(aucs)
#plt.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#         lw=2, alpha=.8)
#
#std_tpr = np.std(tprs, axis=0)
#tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                 label=r'$\pm$ 1 std. dev.')
#
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()


#%%
#  AUPRC metrics 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

def plot_precision_recall(X,y,cv):
    aps=[]
    f1scores=[]
    f2scores=[]
    best_thresholds_f1=[]
    best_thresholds_f2=[]
    i = 0
    for train, test in cv.split(X, y):
        scaler = StandardScaler() 
        scaler.fit(X[train]) 
        X[train] = scaler.transform(X[train])  
        X[test] = scaler.transform(X[test])  
        
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        # Compute precision recall curve and area the curve
        precision, recall, thresholds = precision_recall_curve(y[test], probas_)
        average_precision = average_precision_score(y[test], probas_)
       # auc = metrics.auc(recall, precision)
        
        
       
       
        plt.step(recall, precision,lw=1, alpha=0.3,
                 label=r'f%d AP=%0.2f' % (i,average_precision))
        aps.append(average_precision)
        
    

        
        
        f1score=2*precision*recall/(precision+recall)  
        f2score=5*precision*recall/(4*precision+recall) 

        f1scores.append(np.amax(f1score))
        f2scores.append(np.amax(f2score))
        best_thresholds_f1.append(thresholds[np.argmax(f1score)])
        best_thresholds_f2.append(thresholds[np.argmax(f2score)])    
         
                
         
#        print("Fold=%d max F1 score" % i)
#        print(metrics.classification_report(y[test],proba_temp))
#        print(metrics.confusion_matrix(y[test],proba_temp))
        
        
#        print("Fold=%d max F2 score" % i)
#        print(metrics.classification_report(y[test],proba_temp))
#        print(metrics.confusion_matrix(y[test],proba_temp))   
        
        i += 1
        
        
    plt.plot([0, 3], [0, 3], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    mean_aps = np.mean(aps, axis=0)
    std_aps= np.std(aps)
    #mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    
    #plt.plot(mean_fpr, mean_tpr, color='b',
    #         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #         lw=2, alpha=.8)
    
    #std_tpr = np.std(tprs, axis=0)
    #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall-Curve mean %0.2f $\pm$ %0.2f)' % (mean_aps, std_aps))
    plt.legend(loc="lower left")
    plt.show()
    return  f1scores,f2scores,best_thresholds_f1,best_thresholds_f2



f1scores,f2scores,best_thresholds_f1,best_thresholds_f2=plot_precision_recall(X_train,y_train,cv)


#%%
aaa=np.array([1,2,3,4])   
b=(aaa>1).astype(int) 
c=aaa+b
best_thresholds_f1
