#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Base Package
import streamlit as st
import pandas as pd
import numpy as np
# Modeling Packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import xgboost
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# Viz Packages
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")

st.title("CAA- Machine Learning Application")





def make_predictions(test_data_path, clf, predictions_path):
    features = ['avg_delay_categorical',
                'variance_categorical',
                'LMH_cumulative',
                'avg_of_invoices_closed',
                'avg_of_all_delays',
                'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
                'payment_count_quarter_q4',
                'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
                'invoice_count_quarter_q4',
                'number_invoices_closed']

    data2=pd.read_csv(r''+test_data_path)#inp test data

    if len(data2)!=0:
        X_validation = data2[features]
        y_validation = data2['output']
    else:
        st.write('No Prediction data')
        return

    rfc=clf

    predictions = rfc.predict(X_validation)
    predictions_prob = rfc.predict_proba(X_validation)
    st.write('accuracy_score ' + str(accuracy_score(y_validation, predictions)))
    st.write('confusion matric ')
    st.write(confusion_matrix(y_validation, predictions))
    st.write('classification Report ')
    st.write(classification_report(y_validation, predictions))

    data2['predictions'] = predictions
    for i in range(0, data2.shape[0]):
        data2.at[i, 'pred_proba_0'] = predictions_prob[i][0]
        data2.at[i, 'pred_proba_1'] = predictions_prob[i][1]

    dataset=data2

    payment_without_any_subset=0
    dataset['transformed_output']=0
    for i in dataset['payment_id'].unique():
        max_proba=dataset[dataset['payment_id']==i]['pred_proba_1'].max()
        dataset.loc[(dataset['payment_id']==i) & (dataset['pred_proba_1']==max_proba),'transformed_output']=1
        if len(dataset[dataset['payment_id']==i]['output'].unique())==1:
            payment_without_any_subset=payment_without_any_subset+1

    st.write('***** After Output transformation *****')
    st.write('accuracy_score' + str(accuracy_score(y_validation, dataset['transformed_output'])))
    st.write('confusion matrix ')
    st.write(confusion_matrix(y_validation, dataset['transformed_output']))
    st.write('classification Report ')
    st.table(classification_report(y_validation, dataset['transformed_output']))

    st.write('Total Payment : ' + str(len(dataset['payment_id'].unique())))
    st.write('Total correct payment(s) : ' + str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)])))
    st.write('Total incorrect payment(s) : ' + str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==0)])))
    st.write('Total payments without any subset : ' + str(payment_without_any_subset))
    st.write('%age : '+ str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)])/len(dataset['payment_id'].unique())*100))

    data2.to_csv(predictions_path, index=False)
    st.write('Predictions Ended file saved successfully') 

    
    
    


# EVALUATION method for Classification on Metrics Like Accuracy, Confusion Matrix
# noinspection PyPep8Naming
def classification_eval(X_test, clf, y_train, y_test, y_pred):
    y = clf.predict_proba(X_test)
    result_df = pd.DataFrame(data=y, columns=list(np.unique(y_train)))
    st.header('Probability Matrix For Test Set')
    st.dataframe(result_df)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=list(np.unique(y_train)))
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis", xticklabels=list(np.unique(y_train)),
                yticklabels=list(np.unique(y_train)))
    plt.title('Confusion Matrix')
    ax.xaxis.set_label_position("top")
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    st.pyplot()
    st.write("Classification Report")
    st.table(metrics.classification_report(y_test, y_pred, output_dict=True, target_names=list(np.unique(y_train))))
    st.success(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


def train_model(df):
    features = ['avg_delay_categorical',
                'variance_categorical',
                'LMH_cumulative',
                'avg_of_invoices_closed',
                'avg_of_all_delays',
                'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
                'payment_count_quarter_q4',
                'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
                'invoice_count_quarter_q4',
                'number_invoices_closed']

    X_train = df[features]
    y_train = df['output']
    
    return X_train, y_train




# Classification
# noinspection PyPep8Naming
def lgbm(df):
 
    penalty = st.selectbox(label='Select Penalty Norm', options=['l1', 'l2', 'elasticnet', 'none'], index=1)
    solver = st.selectbox(label='Select Solver Method', options=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                          index=1)
    multi_class = st.selectbox(label='Select Class Type', options=['auto', 'ovr', 'multinomial'], index=0)
    X_train, X_test, y_train, y_test = df_split(df)
    if st.checkbox(label='See The Model Result'):
        logr = LogisticRegression(penalty=penalty, solver=solver, multi_class=multi_class, random_state=0)
        clf = logr.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        classification_eval(clf=clf, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred)
        

        

# noinspection PyPep8Naming,PyTypeChecker
def random_forest_classification(df):
    train_model(df)
    
    
    
    n_estimators = st.number_input(label='Enter Number of Estimator (Integer)', min_value=2, max_value=None)
    random_state=st.number_input(label='Enter Random state(Integer)', min_value=1, max_value=None)
    min_samples_split=st.number_input(label='Enter Min sample splits (Integer)', min_value=1, max_value=500)
    min_samples_leaf=st.number_input(label='Enter Min sample leaf (Integer)', min_value=1, max_value=200)
    max_depth = st.number_input(label='Enter Depth of Tree (Integer)', min_value=1, max_value=None)
    learning_rate=st.number_input(label='Enter learning rate', min_value=None, max_value=10)
    max_features = st.selectbox(label='Number of Features to consider at split', options=['auto', 'sqrt', 'log2'],
                                index=0)
    learning_rate=st.number_input(label='Enter learning rate', min_value=0, max_value=10)
    
    if st.checkbox(label='See The Model Result'):
        xgb= xgboost.XGBClassifier(random_state=random_state,n_estimators=n_estimators,
                               min_samples_split=min_samples_split, min_samples_leaf= min_samples_leaf, max_features= max_features,
                               max_depth=max_depth, learning_rate= learning_rate)
        clf = xgb.fit(X_train, y_train)
        
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        st.write('cross-validation scores: ' + str(scores))
        st.write('accuracy score of cross validation :' + str(scores.mean() * 100))
        
        
        
def xgb(df):
    train_model(df)

    n_estimators = st.number_input(label='Enter Number of Estimator (Integer)', min_value=2, max_value=None)
    random_state=st.number_input(label='Enter Random state(Integer)', min_value=1, max_value=None)
    min_samples_split=st.number_input(label='Enter Min sample splits (Integer)', min_value=1, max_value=500)
    min_samples_leaf=st.number_input(label='Enter Min sample leaf (Integer)', min_value=1, max_value=200)
    max_depth = st.number_input(label='Enter Depth of Tree (Integer)', min_value=1, max_value=None)
    learning_rate=st.number_input(label='Enter learning rate', min_value=0.0, max_value=5.0)
    max_features = st.selectbox(label='Number of Features to consider at split', options=['auto', 'sqrt', 'log2'],
                                index=0)
    
    if st.checkbox(label='See The Model Result'):
        xgb= xgboost.XGBClassifier(random_state=random_state,n_estimators=n_estimators,
                               min_samples_split=min_samples_split, min_samples_leaf= min_samples_leaf, max_features= max_features,
                               max_depth=max_depth, learning_rate= learning_rate)
        clf = xgb.fit(X_train, y_train)
        
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        st.write('cross-validation scores: ' + str(scores))
        st.write('accuracy score of cross validation :' + str(scores.mean() * 100))
        
    
        
        
     

    
def model(df):
    st.header('Choose Model Specifications')
    problemtype = st.sidebar.selectbox(label='Select Problem Type', options=['Classification'])
  
    methodlist = st.sidebar.selectbox(label='Select Algorithm', options=['Default XG-Boost',
                                                                         'Default Random-Forest',
                                                                         'Default Light-GBM',
                                                                         'XG-Boost',  
                                                                         'Random Forest Classification',
                                                                         'Light GBM'],
                                                                        key='classificationmethod')
    if methodlist == 'Light GBM':
        lgbm(df)
    elif methodlist == 'Random Forest Classification':
        random_forest_classification(df)
    elif methodlist == 'XG-Boost': 
        xgb(df)




def main():
    df=pd.read_csv(train_path)
    
    if side_bar == 'Modeling':
        model(df)
    


if __name__ == '__main__':
    side_bar = st.sidebar.selectbox(label='What do you want to do?', options=['Visualisation', 'Modeling'])
    test_path=r"C:/Users/Ayanava/Desktop/ai_cashapps/account_10112/train_test_splitted/train_70.csv"
    train_path=r"C:/Users/Ayanava/Desktop/ai_cashapps/account_10112/train_test_splitted/train_70.csv"
    main()

