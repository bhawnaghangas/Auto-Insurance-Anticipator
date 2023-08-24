#!/usr/bin/env python
# coding: utf-8

# Auto Insurance Anticipator Project by Bhawna Ghangas

# In[295]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import seaborn as sns


# In[296]:


dummy = pd.read_csv(r"C:\Users\ghang\Downloads\Insurance_train.csv", na_values =['NA'])
temp = dummy.columns.values
temp


# In[297]:


new_df= dummy


# In[298]:


new_df.apply(lambda x: x.isnull().sum())


# DATA PREPROCESSING

# In[299]:


def preprocessor(df):
    columns_to_encode = df.select_dtypes(include=['object']).columns
    le = preprocessing.LabelEncoder()
    df[columns_to_encode] = df[columns_to_encode].apply(le.fit_transform)
    return df


# In[300]:


en_df = preprocessor(new_df)


# In[301]:


get_ipython().run_line_magic('matplotlib', 'inline')

import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from urllib.request import urlopen 

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500) 


# In[302]:


print(en_df['is_claim'].unique())


# In[303]:


en_df.shape


# check numbers of 0s and 1s

# In[304]:


print("Total number of is_claim: {}".format(en_df.shape[0]))
print("Number of 0s: {}".format(en_df[en_df.is_claim == 0].shape[0]))
print("Number of 1s: {}".format(en_df[en_df.is_claim == 1].shape[0]))


# In[305]:


feature_space = en_df.iloc[:, en_df.columns != 'is_claim']
feature_class = en_df.iloc[:, en_df.columns == 'is_claim']


# Oversampling 

# In[306]:


get_ipython().system('pip install imbalanced-learn')
from imblearn.over_sampling import SMOTE


# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(feature_space, feature_class)


# Perform train test data split

# In[307]:


training_set, test_set, class_set, test_class_set = train_test_split(X_train_resampled,
                                                                    y_train_resampled,
                                                                    test_size = 0.20, 
                                                                    random_state = 42)


# In[308]:


class_set = class_set.values.ravel() 
test_class_set = test_class_set.values.ravel() 


# In[309]:


#class_weights = {0: 1, 1: 50}


# Build the model

# In[310]:


fit_rf = RandomForestClassifier(random_state=42)


# HyperParameter tunning

# In[311]:



np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [2,3,4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']
              }

cv_rf = GridSearchCV(fit_rf, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 2)

cv_rf.fit(training_set, class_set)
print('Best Parameters using grid search: \n', cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))


# In[312]:


fit_rf.set_params(criterion = 'gini',
                  max_features = 'auto', 
                  max_depth = 4)


# In[313]:


import warnings
warnings.filterwarnings("ignore")
fit_rf.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 15
max_estimators = 2000

error_rate = {}


# In[314]:


fit_rf.set_params(n_estimators=800,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)


# In[315]:


fit_rf.fit(training_set, class_set)


# In[316]:


def variable_importance(fit):
    """
    Purpose
    ----------
    Checks if model is fitted CART model then produces variable importance
    and respective indices in dictionary.

    Parameters
    ----------
    * fit:  Fitted model containing the attribute feature_importances_

    Returns
    ----------
    Dictionary containing arrays with importance score and index of columns
    ordered in descending order of importance.
    """
    try:
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit)) 

        # Captures whether the model has been trained
        if not vars(fit)["estimators_"]:
            return print("Model does not appear to be trained.")
    except KeyError:
        print("Model entered does not contain 'estimators_' attribute.")

    importances = fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {'importance': importances,
            'index': indices}


# In[317]:


var_imp_rf = variable_importance(fit_rf)

importances_rf = var_imp_rf['importance']

indices_rf = var_imp_rf['index']


# In[318]:


names = en_df.columns

print(names)


# In[319]:


dx = [0,1]        


# In[320]:


names_index = names


# In[321]:


def print_var_importance(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on information gain for CART model.
    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                models organized by dataframe index
    * indices: Organized index of dataframe from largest to smallest
                based on feature_importances_
    * name_index: Name of columns included in model

    Returns
    ----------
    Prints feature importance in descending order
    """
    print("Feature ranking:")

    for f in range(0, indices.shape[0]):
        i = f
        print("{0}. The feature '{1}' has a Mean Decrease in Impurity of {2:.5f}"
              .format(f + 1,
                      names_index[indices[i]],
                      importance[indices[f]]))


# In[322]:


print_var_importance(importances_rf, indices_rf, names_index)


# In[323]:


def variable_importance_plot(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints bar chart detailing variable importance for CART model
    NOTE: feature_space list was created because the bar chart
    was transposed and index would be in incorrect order.

    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                models organized by dataframe index
    * indices: Organized index of dataframe from largest to smallest
                based on feature_importances_
    * name_index: Name of columns included in model

    Returns:
    ----------
    Returns variable importance plot in descending order
    """
    importance_desc = sorted(importance)
    feature_space = [name_index[i] for i in indices[::-1]]
    index = np.arange(len(feature_space))

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_facecolor('#fafafa')
    plt.title('Feature importances for Random Forest Model\nCar Insurance')
    plt.barh(index,
             importance_desc,
             align="center",
             color='#875FDB')
    plt.yticks(index,
               feature_space)

    plt.ylim(-1, len(feature_space))
    plt.xlim(0, max(importance_desc) + 0.01)
    plt.xlabel('Mean Decrease in Impurity')
    plt.ylabel('Feature')

    plt.show()
    plt.close()

# Example usage
variable_importance_plot(importances_rf, indices_rf, names_index)


# In[324]:


predictions_rf = fit_rf.predict(test_set)


# In[325]:


test_crosst=0
def create_conf_mat(test_class_set, predictions):
    """Function returns confusion matrix comparing two arrays"""
    if (len(test_class_set.shape) != len(predictions.shape) == 1):
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif (test_class_set.shape != predictions.shape):
        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index = test_class_set,
                                        columns = predictions)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosst


# In[326]:


accuracy_rf = fit_rf.score(test_set, test_class_set)

print("Here is our mean accuracy on the test set:\n {0:.3f}"      .format(accuracy_rf))


# In[327]:


# Here we calculate the test error rate!
test_error_rate_rf = 1 - accuracy_rf
print("The test error rate for our model is:\n {0: .4f}"      .format(test_error_rate_rf))


# In[328]:


predictions_prob = fit_rf.predict_proba(test_set)[:, 1]

fpr2, tpr2, _ = roc_curve(test_class_set,
                          predictions_prob,
                          pos_label = 1)


# In[329]:


auc_rf = auc(fpr2, tpr2)


# In[330]:


def plot_roc_curve(fpr, tpr, auc, estimator, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph

    Parameters
    ----------
    * fpr: Array returned from sklearn.metrics.roc_curve for increasing
            false positive rates
    * tpr: Array returned from sklearn.metrics.roc_curve for increasing
            true positive rates
    * auc: Float returned from sklearn.metrics.auc (Area under Curve)
    * estimator: String represenation of appropriate model, can only contain the
    following: ['knn', 'rf', 'nn']
    * xlim: Set upper and lower x-limits
    * ylim: Set upper and lower y-limits
    """
    my_estimators = {'knn': ['Kth Nearest Neighbor', 'deeppink'],
              'rf': ['Random Forest', 'red'],
              'nn': ['Neural Network', 'purple']}

    try:
        plot_title = my_estimators[estimator][0]
        color_value = my_estimators[estimator][1]
    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()


# In[331]:


plot_roc_curve(fpr2, tpr2, auc_rf, 'rf',
               xlim=(-0.01, 1.05), 
               ylim=(0.001, 1.05))


# In[332]:


dx=['Yes','No']


# In[333]:


def print_class_report(predictions, alg_name):
    """
    Purpose
    ----------
    Function helps automate the report generated by the
    sklearn package. Useful for multiple model comparison

    Parameters:
    ----------
    predictions: The predictions made by the algorithm used
    alg_name: String containing the name of the algorithm used
    
    Returns:
    ----------
    Returns classification report generated from sklearn. 
    """
    print('Classification Report for {0}:'.format(alg_name))
    print(classification_report(predictions, 
            test_class_set, 
            target_names = dx))


# In[334]:


class_report = print_class_report(predictions_rf, 'Random Forest')


# In[335]:


new_data = pd.read_csv(r"C:\Users\ghang\Downloads\archive\test.csv", na_values =['NA'])
new_df2= new_data
new_df2.apply(lambda x: x.isnull().sum())


# In[336]:


#sample_size2 = 10  # Specify the desired sample size
#test_sample = new_df2.sample(n=sample_size2, random_state=42)


# In[337]:


def preprocessor(df):
    columns_to_encode = df.select_dtypes(include=['object']).columns
    le = preprocessing.LabelEncoder()
    df[columns_to_encode] = df[columns_to_encode].apply(le.fit_transform)
    return df


# In[338]:


en_df_test = preprocessor(new_df2)


# In[342]:


new_predictions = fit_rf.predict(en_df_test)


# In[343]:


sdata = pd.read_csv(r"C:\Users\ghang\Downloads\archive\sample_submission.csv")
true_labels = sdata['is_claim']


# In[344]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_labels, new_predictions)
accuracy


# Auto Insurance Anticipator Project by Bhawna Ghangas
