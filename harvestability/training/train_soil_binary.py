import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
np.random.seed(42)
import xgboost as xgb
from xgboost import plot_importance
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union,FeatureUnion
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA 
from datetime import datetime

# Column names of harvestability data
cols_own= ["sand_0-5cm_mean","sand_5-15cm_mean","silt_0-5cm_mean","silt_5-15cm_mean","clay_0-5cm_mean",
           "clay_5-15cm_mean","soc_0-5cm_mean","soc_5-15cm_mean","DTM_height","DTM_slope",
           "DTM_aspect","TCD","WAW","harvestability","TWI"]

#data_dir = "../training/"
data_dir = "/home/ubuntu/data/ml-harvestability/training/soildata/"
model_dir = "/home/ubuntu/ml-harvesterseasons/harvestability/models/"
plot_dir = "/home/ubuntu/ml-harvesterseasons/harvestability/plots/"
# load tarin data with locations
train_data = pd.read_csv(f"{data_dir}train_data.csv",usecols=cols_own)

# load test data LUCAS
test_data = pd.read_csv(f"{data_dir}LUCAS_train.csv",usecols=cols_own)

# Cleaning train data, eremoving nan, null, 0, 254 (no data)
train_clean_data = train_data.loc[~train_data["harvestability"].isin(["",254,0,np.nan])]

# Selecting features to train
train_clean_data = train_clean_data[cols_own]

#converting  harvestability column to binary
train_clean_data.loc[train_clean_data["harvestability"] <4 , "harvestability"] = 0
train_clean_data.loc[train_clean_data["harvestability"] >3 , "harvestability"] = 1

# Cleaning test data, eremoving nan, null, 0, 254 (no data)
test_clean_data = test_data.loc[~test_data["harvestability"].isin(["",254,0,np.nan])]

# selecting features to test
test_clean_data = test_clean_data[cols_own]

#converting  harvestability column to binary
test_clean_data.loc[test_clean_data["harvestability"] <4 , "harvestability"] = 0
test_clean_data.loc[test_clean_data["harvestability"] >3 , "harvestability"] = 1

train_clean_data = train_clean_data.groupby("harvestability").sample(n=240000, random_state=42)

# Combing all traing and test data
all_data = pd.concat([train_clean_data,test_clean_data])

#Storing LUCAS data from all data separetly
lucas_data = all_data.iloc[len(train_clean_data):]

#Validation data as LUCAS data
X_valid = lucas_data.drop(["harvestability"],axis=1)

# Validation data where harvestability class number where readjusted to one bleow
y_valid = lucas_data.harvestability.copy()

# Similarly harvestability class copied from all data and readjusted to one bleow
harvestibility = all_data["harvestability"].copy()

# Harvestability classes dropped from features
predictors = all_data.drop("harvestability",axis=1)

print(predictors.columns)

# Splitting data for Trainin and testing
X_train, X_test, y_train, y_test = train_test_split(predictors, harvestibility, test_size=0.2, random_state=42)

#
xgb_cl = xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=2,
                            learning_rate=0.16188216483685822,
                            early_stopping_rounds=10,
                            max_depth=20,
                            n_estimators=295,
                            subsample=0.7347796276476328,
                            eval_metric="merror")

xgb_cl.fit(X_train, y_train,verbose=1, 
           # set to 1 to see xgb training round intermediate results
            eval_set=[(X_valid,y_valid),(X_test,y_test)])


y_pred = xgb_cl.predict(X_test)


print('\n------------------ Confusion Matrix -----------------\n')
print(confusion_matrix(y_test, y_pred))

print('\n-------------------- Key Metrics --------------------')
print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(y_test, y_pred))


y_pred = xgb_cl.predict(X_valid)

print('\n------------------ Confusion Matrix -----------------\n')
print(confusion_matrix(y_valid, y_pred))

print('\n-------------------- Key Metrics --------------------')
print('\nAccuracy: {:.2f}'.format(accuracy_score(y_valid, y_pred)))
print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(y_valid, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_valid, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_valid, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_valid, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_valid, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_valid, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_valid, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_valid, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_valid, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_valid, y_pred, average='weighted')))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(y_valid, y_pred))
plot_importance(xgb_cl,grid=False)
plt.savefig(plot_dir+'soil_binary_importance.jpg',bbox_inches='tight')

cf_matrix = confusion_matrix(y_valid, y_pred)
fig, ax = plt.subplots(figsize=(8,7))     
group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
labels = [f"{v1}" for v1 in group_counts]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Greens')
plt.title('Harvestability Confusion Matrix')
ax.set_ylabel('True Class')
ax.set_xlabel('Predicted Class')
plt.savefig(plot_dir+'Harvestability_Binary_Confusion_Matrix',bbox_inches='tight')
plt.close()


now = datetime.now()
date_time = now.strftime("%m%d%Y%H%M%S")


print("---Saving model---")
xgb_cl.save_model(f"{model_dir}xgbmodel_harvestability_binary_{date_time}.json")
