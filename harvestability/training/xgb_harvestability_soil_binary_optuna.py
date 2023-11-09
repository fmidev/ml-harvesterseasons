import os,optuna,time,random,warnings
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import plot_importance
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
### XGBoost with Optuna hyperparameter tuning
# note: does not save trained mdl
startTime=time.time()

data_dir = "/home/ubuntu/data/ml-harvestability/training/soildata/"
ml_dir='/home/ubuntu/data/ml-harvestability/ml-harvestability-data'
plot_dir = "/home/ubuntu/ml-harvesterseasons/harvestability/plots/"

os.chdir(ml_dir)
print(os.getcwd())
### optuna objective & xgboost
def objective(trial):
    # hyperparameters
    param = {
    
        "max_depth":trial.suggest_int("max_depth",18,30),
        "objective": "multi:softmax",
        "subsample":trial.suggest_float("subsample",0.01,1),
        "learning_rate":trial.suggest_float("learning_rate",0.1,0.3),
        "n_estimators":trial.suggest_int("n_estimators",185,300),
        "random_state":42,
        "num_class":2,
        "early_stopping_rounds":10,
        "eval_metric":"merror"
    }
    eval_set=[(X_valid,y_valid),(X_test,y_test)]

    xgb_cl=xgb.XGBClassifier(**param)
    bst = xgb_cl.fit(X_train,y_train,eval_set=eval_set)
    y_pred = bst.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
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
    print("\n-----------Test LUCAS---------------------\n")
    pred_all = bst.predict(X_valid)
    print('\nAccuracy: {}'.format(accuracy_score(y_valid, pred_all)))
    print('\n------------------ Confusion Matrix -----------------\n')
    print(confusion_matrix(y_valid, pred_all))
    print('\n--------------- Classification Report ---------------\n')
    print(classification_report(y_valid, pred_all))
    plot_importance(xgb_cl,grid=False)
    plt.savefig(plot_dir+'plot_importance.jpg')
    return accuracy

train_fname="train_data.csv"
test_fname="LUCAS_train.csv"
cols_own= ["sand_0-5cm_mean","sand_5-15cm_mean","silt_0-5cm_mean","silt_5-15cm_mean","clay_0-5cm_mean",
           "clay_5-15cm_mean","soc_0-5cm_mean","soc_5-15cm_mean","DTM_height","DTM_slope",
           "DTM_aspect","TCD","WAW","harvestability","TWI"]
train_data = pd.read_csv(os.path.join(data_dir,train_fname),usecols=cols_own)
test_data = pd.read_csv(os.path.join(data_dir,test_fname),usecols=cols_own)

#removing blank harvestibility classes
train_clean_data = train_data.loc[~train_data["harvestability"].isin(["",254,0,np.nan])]
test_clean_data = test_data.loc[~test_data["harvestability"].isin(["",254,0,np.nan])]

#order columns
train_clean_data = train_clean_data[cols_own]
test_clean_data = test_clean_data[cols_own]

train_clean_data.loc[train_clean_data["harvestability"] <4 , "harvestability"] = 0
train_clean_data.loc[train_clean_data["harvestability"] >3 , "harvestability"] = 1

test_clean_data.loc[test_clean_data["harvestability"] <4 , "harvestability"] = 0
test_clean_data.loc[test_clean_data["harvestability"] >3 , "harvestability"] = 1

train_clean_data = train_clean_data.groupby("harvestability").sample(n=240000, random_state=42)

all_data = pd.concat([train_clean_data,test_clean_data])

# keep lucas as validation set validation 
validation_set = all_data.iloc[len(train_clean_data):]
y_valid = validation_set.harvestability.copy()
X_valid = validation_set.drop(["harvestability"],axis=1)

#XGBoost accepts target values only starting from '0', so deducting 1 from each class
harvestibility = all_data.harvestability.copy()

#drop target value from prdictors
predictors=all_data.drop(["harvestability"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(predictors, harvestibility, test_size=0.2, random_state=42)

### Optuna trials
study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name="harvestability-soil-binary-trail-1",direction="maximize")
study.optimize(objective, n_trials=100, timeout=432000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
