import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# PARAMETERS
C = 1.0
n_splits = 3
output_file = f'model_C={C}.bin'

# DATA PREPARATION

df = pd.read_csv('churnmodel.csv')

if 'Unnamed: 0' in df.columns:
  df = df.drop(columns = ['Unnamed: 0'])

df.drop(['CD_ACCOUNT'], axis = 1, inplace = True)

df['STATUS2'] = df['STATUS2'].map({'CHURNED': 1, 'NOT CHURNED': 0})
df['STATUS2'].value_counts()

df['CD_TYPE'] = df['CD_TYPE'].map({1062: 1, 1063: 0})
df['CD_TYPE'].value_counts()

df['VL_TENOR'] = df['VL_TENOR'].abs()

df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
  df[col] = df[col].str.lower().str.replace(' ', '_')

binary = ['cd_type']
numerical = ['days_acc_open', 'vl_credit_recency', 'vl_debit_recency', 'vl_tenor', 'avgcreditturnover_ly', 
             'avgdebitturnover_ly', 'avgcredit_trans_ly', 'avgdebit_trans_ly', 'avgcreditturnover_ly2', 
             'avgdebitturnover_ly2', 'avgcredit_trans_ly2', 'avgdebit_trans_ly2']

X = df.drop('status2', axis = 1)
y = df['status2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

gb_model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3)

# TRAINING

def train(X_train, y_train, C = 1.0):
   gb_model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3)
   gb_model.fit(X_train, y_train)
   
   return gb_model

def predict (df, gb_model):
   y_pred = gb_model.predict(X_test)
   return y_pred

# VALIDATION

print(f'doing validation with C={C}')

kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

scores = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
   X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
   y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
   
   gb_model.fit(X_train, y_train)
   y_pred = gb_model.predict(X_test)

   score = accuracy_score(y_test, y_pred)
   scores.append(score)
   
   print(f'Fold {fold+1} accuracy: {score:.3f}')

print(f'Mean accuracy: {np.mean(scores):.3f}')
print(f'Standard Deviation: {np.std(scores):.3f}')

with open(output_file, 'wb') as f_out:
    pickle.dump((gb_model), f_out)

print(f'The model is saved in {output_file}')