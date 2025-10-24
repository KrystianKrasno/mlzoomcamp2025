# %%
C = 1.0 
n_splits = 5
output_file = f'model_C={C}.bin'


# %%
import pickle 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# %%
# Load model from Weeks 3/4 
df = pd.read_csv("course_lead_scoring.csv")

cat_features = list(df.dtypes[df.dtypes == 'object'].index)
print(cat_features)

num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(num_features)


for col in cat_features:
    df[col] = df[col].fillna('NA')

for col in num_features:
    df[col] = df[col].fillna(0)


# df.isnull().sum()


# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# %% - turning off removing converted, as we split later? 
# # Check and remove 'converted' from numerical features list
# if 'converted' in num_features:
#     num_features.remove('converted')
    
# # Check and remove 'converted' from categorical features list if it was mistakenly there
# if 'converted' in cat_features:
#     cat_features.remove('converted')

# # Initiate the model
all_features = cat_features +  num_features

# %%
# Define functions that train / run the model 

def train(df_train, y_train, C=1.0):
    """Trains the Logistic Regression model."""
    dicts = df_train[cat_features + num_features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # Using solver='liblinear' is important for C < 1.0 (though here C=1.0)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000) 
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    """Generates probability predictions on the given DataFrame."""
    dicts = df[cat_features + num_features].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# %%
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0 

for train_idx, val_idx in kfold.split(df_full_train): 
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.converted.values
    y_val = df_val.converted.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print('AUC on fold {fold} is {auc}')
    fold = fold + 1


print("\nvalidation results: ")
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# %%
print("\ntraining the final model")
dv, model = train(df_full_train, df_full_train.converted.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.converted.values
auc = roc_auc_score(y_test, y_pred)

print(f'AUC: {auc}')

# %%
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)  # do stuff

print(f'\nthe output is saved to this {output_file}')
    