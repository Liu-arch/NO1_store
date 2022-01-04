import pandas as pd

full=pd.read_csv('data_train_14full(1).csv')
predictors=['brand','serial','model','country','cityid','maketype']
full_dummies=pd.get_dummies(full[predictors])
full_all = full_dummies[full_dummies.maketype.notnull()]
full_train = full_all[['brand','serial','model','country','cityid']]

from sklearn.tree import DecisionTreeClassifier


X=full_train
y=full_all.maketype

full_empty = full[~full.maketype.notnull()]
X_empty = full_empty[['brand','serial','model','country','cityid']]
tree_model=DecisionTreeClassifier()
tree_model.fit(X,y)
maketype_full = tree_model.predict(X_empty)
full_empty['maketype'] = maketype_full
full_not = full[full.maketype.notnull()]
full_full = full_not.append(full_empty).to_csv('data_full.csv')