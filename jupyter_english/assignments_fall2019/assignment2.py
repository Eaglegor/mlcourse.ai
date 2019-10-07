#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#%%
X = np.linspace(-2, 2, 7)
y = X ** 3

#%%
plt.scatter(X, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$');
plt.plot([-2, 2], [np.average(y), np.average(y)])

#%%
plt.scatter(X, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$');
plt.plot([-2, 0, 0, 2], [np.average(y[:4]), np.average(y[:4]), np.average(y[3:]), np.average(y[3:])])

#%%
def regression_var_criterion(X, y, t):
    Xl = np.extract(X < t, X)
    Xr = np.extract(X >= t, X)
    return np.var(y) - (Xl.size / X.size)*np.var(Xl) - (Xr.size / X.size)*np.var(Xr)

#plt.scatter(X, y)
#plt.xlabel(r'$x$')
#plt.ylabel(r'$y$');
s = np.linspace(-1.9, 1.9, 20)
plt.plot(s, [regression_var_criterion(X, y, t) for t in s])


#%%

plt.scatter(X, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$');
a1 = y[0]
a2 = np.average(y[1:4])
a3 = np.average(y[3:6])
a4 = y[6]
plt.plot([-2, -1.5, -1.5, 0, 0, 1.5, 1.5, 2], [a1, a1, a2, a2, a3, a3, a4, a4])

#%%

df = pd.read_csv('data/mlbootcamp5_train.csv', index_col='id', sep=';')
df['age_in_years']= np.floor(df['age'] / 365.25)
df.head()

df = pd.concat([df, pd.get_dummies(df['cholesterol'], prefix='cholesterol')], axis=1)
df = df.drop(['cholesterol'], axis = 1)
df = pd.concat([df, pd.get_dummies(df['gluc'], prefix='gluc')], axis=1)
df = df.drop(['gluc'], axis = 1)
df.head()

#%%
import pydotplus

def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names,
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

df_train = df.drop(['cardio'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(df_train, df['cardio'], test_size = 0.3, random_state = 17)

tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train, y_train)
tree_graph_to_png(tree=tree, feature_names=df_train.columns, png_file_to_save='img/ass3_tree1.png')

#%%
tree_pred = tree.predict(X_valid)
acc1 = accuracy_score(y_valid, tree_pred)

#%%
tree_params = {'max_depth': list(range(2,11))}
tree_grid = GridSearchCV(tree, tree_params, cv=5)
tree_grid.fit(X_train, y_train)
tree_grid.best_params_
tree_grid.best_score_

plt.plot(list(range(2,11)), tree_grid.cv_results_['mean_test_score'])

#%%
acc2 = accuracy_score(y_valid, tree_grid.predict(X_valid))

#%%
eff = (acc2- acc1) / acc1
eff

#%%
df['age_bin'] = pd.cut(df['age_in_years'], right=False, bins=[0, 40, 50, 55, 60, 65], labels=[0, 40, 50, 55, 60])
df['ap_bin'] = pd.cut(df['ap_hi'], right=False, bins=[0, 120, 140, 160, 180, df['ap_hi'].max()], labels=[0, 120, 140, 160, 180])

df = pd.concat([df, pd.get_dummies(df['age_bin'], prefix='age')], axis=1)
df = pd.concat([df, pd.get_dummies(df['ap_bin'], prefix='ap')], axis=1)

df['male'] = df['gender'].map({1:0, 2:1})

df.head()



#%%
new_df = df[['smoke', 'cholesterol_1', 'cholesterol_2', 'cholesterol_3', 'male', 'age_40', 'age_50', 'age_55', 'age_60', 'ap_120', 'ap_140', 'ap_160', 'cardio']]
new_df_X = new_df.drop('cardio', axis=1)
new_df_y = new_df['cardio']

#%%
tree2 = DecisionTreeClassifier(max_depth=3, random_state=17)
tree2.fit(new_df_X, new_df_y)

tree_graph_to_png(tree=tree2, feature_names=new_df_X.columns, png_file_to_save='img/ass3_tree2.png')
