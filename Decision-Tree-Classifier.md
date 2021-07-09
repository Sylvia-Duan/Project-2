# Import Modules
import pandas as pd <br />
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Import Data
heart_attack_data = pd.read_csv('heart.csv')

X = heart_attack_data.drop('output', axis = 1).copy()
y = heart_attack_data['output'].copy()

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns = ['cp',
                                        'restecg',
                                         'slp',
                                        'thall'])

# Building Primary Classification Tree
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state = 42)

clf_dt = DecisionTreeClassifier(random_state = 42)
clf_dt = clf_dt.fit(X_train, y_train)

plt.figure(figsize = (15,7.5))
plot_tree(clf_dt,
         filled = True,
         rounded = True,
         class_names = ["NO", "YES"],
         feature_names = X_encoded.columns)

## Constructing Confusion Matrix
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels = ["Does not have HD", "Does have HD"])

# Pruning: Visualize parameter alpha
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)
    
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs. alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker = 'o', label = "train", drawstyle = "steps-post")
ax.plot(ccp_alphas, test_scores, marker = '^', label = "test", drawstyle = "steps-post")
ax.legend()
plt.show()

# Pruning: Finding the Best alpha Through Cross Validation
clf_dt = DecisionTreeClassifier(random_state = 42, ccp_alpha = 0.016)

scores = cross_val_score(clf_dt, X_train, y_train, cv = 4)
df = pd.DataFrame(data = {'tree': range(5), 'accuracy': scores})

df.plot(x = 'tree', y='accuracy', marker = 'D', linestyle = '--')

alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv = 4)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
    
alpha_results = pd.DataFrame(alpha_loop_values,
                            columns = ['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x = 'alpha',
                  y = 'mean_accuracy',
                  yerr = 'std',
                  marker = 'o',
                  linestyle = '--')
                  
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.0120)
                                &
                                (alpha_results['alpha'] < 0.0127)]['alpha']

ideal_ccp_alpha = float(ideal_ccp_alpha)

# Building the Final Classification Tree
clf_dt_pruned = DecisionTreeClassifier(random_state = 42,
                                      ccp_alpha = ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                     X_test,
                     y_test,
                     display_labels = ["Does not have HD", "Does have HD"])
                     
plt.figure(figsize = (15, 7.5))
plot_tree(clf_dt_pruned,
         filled = True,
         rounded = True,
         class_names = ["NO", "YES"],
         feature_names = X_encoded.columns)
