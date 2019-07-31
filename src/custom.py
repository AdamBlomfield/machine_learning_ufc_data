# -*- coding: utf-8 -*-

# custom.py
##----------
## This is where you put your custom code that doesn't
## fit the standard categories of data cleaning, feature engineering,
## modeling, or visualization.
# For Dataframes and arrays
import numpy as np
import pandas as pd
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing Data
    # Train:Test split
from sklearn.model_selection import train_test_split
    # Scaling
from sklearn.preprocessing import StandardScaler
    # Feature Extraction
from sklearn.decomposition import PCA

# Modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

# Neural Network
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.models import Sequential
from keras.optimizers import SGD

# Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

# Warnings
# import warnings
# warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(123)
tf.set_random_seed(123)

models_summary = pd.DataFrame()
models_summary.rename_axis('Model', axis='columns', inplace=True)


def test_custom():
    print("In custom module")
    return None

def ufc_intro_vid():
    '''Embeds a youtube video with an introduction to UFC'''
    from IPython.display import HTML
    return HTML('<iframe width="900" height="600" src="https://www.youtube.com/embed/K4LRXBffbB0?start=0&end=66" frameborder="0" allowfullscreen></iframe>')

def save_figure(ax, title):
    '''Save the figure in the reports/figures folder in the format: title_string.png'''
    
    fig = ax.get_figure() 
    filename = '_'.join(title.split(' ')).lower()
    fig.savefig("../reports/figures/{}.png".format(filename))  
    
def plot_training_loss(loss, title='Training Loss'):
    '''Plot the loss on the training data, across each epoch'''
    # Set figure
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")
    
    # Plot
    epochs = range(1, len(loss) + 1)
    ax = sns.scatterplot(x=epochs, y=loss, color='darkred');
    
    # Title and Axis
    ax.set_title(title)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Loss')
    sns.despine()
    
    max_loss = max(loss)
    min_loss = min(loss)
    loss_reduction = max_loss - min_loss
    loss_reductions_pct = round((loss_reduction / max_loss) * 100,2)
    print('We have reduced the loss by {}% by training the model through {} epochs'.format(loss_reductions_pct, len(loss)))
    
    # Save the figure
    save_figure(ax, title)

def training_and_validation_accuracy(acc, val_acc, history_mod,
                                          title='Training and Validation Accuracy',
                                          y_lim=(0,1), mov_avg_n=25,
                                          show_mov_avg=False):
    '''Plot the accuracy on the training data and the validation data, across each epoch'''
    # Set figure
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")
    
    # Plot
    epochs = range(1, len(acc) + 1)
        # Dataframe for the regplot
    df = pd.DataFrame({'acc': history_mod.history['acc'], 'val_acc': history_mod.history['val_acc'], 'val_acc_avg': 0, 'epochs': range(1, len(epochs) + 1)})
        # Plot Training and Validation Accuracy
    ax = sns.regplot(x='epochs', y='acc', data=df, color='blue', label='Training', fit_reg=False, marker='.')
    ax2 = sns.regplot(x='epochs', y='val_acc', data=df, color='forestgreen', label='Validation', fit_reg=False, marker='.')
        # Plot Moving Average    
    if show_mov_avg:
        val_acc_avg = np.convolve(history_mod.history['val_acc'], np.ones((mov_avg_n,))/mov_avg_n, mode='valid') # Moving average of 3
        df['val_acc_avg'][mov_avg_n-1:] = list(val_acc_avg)
        ax3 = sns.regplot(x='epochs', y='val_acc_avg', data=df, color='orange', label='Validation Moving Avg ({} epochs)'.format(mov_avg_n), fit_reg=False, marker='.')

    # Title and Axis
    ax.set_title(title, fontsize='large', pad=20)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
    ax.legend()
    sns.despine()
    
    # Calculate the change in Accuracy
    max_acc = max(acc)
    min_acc = min(acc)
    acc_increase = max_acc - min_acc
    acc_increase_pct = round((acc_increase / min_acc) * 100,2)
    print('We have increased the accuracy by {}% by training the model through {} epochs'.format(acc_increase_pct, len(acc)))
    
    # Save the figure
    save_figure(ax, title)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''Plot the confusion matrix of predictions (can be normalized)'''
    
    # Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum()
        print("Normalized confusion matrix\n")
    else:
        print('Confusion matrix, without normalization\n')
#     print(cm)

    # Plot
    ax = plt.imshow(cm, interpolation='nearest', cmap=cmap)

    # Make Pretty
    plt.title('{}\n'.format(title))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    fmt = '.2f' if normalize else 'd'
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")
        
    # Save the figure
    save_figure(ax, title)
    
def acc_by_k_value(X_train, y_train, X_test, y_test, k_range_min=1, k_range_max=30, title='Accuracy of KNN, by Number of Neighbors'):
    '''Calculates the Accuracy score at each k-value and plots a visualization'''
    
    # List of k-scores
    k_range = list(range(k_range_min, k_range_max))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        score = metrics.accuracy_score(y_test, y_predict, normalize=True)
        k_scores.append( score)
    k_df = pd.DataFrame({'k_range': k_range, 'k_scores': k_scores})
    
    # Plot
        # Set figure
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")
        # Scatter Plot
    ax = sns.scatterplot(k_range, k_scores)
        # Title and Axis
    ax.set_title(title)
    ax.set_xlabel('Number of neighbors (K)')
    ax.set_ylabel('Accuracy Score')
#     ax.set_ylim(0.55,0.65)
    sns.despine()
        # Place horizontal line
    best_k_score = max(k_scores)
    optimal_k = k_scores.index(best_k_score)+1
    plt.axhline(xmax=(optimal_k-1)/len(k_range), y=best_k_score, ls="--", lw=3, c='r')
    plt.axvline(x=optimal_k , ymax=best_k_score+0.25, ls="--", lw=3, c='r', label='Optimal K Value')

    print('Highest Accuracy is {}%, when K is {}'.format(round(best_k_score*100,2), optimal_k))

    # Save the figure
    save_figure(ax, title)

def plot_feature_importance(importances, indices, features_to_show=10, title='Top {} Feature Importances'):
    '''Plot the importances of features in a dataset'''
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")

    ax = sns.barplot(importances[indices][:features_to_show],
                     features[indices][:features_to_show],
                     color='forestgreen')

    # Title and Axis
    ax.set_title(title.format(features_to_show))
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    sns.despine()

    # Save the figure
    save_figure(ax, title)

def plot_explained_variance(X_train_std, features_to_show=45, title='Total and Explained Variance'):
    '''Calculates the eigen values and eigen vectors in order to plot the explained variance of each principal component'''
    
    # Compute the covariance matrix of the standardized training dataset
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    
    # Compute cumulative sum of the explained variances
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    # Plot Explained Variance with step function
        # Set figure
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")
        # Plot
    ax = sns.barplot(list(range(1, len(eigen_vals)+1)), var_exp, color='royalblue', label='Individual Explained Variance')
    sns.lineplot(range(1, len(eigen_vals)+1), cum_var_exp, drawstyle='steps-pre', color='royalblue', label='Cumulative Explained Variance')
        # Title and Axis
    ax.set_title(title, fontsize='large', pad=20)
    ax.set_xlabel('Principal Component Index')
    ax.set_xticklabels(range(1, features_to_show+1))
    # ax.set_xticks(ticks=range(45))
    ax.set_xlim(left=-0.5, right=features_to_show-0.5)
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_ylim(top=cum_var_exp[features_to_show-1])
    ax.legend(loc='upper left')
    sns.despine()

    # Print cumulative results
    print('The top {} principal components explains {}% of the variance'.format(features_to_show,round(cum_var_exp[features_to_show-1]*100, 2)))
    
    # Save figure
    save_figure(ax, title)

def model_baseline(y_train_):
    '''Calculate the majority class and use the Zero Rule for a baseline model'''
    # Calculate the number of times each fighter (1 and 2) won a bout
    f1_wins = y_train_.value_counts()[1]
    f2_wins = y_train_.value_counts()[0]
    print('Fighter 1 wins: {}\nFighter 2 wins: {}'.format(f1_wins, f2_wins))

    # Work out which outcome is in the majority
    f1_wins_pct = y_train_.value_counts(normalize=True)[1]
    f2_wins_pct = y_train_.value_counts(normalize=True)[0]
    if f1_wins > f2_wins:
        print('\nOur Baseline Model will always predict Fighter 1 wins')
        majority_wins_pct = f1_wins_pct  
    else:
        print('\nOur Baseline Model will always predict Fighter 2 wins')
        majority_wins_pct = f2_wins_pct

    print('Null Rate: {}%'.format(round(majority_wins_pct*100, 2)))

    return majority_wins_pct 
    
def model_knn(X_train_, y_train_, X_test_, y_test_,n_neighbors=5):
    '''Performs a K Nearest Neighbours algorithm and returns the Accuracy of the model.  
    Also updates the model summary df'''
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_, y_train_)

    # Class prediction for testing data
    features_used = X_train_.shape[1]
    y_pred_class = knn.predict(X_test_[:,:features_used])

    # Print Results
    knn_accuracy = metrics.accuracy_score(y_test_, y_pred_class)
    print('Accuracy for K-Nearest Neighbors model (k = {}): {}%'.format(n_neighbors, round(knn_accuracy*100,1)))
    
    return knn_accuracy

def model_knn_grid(X_train_, y_train_):
    '''Uses a pipeline and Gridsearch for K-Nearest Neighbors to output the optimal hyperparameters'''
    
    # Build Pipeline
    pipe_knn = Pipeline([('scl', StandardScaler()), 
                         ('clf', KNeighborsClassifier(n_jobs=-1))
                        ]
                       )
    
    # Only select odd number of neighbors
    param_grid_knn = [
        {'clf__n_neighbors': list(range(3,25,2))}
    ]

    # Build GridSearch
    gs_knn = GridSearchCV(estimator=pipe_knn,
                param_grid=param_grid_knn,
                scoring='accuracy',
                cv=5, 
                verbose=0, 
                return_train_score = True)

    # Fit using grid search
    gs_knn.fit(X_train_, y_train_)

    # Best accuracy
    print('Best Accuracy: {}%'.format(round(gs_knn.best_score_ * 100, 2)))

    # Best params
    print('\nBest Params:\n', gs_knn.best_params_)

def model_logreg(C_, X_train_, y_train_, X_test_, y_test_):
    '''Performs a Logistic Regression on the '''
    
    logreg = LogisticRegression(C = C_, solver='lbfgs', random_state=123)
    model_log = logreg.fit(X_train_, y_train_)

    # Predict
    y_pred_class = logreg.predict(X_test_)
    lr_accuracy = metrics.accuracy_score(y_test_, y_pred_class)

    print('Accuracy for Logistic Regression model: {}%'.format(round(lr_accuracy*100,1)))

    return lr_accuracy
    
def model_logreg_grid(X_train_, y_train_):    
    '''Uses a pipeline and Gridsearch for Logistic Regression to output the optimal hyperparameters'''
    
    # Construct pipeline
    pipe_lr = Pipeline([('scl', StandardScaler()),
                ('clf', LogisticRegression(fit_intercept=False, random_state=123, solver='lbfgs'))])

    # Set grid search params
    param_grid_logreg = {'clf__C':np.logspace(-3,3,7)
    }

    # Construct grid search
    gs_lr = GridSearchCV(estimator=pipe_lr,
                param_grid=param_grid_logreg,
                scoring='accuracy',
                cv=5,
                verbose=1,
                return_train_score = True,
                n_jobs=-1)

    # Fit using grid search
    gs_lr.fit(X_train_, y_train_)

    # Best accuracy
    print('Best Accuracy: {}%'.format(round(gs_lr.best_score_ * 100, 2)))

    # Best params
    print('\nBest params:\n', gs_lr.best_params_)    
    
def model_random_forest(X_train_, y_train_, X_test_, y_test_,
                        max_depth_, min_samples_leaf_, min_samples_split_,
                        n_estimators_,
                        plot_importances=False, features_to_show=10):
    '''Performs a Random Forest Classification on the dataset'''
    clf = RandomForestClassifier(max_depth=max_depth_, min_samples_leaf=min_samples_leaf_, 
                             min_samples_split=min_samples_split_, n_estimators=n_estimators_, random_state=123)
    clf.fit(X_train_, y_train_)
    y_pred = clf.predict(X_test_)
    
    # Print Results
    rf_accuracy = metrics.accuracy_score(y_test_, y_pred)
    print('Accuracy for Random Forest model : {}%'.format(round(rf_accuracy*100,1)))
        
    # Plot the most important features
    features = X_train_.columns
    # importances = sorted(forest.feature_importances_, reverse=True)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    if plot_importances:
        plot_feature_importance(importances, indices, features)

    return rf_accuracy    

def plot_feature_importance(importances, indices, features_to_show=10, title='Top {} Feature Importances'):
    '''Plot the importances of features in a dataset'''
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")

    ax = sns.barplot(importances[indices][:features_to_show],
                     features[indices][:features_to_show],
                     color='forestgreen')

    # Title and Axis
    ax.set_title(title.format(features_to_show))
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    sns.despine()

    # Save the figure
    save_figure(ax, title)

def model_rf_grid(X_train_, y_train):    
    '''Uses a pipeline and Gridsearch for Random Forest to output the optimal hyperparameters'''
    
    pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state = 123,
                                                       n_jobs=-1
                                                      )
                        )
                       ]
                      )

    # Set grid search params
    param_grid_forest = [ 
      {'clf__n_estimators': [100, 110, 120],
       'clf__max_depth': [4, 5, 6],  
       'clf__min_samples_leaf':[0.05 ,0.1],  
       'clf__min_samples_split':[0.05 ,0.1]
      }
    ]

    # Construct grid search
    gs_rf = GridSearchCV(estimator=pipe_rf,
                param_grid=param_grid_forest,
                scoring='accuracy',
                cv=5, 
                verbose=1, 
                return_train_score = True)

    # Fit using grid search
    gs_rf.fit(X_train_, y_train)

    # Best accuracy
    print('Best accuracy: {:3f}'.format(gs_rf.best_score_))

    # Best params
    print('\nBest params:\n', gs_rf.best_params_)

    # Save to models_summary dataframe
    models_summary['Random Forest'] = gs_rf.best_score_

def rename_top_features(X_train_):
    '''Renames the top 10 features of our dataset to more understandable labels'''
    top_features = ['fighter1_win_rate', 'fighter2_win_rate', 'fighter2_slpm',
                    'fighter1_slpm', 'fighter2_td_def', 'fighter1_win', 
                    'fighter1_td_def', 'fighter2_str_def', 'fighter1_str_def', 'fighter2_win', 'fighter2_td_acc']  


    feature_labels = ["Fighter 1 Win Rate", "Fighter 2 Win Rate", "Fighter 2 Strikes Landed Per Minute",
                      "Fighter 1 Strikes Landed Per Minute", "Fighter 2 Take-Down Defence",
                      "Fighter 1 Total Wins", "Fighter 1 Take-Down Defence", "Fighter 2 Strike Defence",
                      "Fighter 1 Strike Defence", "Fighter 2 Total Wins", "Fighter 2 Take-Down Accuracy"]

    features_renamed = dict(zip(top_features, feature_labels))

    X_train_ = X_train_.rename(features_renamed, axis=1)
    return X_train_.rename(features_renamed, axis=1)    
    
def model_ada_boost(X_train_, y_train_, X_test_, y_test_, learning_rate, n_estimators):    
    '''Fits an Ada Boost algorithm to the dataset'''
    
    clf = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
    clf.fit(X_train_, y_train_)
    y_pred = clf.predict(X_test_)
    
    # Print Results
    ada_accuracy = metrics.accuracy_score(y_test_, y_pred)
    print('Accuracy for Ada Boost model : {}%'.format(round(ada_accuracy*100,1)))
        
    return ada_accuracy

def model_ada_boost_grid(X_train_, y_train):
    '''Uses a pipeline and Gridsearch for Ada Boost to output the optimal hyperparameters'''
    
    # Construct pipeline
    pipe_ab = Pipeline([('scl', StandardScaler()),
                ('clf', AdaBoostClassifier(random_state = 123))])

    # Set grid search params
    param_grid_adaboost = {
        'clf__n_estimators': [30, 50, 70, 90],
        'clf__learning_rate': [1.0, 0.5, 0.1, 0.01]
    }

    # Construct grid search
    gs_ab = GridSearchCV(estimator=pipe_ab,
                param_grid=param_grid_adaboost,
                scoring='accuracy',
                cv=5,
                verbose=1,
                return_train_score = True,
                n_jobs=-1)

    # Fit using grid search
    gs_ab.fit(X_train_, y_train)

    # Best accuracy
    print('Best Accuracy: {}%'.format(round(gs_ab.best_score_ * 100, 2)))

    # Best params
    print('\nBest params:\n', gs_ab.best_params_)

def model_xgboost(X_train, y_train, X_test, y_test):
    '''Fits an XG Boost algorithm to the dataset'''
    clf = xgb.XGBClassifier(learning_rate=0.1,
                            max_depth=5,
                            min_child_weight=10,
                            n_estimators=150,
                            subsample=0.7,
                            random_state=148,
                            n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    xgb_accuracy = metrics.accuracy_score(y_test, y_pred)
    # Print Results
    print('Accuracy for XGBoost model : {}%'.format(round(xgb_accuracy*100,1)))
    
    # For ensemble method
    xgb_pred_proba = clf.predict_proba(X_test)
    
    return (xgb_accuracy,xgb_pred_proba, y_test, y_pred, clf)     
    
def model_xgb_grid(X_train_, y_train_, X_test_, y_test_):    
    '''Uses a pipeline and Gridsearch for Random Forest to output the optimal hyperparameters'''
    
    # Construct pipeline
    clf = xgb.XGBClassifier()

    # Set grid search params
    param_grid = {
        "learning_rate": [0.1, 0.01],
        'max_depth': [4,5,6],
        'min_child_weight': [10],
        'subsample': [0.7],
        'n_estimators': [100, 150, 200]
    }

    grid_clf = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_clf.fit(X_train_, y_train_)

    best_parameters = grid_clf.best_params_
    print("Grid Search found the following optimal parameters: \n", best_parameters)

    training_preds = grid_clf.predict(X_train_)
    val_preds = grid_clf.predict(X_test_)
    training_accuracy = accuracy_score(y_train_, training_preds)
    val_accuracy = accuracy_score(y_test_, val_preds)

    print("")
    print("Training Accuracy:\t {:.4}%".format(training_accuracy * 100))
    print("Validation Accuracy:\t {:.4}%".format(val_accuracy * 100))

def model_compile_neural_network(X_train_std, activation):    
    '''Compile a Neural Network'''
    model = Sequential()

    # Input Layer
    model.add(Dense(64, input_dim=X_train_std.shape[1], activation = activation))
    # Hidden Layer 1
    model.add(Dense(32, input_dim = 64, activation = activation))
    # Hidden Layer 2
    model.add(Dense(units = 1, input_dim = 32, activation = 'sigmoid'))
    # Output Layer
    sgd_optimizer = SGD(lr=0.01, decay=1e-7, momentum=0.9) #learn rate, weight decay constant, momentum learning
    
    # Compile
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return model

def model_fit_neural_network(X_train_std, y_train, model, num_epochs):
    '''Fit model'''
    history_model = model.fit(X_train_std, y_train, batch_size=256, epochs=num_epochs, verbose=0, validation_split=0.1)  
    
    return history_model

def model_test_acc_neural_network(X_test_std, y_test, model):
    '''Calculate the Test Accuracy for the neurl network'''

    # Testing Accuracy
    y_test_pred = model.predict_classes(X_test_std, verbose=0)
    y_test_pred = [item for sublist in y_test_pred for item in sublist]
    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]
    
    print('Test Accuracy: {}%'.format(round(test_acc * 100), 2))

    return test_acc

def plot_compare_models(data, column):        
    '''Barchart to compare the different machine learning models'''
    values = data[column]
    clrs = ['lightgrey' if (x < max(values)) else 'forestgreen' for x in values ]
    ax = sns.barplot(x=column, y=list(data.index), data=data, palette=clrs)

    # Vertical Line at Baseline
    plt.axvline(x=min(values), ls= "--", lw=3, color='firebrick', label='Baseline Model Accuracy')
    
    # Title and Axis
    ax.set_title('Accuracy of Machine Learning Models', pad=20)
    ax.set_xlim(left=0, right=1)
    sns.despine()
    
# Cleaning functions

def calc_age_at_fight(data, new_col_name, fighter_dob, date='date'):
    '''Calculate the age of a fighter at the time of the fight, based on their date of birth'''
    # Calculate the time difference between 2 dates (in days)
    data[new_col_name] = data['date'] - data[fighter_dob]
    # Remove the 'days' value in column
    data[new_col_name] /= np.timedelta64(1, 'D')
    # Convert form days to years
    data[new_col_name] //= 365
    
def plot_sns_distplot(df, column, x_label):
    '''Plot the Distribution plot of a column'''
    
    # Set figure
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")

    # Plot
    ax = sns.distplot(df[column], bins=40, color='forestgreen', kde=False);
    
    # 50% vertical line
    plt.axvline(x=0.5, ls="--", lw=5, color='tomato', label='50% Win Rate')
    
    # Mean vertical line
    mean = np.mean(df[column])
    plt.axvline(x=mean, ls= "--", lw=5, color='darkgreen', label='Mean Win Rate: {}'.format(round(mean, 3)))
    
    # Title and Axis
    title = 'Distribution of {}'.format(x_label)
    ax.set_title(title, pad=25)
    ax.set_xlabel(x_label)
    ax.set_yticklabels('')
    sns.despine(left=True)
    ax.legend()
    
    # Save Figure
    save_figure(ax, title)

def plot_sns_displot_ages(list_, x_label):
    # Set figure
    sns.set(rc={'figure.figsize':(12,6)},style="white", context="talk")

    # Print min and max ages
    min_ = int(np.min(list_))
    max_ = int(np.max(list_))
    range_ = (max_ - min_)
    
    print("Fighters' Ages Range from {} to {}".format(min_, max_))
    # Plot
    ax = sns.distplot(list_, bins=range_, color='forestgreen', kde=False);
    
    # Mean vertical line
    mean = np.mean(list_)
    plt.axvline(x=mean, ls= "--", lw=3, color='darkgreen', label='Mean Age: {}'.format(round(mean, 1)))
    
    # Median vertical line
    median = np.median(list_)
    plt.axvline(x=median, ls= ":", lw=3, color='black', label='Median Age: {}'.format(round(median, 1)))
    
    # Title and Axis
    title = '{}'.format(x_label)
    ax.set_title(title, pad=25)
    ax.set_xlabel(x_label)
    ax.set_yticklabels('')
    sns.despine(left=True)
    ax.legend()
    
    # Save Figure
    save_figure(ax, title) 
    
    return {'mean': mean, 'median': median}
    
def column_countplot(dataframe, column, show_count=False):
    '''Plot the count of each category for a column in the dataframe'''
    
    # Set the figure
    sns.set(rc={'figure.figsize':(10,5)},style="white", context="talk")
    
    # Plot
    ax = sns.countplot(x=column, data=dataframe, color="forestgreen")
  
    # Title and Axis
    ax.set_title("Counts of each {} Category".format(column.title()))
    ax.set_xlabel(str(column).capitalize())
    ax.set_ylabel('Count')
    sns.despine()
    
    if show_count:
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+10))