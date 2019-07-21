# -*- coding: utf-8 -*-

# visualize

# code for visualization goes in this file.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def test_viz():
    print('In Visualize')
    pass

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

def column_distplot(dataframe, column, xlabel=None): 
    '''Plot the histogram of a column using Seaborn'''
    # Set Figure
    sns.set(rc={'figure.figsize':(10,5)},style="white", context="talk")

    # Plot
    column = column
    data = dataframe[column][~dataframe[column].isna()]
    ax = sns.distplot(data);

    # Title and Axis
    ax.set_title("Histogram of the fighters' {}".format(column));
    if type(xlabel)==str:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(str(column).capitalize())
    sns.despine()
    
    col_cap = str(column).capitalize()
    print("{} Skewness: {}".format(col_cap, round(dataframe[column].skew(), 2)))
    print("{} Kurtosis: {}".format(col_cap, round(dataframe[column].kurt(), 2)))
    
    print('{} Mean: {}'.format(col_cap, round(data.mean(), 2)))
    print('{} Median: {}'.format(col_cap, data.median()))
    
def column_countplot(dataframe, column, show_count=False):
    '''Plot the count of each category for a column in the dataframe'''
    
    # Set the figure
    sns.set(rc={'figure.figsize':(10,5)},style="white", context="talk")
    
    # Plot
    ax = sns.countplot(x=column, data=dataframe, color="b")
  
    # Title and Axis
    ax.set_title("Counts of each {} category".format(column))
    ax.set_xlabel(str(column).capitalize())
    ax.set_ylabel('Count')
    sns.despine()
    
    if show_count:
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+10))
