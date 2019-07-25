# -*- coding: utf-8 -*-
# make_data.py

# General Imports
import pandas as pd

from .pandas_operators import drop_df_columns, insert_df_column
def test_make_data():
    print("In make_data")
    pass

# stubs for cleaning and feature engineering

def clean_data(df):
    """ Stub to clean data:
    :param: df : dataframe to clean
    
    """
    pass


def make_features(df, cols, new_cols):
    """ Stub to create and or transform features :
    :param: df : dataframe and column names create or transform features
    
    """

    pass

def list_to_lowercase(_list):
    """Takes in a list of strings and returns the list where each item is now lowercase:
    :param: _list : list which you would like to change
    """
    _list = list(map(lambda x: x.lower(), _list))
    return _list

def parse_height(height):
    '''Splits the height value into feet and inches'''
    ht_ = height.split(" ")
    ft_ = float(ht_[0])
    in_ = float(ht_[1])
    return (12*ft_) + in_