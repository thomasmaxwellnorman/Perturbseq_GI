# Perturbseq library for loading and manipulating single-cell experiments
# Copyright (C) 2019  Thomas Norman

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def nzflat(M, k=1):
    """Return the non-zero part of the upper triangle of a matrix in stacked format
    """
    t = M.where(np.triu(np.ones(M.shape), k=k).astype(np.bool)).stack()
    return t[t != 0]

def upper_triangle(M, k=1):
    """Return the upper triangular part of a matrix in stacked format (i.e. as a vector)
    """
    keep = np.triu(np.ones(M.shape), k=k).astype('bool').reshape(M.size)
    # have to use dropna=FALSE on stack, otherwise will secretly drop nans and upper triangle
    # will not behave as expected!!
    return M.stack(dropna=False).loc[keep]

def _strip_cat_cols(df):
    """Convert categorical columns to string labels (as categorical columns cause problems
    for HDF5 storage)
    """
    cat_cols = df.select_dtypes('category').columns
    if len(cat_cols) > 0:
        print('! Converting categorical columns to string...')
        out = df.copy()
        for col in cat_cols:
            out[col] = out[col].astype('str')
    else:
        out = df
    return out
    
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # From: https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient
