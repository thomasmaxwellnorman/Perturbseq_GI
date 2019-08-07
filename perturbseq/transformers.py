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

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
import tsne as tsne_bh
from umap import UMAP

def _prepare_dataframe(Z, index=None, columns=None, prefix=None):
    """Helper function to convert a numpy array to a pandas DataFrame
    """
    Z = pd.DataFrame(Z)
    
    if index is not None:
        Z.index = index
    if columns is not None:
        Z.columns = columns
    elif prefix is not None:
        Z.columns = [prefix + '{0}'.format(i) for i in Z.columns]
        
    return Z

class PCAReducer(BaseEstimator, TransformerMixin):
    """Class that implements dimensionality reduction of a CellPopulation by
    principal components analysis
    
    Args:
        PCA: fitted sklearn PCA object
        loadings_: DataFrame of loadings for each gene on each PC
        reduced_: DataFrame of PCs for each cell
    """
    def __init__(self, n_components=10, svd_solver='arpack', whiten=False, random_state=None):
        self.PCA = PCA(n_components=n_components,
                      svd_solver=svd_solver,
                      whiten=whiten,
                      random_state=random_state)
        
        self.loadings_ = None
        self.reduced_ = None
        
    def fit(self, X, y=None):
        Z = self.PCA.fit_transform(X)
        self.reduced_ = _prepare_dataframe(Z, index=X.index, prefix='PCA')
        self.loadings_ = _prepare_dataframe(self.PCA.components_.T, index=X.columns, prefix='PCA')
        self.explained_variance_ratio_ = pd.Series(self.PCA.explained_variance_ratio_, index=self.loadings_.columns)
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.reduced_
        
    def transform(self, X, y=None):
        Z = self.PCA.transform(X)
        if isinstance(X, pd.DataFrame):
            Z = _prepare_dataframe(Z, index=X.index, prefix='PCA')
        return Z
    
class ICAReducer(BaseEstimator, TransformerMixin):
    """Class that implements dimensionality reduction of a CellPopulation by
    independent components analysis
    
    Args:
        ICA: fitted sklearn FastICA object
        reduced_: DataFrame of ICs for each cell
        mixing_matrix_: DataFrame of mixing matrix for each gene on each IC
        sort_components: if True, sort ICs by the norms of the mixing matrix columns (as there
            is no canonical ordering of ICs)
    """
    
    def __init__(self, n_components=10, algorithm='deflation', tol=5e-6, fun='logcosh', max_iter=1000, sort_components=True):
        self.ICA = FastICA(n_components,
                           fun=fun,
                           algorithm=algorithm,
                           tol=tol,
                           max_iter=max_iter)
        
        self.reduced_ = None
        self.mixing_matrix_ = None
        self.sort_components = sort_components
        
    def fit(self, X, y=None):
        Z = self.ICA.fit_transform(X)
        print("Finished after " + str(self.ICA.n_iter_) + " iterations.")
        
        mixing = self.ICA.mixing_
        
        if self.sort_components:
            energy = np.sqrt(np.square(mixing).sum(axis=0))
            order = np.argsort(energy)
            mixing = mixing[:, order[::-1]]
            Z = Z[:, order[::-1]]
            
        self.reduced_ = _prepare_dataframe(Z, index=X.index, prefix='ICA')
        self.mixing_matrix_ = _prepare_dataframe(mixing, index=X.columns, prefix='ICA')
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.reduced_
        
    def transform(self, X, y=None):
        Z = self.ICA.transform(X)
        if isinstance(X, pd.DataFrame):
            Z = _prepare_dataframe(Z, index=X.index, prefix='ICA')
        return Z  
    
class PCATSNEReducer(BaseEstimator, TransformerMixin):
    """Class that implements dimensionality reduction of a CellPopulation by
    principal components analysis followed by t-sne
    
    Args: 
        PCA: fitted sklearn PCA object
        TSNE: if using sklearn, fitted TSNE object
        pca_matrix_: PCs for each cell
        reduced_: DataFrame of t-sne coordinates for each cell
        use_pca: whether to use PCA reduction first
        use_sklearn: whether to use the sklearn t-sne implementation or the C++ implementation
        n_components: number of principal components
        
        Other parameters relate to the algorithms
    """
    
    def __init__(self, n_components=10, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 use_sklearn=False, use_pca=True, svd_solver='auto'):
        
        self.use_pca = use_pca
        self.use_sklearn = use_sklearn
        
        if use_pca:
            self.PCA = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
        else:
            self.PCA = None     
        
        if use_sklearn:
            self.TSNE = TSNE(perplexity=perplexity,
                 early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress, min_grad_norm=min_grad_norm,
                 metric=metric, init=init, verbose=verbose,
                 random_state=random_state, method=method, angle=angle)
        else:
            self.TSNE = None
            
        self.n_components = n_components
        self.angle = angle
        self.perplexity = perplexity
        self.reduced_ = None
        self.pca_matrix_ = None
        
    def fit(self, X, y=None):
        if self.use_pca:
            print('Performing PCA...')
            Y = self.PCA.fit_transform(X)
            self.pca_matrix_ = Y.copy()
        else:
            Y = X
            
        print('Performing TSNE...')
        if self.use_sklearn:
            Z = self.TSNE.fit_transform(Y)
        else:
            if isinstance(Y, pd.DataFrame):
                Yp = Y.values
            else:
                Yp = Y.copy()
            Z = tsne_bh.bh_sne(Yp,
                               d=2,
                               theta=self.angle,
                               perplexity=self.perplexity)
        print('Done.')
        
        self.reduced_ = _prepare_dataframe(Z, index=X.index, prefix='TSNE')
     
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.reduced_
 
    def transform(self, X, y=None):
        return 'Not implemented'
    
class PCAUMAPReducer(BaseEstimator, TransformerMixin):
    """Class that implements dimensionality reduction of a CellPopulation by principal components
    analysis followed by UMAP
    
    Args:
        n_components: number of principal components
        metric: which metric to use with UMAP (default: 'euclidean')
        n_neighbors: number of neighbors to use for UMAP (default: 10)
        random_state: can set for reproducibility
        PCA: fitted sklearn PCA object
        UMAP: fitted UMAP object
        reduced_: DataFrame of UMAP coordinates for each cell
        graph_: nearest neighbor graph from UMAP
        pca_matrix_: PCs for each cell
        use_pca: whether to use PCA reduction first
    """
    
    def __init__(self, n_components=10, svd_solver='auto', metric='euclidean', n_neighbors=10, random_state=None, use_pca=True):
        
        if use_pca:
            self.PCA = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
        else:
            self.PCA = None     
        
        self.UMAP = UMAP(metric=metric, n_neighbors=n_neighbors, random_state=random_state)
        self.n_components = n_components
        self.reduced_ = None
        self.pca_matrix_ = None
        self.use_pca = use_pca
        self.graph_ = None
        
    def fit(self, X, y=None):
        if self.use_pca:
            print('Performing PCA...')
            Y = self.PCA.fit_transform(X)
            self.pca_matrix_ = Y.copy()
        else:
            Y = X
            
        print('Performing UMAP...')
        Z = self.UMAP.fit_transform(Y)
        self.graph_ = self.UMAP.graph_
        print('Done.')
        
        self.reduced_ = _prepare_dataframe(Z, index=X.index, prefix='UMAP')
     
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.reduced_
 
    def transform(self, X, y=None):
        return 'Not implemented'