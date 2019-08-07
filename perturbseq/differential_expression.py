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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, anderson_ksamp
from statsmodels.stats.multitest import multipletests
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def find_noisy_genes(pop, noisy_threshold=0.05, mean_threshold=0.05, exclude=[], resolution=1000):
    """Finds genes that exceed the baseline relationship observed between mean and coefficient
    of variation (i.e. genes that are overdispersed). Briefly, a curve fitting procedure is
    applied to establish the baseline expected CV for a given expression level. This is used
    to define 
        excess CV = observed CV - predicted CV
    The function returns genes based on the quantile of excess CV they lie in.
    
    Args:
        pop: CellPopulation instance
        noisy_threshold: quantile of excess CV (e.g. 0.05 returns top 5% of genes with abnormally
            high CVs for their expression level)
        mean_threshold: only consider genes with mean expression level above this value
        exclude: list of gene names or gene ids to not include in the search (e.g. perturbed genes) 
        resolution: number of bins to use when interpolating mean-CV relationship
        
    Returns:
        List of overdispersed gene ids
        
    Example:
        >>>noisy_genes = find_noisy_genes(pop, exclude=perturbation_ids)
        would return the top 5% of genes showing unexpectedly high CVs, excluding those that were
        in the list perturbation_ids (meaning the genes that are perturbed in the experiment)
    """
    if len(exclude) > 0:
        exclude = pop.gene_ids(exclude)
    
    # median filter the CVs in descending order of abundance
    # For single-cell data, this will pick out the background 1/sqrt(<x>) trend
    thresholded_genes = pop.genes.query('mean > @mean_threshold').sort_values('mean', ascending=False)
    
    gene_means_mean_ordered = thresholded_genes['mean']
    gene_cvs_mean_ordered = thresholded_genes['cv']
    gene_cvs_mean_ordered_medfilt = pd.Series(medfilt(gene_cvs_mean_ordered, kernel_size=15), index=gene_cvs_mean_ordered.index)
    
    ind = np.arange(0, len(gene_means_mean_ordered), len(gene_means_mean_ordered)/resolution)
    cv_interpolater = interp1d(gene_means_mean_ordered.iloc[ind], gene_cvs_mean_ordered_medfilt.iloc[ind], fill_value="extrapolate")
    
    predicted_cv = pd.Series(cv_interpolater(gene_means_mean_ordered), index=gene_means_mean_ordered.index)
    excess_cv = gene_cvs_mean_ordered/predicted_cv
    thresholded_genes['excess_cv'] = excess_cv
    
    cv_threshold = excess_cv.quantile(1 - noisy_threshold)
    noisy_genes_raw = thresholded_genes.query('excess_cv > @cv_threshold').index.values
    noisy_genes = thresholded_genes.query('excess_cv > @cv_threshold and index not in @exclude').index.values
    
    plt.scatter(1/np.sqrt(thresholded_genes['mean']), thresholded_genes['cv'], s=5, alpha=0.25)
    plt.plot(1/np.sqrt(thresholded_genes['mean']), cv_interpolater(thresholded_genes['mean']), c='gray', alpha=0.5)
    plt.scatter(1/np.sqrt(pop.genes.loc[noisy_genes, 'mean']), pop.genes.loc[noisy_genes, 'cv'], s=5, c='r')
    print('{0} variable genes found ({1} excluded)'.format(len(noisy_genes), len(np.intersect1d(noisy_genes_raw, exclude))))
    
    return noisy_genes

def ks_de(pop, key, control_cells, genes=None, cells=None, normalized=False, n_jobs=1, alpha=0.05, multi_method='fdr_by', **kwargs):
    """Look for differential gene expression relative to a control population based on Kolmogorov-Smirnov test.
    The function will do the test for each subpopulation defined by a category in pop.cells.
    
    Args:
        pop: CellPopulation instance to look in
        key: name of column in pop.cells metadata that defines subpopulations
        control_cells: a query of pop that defines the control cell population that differences are defined with respect to
        genes: query of which genes to consider (e.g. 'mean > 0.1')
        normalized: use normalized expression matrix for comparison (default: False)
        n_jobs: number of cores to use in parallel processing (default: 1)
        alpha: FWER/FDR in multiple hypothesis testing correction
        multi_method: method of multiple hypothesis testing correction (default: 'fdr_by')
    
    Returns:
        ks_matrix: matrix of test statistics for each gene in each subpopulation against the control population
        p_matrix: p-values
        adj_p_matrix: p-values corrected for multiple hypothesis testing

    Example: 
        >>>ks, p, adj_p = ks_de(pop,
                                key='guide_target',
                                control_cells='guide_target == "control"',
                                genes='mean > 0.25',
                                normalized=True,
                                n_jobs=16)
    """

    control_matrix = pop.where(cells=control_cells, genes=genes, normalized=normalized, **kwargs)
    print("{0} control cells".format(control_matrix.shape[0]))
    subpops = pop.groupby(key, cells=cells, genes=genes, normalized=normalized, **kwargs)
    
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_ks_compare_pops)(subpop, control_matrix, name) for name, subpop in subpops)
    
    Ks, ps = zip(*out)
    Ks = pd.DataFrame(list(Ks)).T
    ps = pd.DataFrame(list(ps)).T
    
    adj_ps = ps.copy()
    adj_ps = adj_ps.apply(lambda x: _multi_test_correct(x, alpha, multi_method))
    
    return Ks, ps, adj_ps

def ad_de(pop, key, control_cells, genes=None, cells=None, normalized=False, n_jobs=1, alpha=0.05, multi_method='fdr_by', **kwargs):
    """Look for differential gene expression relative to a control population based on Anderson-Darling test.
    The function will do the test for each subpopulation defined by a category in pop.cells. See documentation
    for ks_de.
    """
    control_matrix = pop.where(cells=control_cells, genes=genes, normalized=normalized, **kwargs)
    print("{0} control cells".format(control_matrix.shape[0]))
    subpops = pop.groupby(key, cells=cells, genes=genes, normalized=normalized, **kwargs)
    
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_anderson_compare_pops)(subpop, control_matrix, name) for name, subpop in subpops)
    
    ADs, ps = zip(*out)
    ADs = pd.DataFrame(list(ADs)).T
    ps = pd.DataFrame(list(ps)).T
    
    adj_ps = ps.copy()
    adj_ps = adj_ps.apply(lambda x: _multi_test_correct(x, alpha, multi_method))
    
    return ADs, ps, adj_ps

def _anderson_compare_pops(first_pop_matrix, second_pop_matrix, name=None):
    """Helper function used to execute Anderson-Darling test. See anderson_de.
    """
    AD_stats = dict()
    p_stats = dict()
    for gene_id in first_pop_matrix.columns:
        AD, _, p = anderson_ksamp([first_pop_matrix[gene_id],
                               second_pop_matrix[gene_id]])
        AD_stats[gene_id] = AD
        p_stats[gene_id] = p
    return pd.Series(AD_stats, name=name), pd.Series(p_stats, name=name)

def _multi_test_correct(p, alpha, multi_method):
    """Helper function for multiple hypothesis testing correction
    """
    _, corr_p_values, _, _ = multipletests(p, alpha=alpha, method=multi_method)
    return corr_p_values
    
def _ks_compare_pops(first_pop_matrix, second_pop_matrix, name=None):
    """Helper function used to execute Kolmogorov-Smirnov test. See ks_de.
    """
    KS_stats = dict()
    p_stats = dict()
    for gene_id in first_pop_matrix.columns:
        KS, p = ks_2samp(first_pop_matrix[gene_id], \
                                     second_pop_matrix[gene_id])
        KS_stats[gene_id] = KS
        p_stats[gene_id] = p
    return pd.Series(KS_stats, name=name), pd.Series(p_stats, name=name)
       
def _prep_X_y(pop, key, cells=None, genes=None, normalized=True, feature_table=None, ignore=None, verbose=False, **kwargs):
    """Helper function that formats expression data and class labels to feed into classifier
    """
    # are we using gene expression or an auxiliary table of features?
    if feature_table is None:
        matrix = pop.where(cells=cells, genes=genes, normalized=normalized, **kwargs)
    else:
        matrix = feature_table
    
    # genes to ignore
    if ignore is not None:
        test_gene = ignore[0]      
        if test_gene[0:4] != 'ENSG': # list of ensembl ids
            ignore = pop.gene_ids(ignore)
        matrix.drop(ignore, inplace=True, axis=1, errors='ignore')
        
    # get class labels
    X = matrix.values
    y_values, y = np.unique(pop.cells.loc[matrix.index, key], return_inverse=True)
    
    if verbose:
        if feature_table is None:
            print("Training (genes: {0})...".format(genes))
        else:
            print("Training using supplied feature table...")
    
    return matrix, X, y, y_values
    
def _get_tree_classifier(clf, n_jobs=1, n_estimators=None, random_state=None):
    """Helper function that gets an appropriate random forest classifier
    """
    if isinstance(clf, basestring):
        if clf == 'extra':
            tree = ExtraTreesClassifier(n_jobs=n_jobs, n_estimators=n_estimators, class_weight='balanced', random_state=random_state)
        elif clf_type == 'random':
            tree = RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators, class_weight='balanced', random_state=random_state)
    else:
        tree = clf
    return tree
    
# taken from BorutaPy https://github.com/scikit-learn-contrib/boruta_py
def _get_tree_num(n_feat, depth=None):
    if depth == None:
        depth = 10
    # how many times a feature should be considered on average
    f_repr = 100
    # n_feat * 2 because the training matrix is extended with n shadow features
    multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
    n_estimators = int(multi * f_repr)
    return n_estimators
    
def _test_feature_performance(X, y, clf='extra', n_estimators=None, random_state=None, test_size=0.2, target_names=None, n_jobs=1):
    """Test classifier performance on 20% of excluded data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if n_estimators is None:
        n_estimators = _get_tree_num(X.shape[1], depth=None)
    
    print('Using {0} estimators for {1} features...').format(n_estimators, X.shape[1])
    clf_tree = _get_tree_classifier(clf, n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)
    clf_tree.fit(X_train, y_train)
    
    y_predict = clf_tree.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    print('Feature prediction accuracy (test size: {1:.1f}%): {0}\n'.format(acc, 100*test_size))
    report = classification_report(y_test, y_predict, target_names=target_names)
    print(report)
    
    return clf_tree, acc, report, clf_tree.feature_importances_
     
def tree_selector(pop, key, num_features=None, cells=None, genes=None, normalized=True, feature_table=None, ignore=None, clf='extra', n_jobs=1, random_state=None, n_estimators=None, **kwargs):
    """Select genes that vary using a random forest classifier. Briefly, the approach uses a categorical property
    within a population to subdivide it into groups (e.g. the gene targeted for perturbation). Cells are then
    used as training data for a random forest classifier that predicts the categorical property from gene 
    expression data. Genes that vary across condition then naturally fall out as the most "important", with the
    advantage that this approach scales trivially to comparisons among >2 populations. The classifier will be
    trained on only 80% of supplied data, with the remaining 20% withheld to assess accuracy.
    
    Args:
        pop: CellPopulation instance
        key: name of property in pop.cells used to subdivide population into classes
        num_features: only allow the random forest to use this many genes for classification
        cells, genes: queries for pop.where to select subpopulation if desired
        normalized: whether to train on normalized expression data
        feature_table: auxiliary table to use instead of gene expression data
        ignore: list of gene names or ids to ignore when classifying (e.g. perturbed genes will obviously vary
            across condition)
        clf: type of random forest classifier
        n_jobs: number of cores to use
        random_state: can supply for reproducible results
        n_estimators: number of trees in forest. If not provided a rule of thumb will be used
        **kwargs: any additional keyword arguments are passed to pop.where
        
    Returns:
        A TreeSelectorResult object containing a trained classifier, chosen genes, and their importances
        
    Example:
        >>>res = tree_selector(pop,
                               key='guide_target',
                               num_features=100,
                               genes='mean > 0.25',
                               normalized=True,
                               n_jobs=16)
        would train a classifier to recognize which guide cells had received using the expression of 100 genes.
    """
    # get expression daata
    matrix, X, y, y_values = _prep_X_y(pop, key, cells=cells, genes=genes, normalized=normalized, feature_table=feature_table, ignore=ignore, verbose=True, **kwargs)
        
    # choose number of trees in forest
    if n_estimators is None:
        n_estimators_to_use = _get_tree_num(X.shape[1], depth=None)
    else:
        n_estimators_to_use = n_estimators
    # train initial classifier on all possible genes
    clf_tree, total_acc, total_report, total_importances = _test_feature_performance(X, y, clf=clf, n_estimators=n_estimators_to_use, random_state=random_state, target_names=y_values, n_jobs=n_jobs)
    
    if num_features is not None:
        # retrain final classifier using only top features   
        idx = np.argsort(clf_tree.feature_importances_)[::-1]
        gene_indices = idx[0:num_features]
        print("Picked {0} features.".format(num_features))
        selected_genes = matrix.columns[gene_indices]
        selected_importances = clf_tree.feature_importances_[gene_indices]
        selected_importances = pd.Series(selected_importances, index=pop.gene_names(selected_genes)).sort_values(ascending=False)

        if n_estimators is None:
            n_estimators_to_use = _get_tree_num(num_features, depth=None)
        else:
            n_estimators_to_use = n_estimators
        
        clf_selected, acc, report, importances = _test_feature_performance(X[:, gene_indices], y, clf=clf, n_estimators=n_estimators_to_use, random_state=random_state, target_names=y_values, n_jobs=n_jobs)
        importances = pd.Series(importances, index=pop.gene_names(selected_genes)).sort_values(ascending=False)
    else:
        # if not asked to restrict number of genes, just return the initial classifier
        clf_selected = clf_tree
        selected_genes = matrix.columns
        importances = pd.Series(total_importances, index=pop.gene_names(selected_genes)).sort_values(ascending=False)
        total_importances = importances
        acc = total_acc
        report = total_report
    
    return TreeSelectorResult(clf_selected, selected_genes, total_importances, importances, acc, report, y_values)

class TreeSelectorResult():
    """A class for holding the results of a random forest approach to selecting differentially expressed genes
    
    Attributes:
        classifier: the trained random forest classifer
        selected_genes: differentially expressed genes identified by the procedure
        importances: relative ranking of the importances of differentially expressed genes
        total_importances: relative ranking of the importances of all genes (selected and nonselected)
        acc: accuracy of prediction on 20% of withheld data
        report: a string report describing classifier performance
        categories: labels of subdivisions of original CellPopulation that were used to divide cells
        numerical_categories: corresponding numerical labels returned by the classifiers predict method
    """
    def __init__(self, classifier, selected_genes, total_importances, importances, acc, report, categories):
        self.classifier = classifier
        self.selected_genes = selected_genes
        self.total_importances = total_importances
        self.importances = importances
        self.acc = acc
        self.report = report
        self.categories = pd.Series({i: cat for i, cat in enumerate(categories)})
        self.numerical_categories = pd.Series({cat: i for i, cat in enumerate(categories)})

    def __repr__(self):
        s = '{0} differentially expressed features\n'.format(len(self.selected_genes))
        s = s + 'Feature prediction accuracy: {0}\n\n'.format(self.acc)
        return s + self.report
    
    def predict(self, matrix):
        """Return label predictions for cells in gene expression data
        
        Args:
            matrix: gene expression data for cells (DataFrame)
            
        Returns:
            A series of predicted labels
        """
        X = self.transform(matrix)
        y = self.classifier.predict(X)
        y = pd.Series(y, index=matrix.index)
        return y.map(self.categories)
    
    def transform(self, matrix):
        """Subset gene expression data to differentially expressed genes 
        
        Args:
            matrix: gene expression data for cells (DataFrame)

        Returns:
            Subsetted gene expression data
        """
        X = matrix[self.selected_genes]
        return X   
    
    def score(self, matrix, categories):
        """Classifier performance
        
        Args:
            matrix: gene expression data for cells (DataFrame)
            categories: class labels for cells (Series)
           
        Returns:
            accuracy of prediction of labels on this dataset
        """
        X = self.transform(matrix)
        y_values, y = np.unique(categories, return_inverse=True)
        yp = self.predict(X).map(self.numerical_categories)
        return accuracy_score(y, yp)