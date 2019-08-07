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

from __future__ import absolute_import

__version__ = 0.1

from .cell_population import CellPopulation, MeanPopulation, fancy_dendrogram, fit_dendrogram, correlation_heatmap, metaapply
from .expression_normalization import z_normalize_expression, normalize_to_control, normalize_matrix_to_control, strip_low_expression, log_normalize_expression, equalize_UMI_counts, normalize_to_gemgroup_control, inherit_normalized_matrix, normalize_matrix_by_key
from .cell_cycle import get_cell_phase_genes, add_cell_cycle_scores, cell_cycle_position_heatmap
from .transformers import PCAReducer, ICAReducer, PCATSNEReducer, PCAUMAPReducer
from .differential_expression import ks_de, ad_de, tree_selector, TreeSelectorResult, find_noisy_genes
from .util import upper_triangle, nzflat, gini