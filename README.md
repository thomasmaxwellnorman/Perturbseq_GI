# Perturbseq_GI

These notebooks contain code reproducing the single-cell analyses from:

Norman, T.M., Horlbeck, M.A., Replogle, J.M., Ge, A.Y., Xu, A., Jost, M., Gilbert, L.A., & Weissman, J.S. "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes", *Science*, 2019.

This repository also contains a version of a library for loading and manipulating Perturb-seq experiments (in the `perturbseq` subdirectory). A fully self-contained tutorial for using this library can be found in the `perturbseq_demo` repository, and it may be useful to go through that before attempting to use these notebooks. Finally, this repository also contains a Numpy implementation of the Maxide method for constrained matrix completion (in `maxide.py`).

In order to use the notebooks, you will need to download the sequencing data from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344. Only the outputs from `cellranger` are necessary. They should be placed in a directory structure mimicking the output of `cellranger` (i.e. with an `outs` folder and appropriate `raw_gene_bc_matrices_mex` and `filtered_gene_bc_matrices_mex` subdirectories). The `cell_identities.csv` files should be placed in the `outs` folder.

The notebooks are commented but are not "production" software. As such some tinkering will be necessary to get dependencies installed and data into appropriate locations. The appropriate starting point is the notebook *GI_generate_populations* which does basic loading and normalization of the raw single-cell sequencing data. Most of the notebooks expect to be run from Python 2.7 kernels, though the underlying Perturb-seq library is compatible with Python 3 as well.
