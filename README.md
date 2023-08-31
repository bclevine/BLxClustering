# Blending_Project_2023

Code for "The Impact of Blending on Galaxy Clustering Analyses in LSST" by B. Levine et al. Contents of repository:

1. `blending_correlations_latest.ipynb`: generates redshift distributions and correlation functions from GCRCatalogs; also produces various low-level plots. Run on NERSC using `desc-stack-weekly-latest` kernel.
2. `Analysis_FINAL_Aug3.ipynb`: initializes and runs MCMC on corrleaiton functions.
3. `Final_Plots.ipynb`: produces high-level plots of correlation functions and cosmological analyses.
4. `helper_functions.py`: contains helper functions for `Analysis_FINAL_Aug3.ipynb` and `Final_Plots.ipynb`.
