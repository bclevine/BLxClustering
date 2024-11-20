# BLxClustering

Code for "Galaxy Clustering with LSST: Effects of Number Count Bias from Blending" by Levine et al. Contents of repository:

1. `Analysis_FINAL_Mar1.ipynb`: initializes and runs MCMC on correlation functions for the fiducial (Y1/true redshift) analysis.
2. `Analysis_Y5_Jul25.ipynb`: same as above, but for Y5 magnitude cut.
3. `Analysis_Y5_photz_Sep3.ipynb`: same as above, but for Y5 magnitude cut and photometric redshifts.
4. `Analysis_photz_Sep3.ipynb`: same as above, but for Y1 magnitude cut and photometric redshifts.
5. `Final_Plots.ipynb`: produces high-level plots of correlation functions and cosmological analyses.
7. `blending_correlations_fiducial.ipynb`: generates redshift distributions and correlation functions from GCRCatalogs for the fiducial analysis; also produces various low-level plots. Run on NERSC using `desc-stack-weekly-latest` kernel.
8. `blending_correlations_Y5.ipynb`: same as above, but for Y5 magnitude cut.
9. `blending_correlations_pz_Y1.ipynb`: same as above, but for Y1 magnitude cut and photometric redshifts.
10. `blending_correlations_pz_Y5.ipynb`: same as above, but for Y5 magnitude cut and photometric redshifts.
11. `compare_different_cuts.ipynb`: produce comparison plots between the fiducal analysis and the Y1/Y5 and true z/photo z analyses.
14. `helper_functions.py`: contains helper functions for `Analysis_x.ipynb` and `Final_Plots.ipynb`.
