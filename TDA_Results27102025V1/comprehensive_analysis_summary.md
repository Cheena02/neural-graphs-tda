# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: False
    - **Compare Filtrations**: True
    - **Random Seed**: 42
    - **Datasets Processed**: ReportImages, test

    ## Results Overview
    - **Total Images Analyzed**: 8
    - **Total Comparisons**: 400
    - **Average Noise Impact**: 23457.56 features
    - **Average Recovery Rate**: -5.89%

    ## Noise Type Analysis
                      mean           std
noise_type                          
gaussian     21385.640  34387.660443
salt_pepper  25529.475  36607.905052

    ## Denoising Method Effectiveness
                                 mean        std
denoise_method                              
bilateral_filter        -4.383947  25.702615
median_filter           -8.855626  62.328368
morphological_denoising -5.370561  19.224217
non_local_means         -5.660226  18.998105
topological_denoising   -5.158618  22.039213

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 2970.7111
    - **Average Wasserstein H1 (Clean->Noisy)**: 168.4156
    