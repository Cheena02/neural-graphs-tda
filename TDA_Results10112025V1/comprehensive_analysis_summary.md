# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: True
    - **Compare Filtrations**: True
    - **Random Seed**: 42
    - **Datasets Processed**: ReportImages

    ## Results Overview
    - **Total Images Analyzed**: 7
    - **Total Comparisons**: 350
    - **Average Noise Impact**: 2708.55 features
    - **Average Recovery Rate**: -5.38%

    ## Noise Type Analysis
                        mean           std
noise_type                            
gaussian     6411.937143  14870.575061
salt_pepper  -994.828571   2918.765066

    ## Denoising Method Effectiveness
                                  mean        std
denoise_method                               
bilateral_filter         -2.995435  14.828570
median_filter            -0.863529   2.937405
morphological_denoising -12.090667  25.176916
non_local_means          -5.745244   9.715479
topological_denoising    -5.208304  39.197290

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 3156.9340
    - **Average Wasserstein H1 (Clean->Noisy)**: 10672.7543
    