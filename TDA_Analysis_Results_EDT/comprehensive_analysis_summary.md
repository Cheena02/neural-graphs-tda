# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: True
    - **Compare Filtrations**: False
    - **Random Seed**: 42
    - **Datasets Processed**: ReportImages

    ## Results Overview
    - **Total Images Analyzed**: 5
    - **Total Comparisons**: 60
    - **Average Noise Impact**: 317.17 features
    - **Average Recovery Rate**: -0.15%

    ## Noise Type Analysis
                       mean          std
noise_type                          
gaussian     717.333333  1383.267504
salt_pepper  -83.000000  1285.435603

    ## Denoising Method Effectiveness
                          mean       std
denoise_method                      
bilateral_filter -0.011263  0.760382
median_filter     0.257282  0.830857
non_local_means  -0.709917  1.693383

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 0.0000
    - **Average Wasserstein H1 (Clean->Noisy)**: 0.0000
    