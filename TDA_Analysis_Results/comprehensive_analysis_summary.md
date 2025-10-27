# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: False
    - **Compare Filtrations**: False
    - **Random Seed**: 42
    - **Datasets Processed**: ReportImages

    ## Results Overview
    - **Total Images Analyzed**: 7
    - **Total Comparisons**: 84
    - **Average Noise Impact**: 34435.30 features
    - **Average Recovery Rate**: -0.05%

    ## Noise Type Analysis
                         mean           std
noise_type                             
gaussian     30902.833333  42765.988434
salt_pepper  37967.761905  43469.651942

    ## Denoising Method Effectiveness
                          mean       std
denoise_method                      
bilateral_filter  0.029815  2.064906
median_filter    -0.197322  2.603617
non_local_means   0.019697  2.049322

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 0.0000
    - **Average Wasserstein H1 (Clean->Noisy)**: 0.0000
    