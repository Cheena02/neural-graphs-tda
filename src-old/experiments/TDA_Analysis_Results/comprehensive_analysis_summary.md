# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: False
    - **Compare Filtrations**: False
    - **Random Seed**: 42
    - **Datasets Processed**: ReportImages, test

    ## Results Overview
    - **Total Images Analyzed**: 8
    - **Total Comparisons**: 240
    - **Average Noise Impact**: 23746.62 features
    - **Average Recovery Rate**: -1.46%

    ## Noise Type Analysis
                     mean           std
noise_type                         
gaussian     21761.25  34194.840605
salt_pepper  25732.00  36747.315014

    ## Denoising Method Effectiveness
                          mean        std
denoise_method                       
bilateral_filter -1.087255   5.128255
median_filter    -1.900879  12.187781
non_local_means  -1.399270   6.612619

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 0.0000
    - **Average Wasserstein H1 (Clean->Noisy)**: 0.0000
    