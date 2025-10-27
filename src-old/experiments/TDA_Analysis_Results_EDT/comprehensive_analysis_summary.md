# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: True
    - **Compare Filtrations**: False
    - **Random Seed**: 42
    - **Datasets Processed**: ReportImages

    ## Results Overview
    - **Total Images Analyzed**: 7
    - **Total Comparisons**: 84
    - **Average Noise Impact**: 3467.95 features
    - **Average Recovery Rate**: -1.28%

    ## Noise Type Analysis
                        mean           std
noise_type                            
gaussian     7956.571429  17063.101390
salt_pepper -1020.666667   3081.959257

    ## Denoising Method Effectiveness
                          mean      std
denoise_method                     
bilateral_filter -0.179993  1.22022
median_filter    -0.607500  2.78186
non_local_means  -3.039835  5.46253

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 0.0000
    - **Average Wasserstein H1 (Clean->Noisy)**: 0.0000
    