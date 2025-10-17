# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: False
    - **Compare Filtrations**: False
    - **Random Seed**: 42
    - **Datasets Processed**: ReportImages

    ## Results Overview
    - **Total Images Analyzed**: 5
    - **Total Comparisons**: 60
    - **Average Noise Impact**: 44655.12 features
    - **Average Recovery Rate**: 0.29%

    ## Noise Type Analysis
                         mean           std
noise_type                             
gaussian     40182.700000  47244.129781
salt_pepper  49127.533333  46936.547378

    ## Denoising Method Effectiveness
                          mean       std
denoise_method                      
bilateral_filter  0.387448  0.591096
median_filter     0.054218  2.319701
non_local_means   0.429006  0.927970

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 0.0000
    - **Average Wasserstein H1 (Clean->Noisy)**: 0.0000
    