# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: True
    - **Compare Filtrations**: True
    - **Random Seed**: 42
    - **Datasets Processed**: MOUSEBIRN

    ## Results Overview
    - **Total Images Analyzed**: 5
    - **Total Comparisons**: 60
    - **Average Noise Impact**: 3840.30 features
    - **Average Recovery Rate**: -47.21%

    ## Noise Type Analysis
                        mean          std
noise_type                           
gaussian       37.033333    51.888131
salt_pepper  7643.566667  6625.332110

    ## Denoising Method Effectiveness
                           mean         std
denoise_method                         
bilateral_filter -34.882596   47.135927
median_filter    -35.868847   53.222704
non_local_means  -70.889145  104.137925

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: nan
    - **Average Wasserstein H1 (Clean->Noisy)**: nan
    