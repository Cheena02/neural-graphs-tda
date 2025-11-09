# TDA Pipeline Analysis Summary

    ## Configuration
    - **EDT Filtration**: True
    - **Compare Filtrations**: False
    - **Random Seed**: 42
    - **Datasets Processed**: test

    ## Results Overview
    - **Total Images Analyzed**: 1
    - **Total Comparisons**: 50
    - **Average Noise Impact**: -685.46 features
    - **Average Recovery Rate**: -25.70%

    ## Noise Type Analysis
                    mean         std
noise_type                      
gaussian      736.52  830.060144
salt_pepper -2107.44  540.384900

    ## Denoising Method Effectiveness
                                  mean         std
denoise_method                                
bilateral_filter         -8.397481   22.331225
median_filter           -10.203429   34.414139
morphological_denoising -13.514674   32.572494
non_local_means         -91.823097  271.572116
topological_denoising    -4.583272   10.563164

    ## Distance Analysis
    - **Average Wasserstein H0 (Clean->Noisy)**: 206.9024
    - **Average Wasserstein H1 (Clean->Noisy)**: 2033.2929
    