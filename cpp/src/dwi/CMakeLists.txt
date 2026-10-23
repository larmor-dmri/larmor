# Diffusion-weighted images: Gibbs-ringing removal, MP-PCA denoising and the log-signal fits.
# Lists this folder's sources, relative to it; the folder's tests go in tests/dwi/CMakeLists.txt.
target_sources(larmor_numerics PRIVATE
    gibbs.cpp
    log_fit.cpp
    mppca.cpp)
