# Diffusion-weighted images: Gibbs-ringing removal, MP-PCA denoising, the log-signal fits and
# the NODDI fit.
# Lists this folder's sources, relative to it; the folder's tests go in tests/dwi/CMakeLists.txt.
target_sources(larmor_numerics PRIVATE
    gibbs.cpp
    log_fit.cpp
    mppca.cpp
    noddi.cpp)
