# This folder holds the spherical deconvolution code, with the FOD fits and their solvers.
# Lists this folder's sources, relative to it; the folder's tests go in tests/sd/CMakeLists.txt.
target_sources(larmor_numerics PRIVATE
    csd.cpp
    drl.cpp
    elastic_net.cpp
    fractions.cpp
    lsi.cpp
    multi_tissue.cpp
    nnls.cpp
    solver.cpp)
