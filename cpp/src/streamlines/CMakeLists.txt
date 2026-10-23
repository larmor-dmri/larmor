# Operations on streamlines once tracked: resampling and track density maps.
# Lists this folder's sources, relative to it; the folder's tests go in tests/streamlines/CMakeLists.txt.
target_sources(larmor_numerics PRIVATE
    density.cpp
    resample.cpp)
