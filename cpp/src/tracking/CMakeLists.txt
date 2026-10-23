# Streamline tracking: the step loop and the tracking algorithms.
# Lists this folder's sources, relative to it; the folder's tests go in tests/tracking/CMakeLists.txt.
target_sources(larmor_numerics PRIVATE
    algorithm.cpp
    calibrate.cpp
    det.cpp
    grid.cpp
    prob.cpp
    rng.cpp
    testing.cpp
    tracker.cpp)
