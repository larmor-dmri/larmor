# The simulation engines and the compartments they share.
# Lists this folder's sources, relative to it; the folder's tests go in tests/simulation/CMakeLists.txt.
target_sources(larmor_numerics PRIVATE
    compartments/compartments.cpp
    instant/instant.cpp)
