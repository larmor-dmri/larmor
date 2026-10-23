# Chooses the BLAS that Eigen sends its matrix products to (EIGEN_USE_BLAS) and links it to
# larmor_eigen, so every target that includes Eigen sees the same definitions and the same library.
#
#   accelerate  Apple's Accelerate framework (the default on macOS): a system framework, nothing to
#               bundle. It starts its own threads unless told otherwise, per thread;
#               larmor::SingleThreadedBlas does so on the threads of parallel_for_blocks.
#   none        Eigen's own products (the default elsewhere).

if(LARMOR_BLAS STREQUAL "auto")
    if(APPLE)
        set(LARMOR_BLAS_USED accelerate)
    else()
        set(LARMOR_BLAS_USED none)
    endif()
else()
    set(LARMOR_BLAS_USED ${LARMOR_BLAS})
endif()

if(LARMOR_BLAS_USED STREQUAL "accelerate")
    if(NOT APPLE)
        message(FATAL_ERROR "LARMOR_BLAS=accelerate exists on macOS only")
    endif()
    target_compile_definitions(larmor_eigen INTERFACE EIGEN_USE_BLAS LARMOR_BLAS_ACCELERATE)
    target_link_libraries(larmor_eigen INTERFACE "-framework Accelerate")
elseif(NOT LARMOR_BLAS_USED STREQUAL "none")
    message(FATAL_ERROR "Unknown LARMOR_BLAS '${LARMOR_BLAS}': choose auto, accelerate or none")
endif()

message(STATUS "BLAS for Eigen: ${LARMOR_BLAS_USED}")
