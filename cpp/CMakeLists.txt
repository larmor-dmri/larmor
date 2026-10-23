# larmor's compiled kernels in C++: a numerical library without Python (larmor_numerics), the
# Python module that binds it (nanobind), and the unit tests.
cmake_minimum_required(VERSION 3.28)

# On macOS the SDK goes into every compile command (-isysroot), so that clang-tidy, which is not
# Apple's compiler, finds the system headers through compile_commands.json.
if(CMAKE_HOST_APPLE AND NOT CMAKE_OSX_SYSROOT)
    execute_process(COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE larmor_sdk_path OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_OSX_SYSROOT "${larmor_sdk_path}" CACHE PATH "macOS SDK")
endif()

project(larmor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # clang-tidy reads compile_commands.json

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------------------------------
# Options

set(LARMOR_MODULE_NAME "_core" CACHE STRING "Name of the Python module, imported as larmor.<name>")
option(LARMOR_BUILD_MODULE "Build the Python module (needs Python and nanobind)" ON)
option(LARMOR_BUILD_TESTS "Build the doctest unit tests" ON)
option(LARMOR_WARNINGS_AS_ERRORS "Turn compiler warnings into errors (ON in CI)" OFF)
option(LARMOR_LTO "Link-time optimisation in Release builds" ON)
option(LARMOR_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer (Clang or GCC)" OFF)
set(LARMOR_BLAS "auto" CACHE STRING
    "BLAS for Eigen's matrix products: auto (Accelerate on macOS, none elsewhere), accelerate or none")
set_property(CACHE LARMOR_BLAS PROPERTY STRINGS auto accelerate none)

# ---------------------------------------------------------------------------------------------------
# Compiler flags, shared by every Larmor target through larmor_options. Third-party code does not link it.

# Link-time optimisation in Release only: it makes Debug and sanitizer builds slow to link. CI's
# test builds turn it off (LARMOR_LTO=OFF), as they are built once and run once.
include(CheckIPOSupported)
check_ipo_supported(RESULT larmor_ipo_supported OUTPUT larmor_ipo_message LANGUAGES CXX)
if(larmor_ipo_supported AND LARMOR_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# Release optimises with CMake's defaults, -O3 (GCC, Clang) and /O2 (MSVC).
add_library(larmor_options INTERFACE)
if(MSVC)
    # /fp:precise keeps the order of floating-point operations as written and does not contract
    # a * b + c into a fused multiply-add.
    target_compile_options(larmor_options INTERFACE /W4 /permissive- /fp:precise)
    if(LARMOR_WARNINGS_AS_ERRORS)
        target_compile_options(larmor_options INTERFACE /WX)
    endif()
else()
    # -ffp-contract=off: Clang on arm64 and GCC fuse a * b + c into one fused multiply-add by
    # default, which rounds once instead of twice. With fusing off, scalar code gives the bits of
    # the expression as written, in its term order, whatever the compiler. It does not reach
    # Eigen's vectorised kernels or BLAS, which use fused multiply-adds and their own order.
    # No -ffast-math and no -march=native: a wheel must give IEEE results and run on every
    # processor of its architecture.
    target_compile_options(larmor_options INTERFACE
        -ffp-contract=off
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow)
    if(LARMOR_WARNINGS_AS_ERRORS)
        target_compile_options(larmor_options INTERFACE -Werror)
    endif()
endif()

if(LARMOR_SANITIZE)
    if(MSVC)
        message(FATAL_ERROR "LARMOR_SANITIZE needs Clang or GCC")
    endif()
    # float-cast-overflow catches a double converted to an integer it does not fit in.
    set(larmor_sanitize_flags
        -fsanitize=address,undefined,float-cast-overflow
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer)
    target_compile_options(larmor_options INTERFACE ${larmor_sanitize_flags})
    target_link_options(larmor_options INTERFACE ${larmor_sanitize_flags})
endif()

# ---------------------------------------------------------------------------------------------------
# Dependencies. Eigen, pocketfft and doctest are header-only and pinned by the hash of their
# archive; only their headers are used, so their own CMake projects are not configured
# (SOURCE_SUBDIR points to a folder that does not exist). Their headers are SYSTEM includes: their
# warnings do not count. A source distribution carries Eigen and pocketfft in third_party/ (see
# cmake/vendor.cmake); a build from it uses those and downloads nothing.

include(FetchContent)
include(cmake/third_party.cmake)

foreach(dependency IN ITEMS eigen pocketfft)
    string(TOUPPER ${dependency} upper)
    set(vendored ${CMAKE_CURRENT_SOURCE_DIR}/third_party/${LARMOR_${upper}_NAME})
    if(IS_DIRECTORY ${vendored})
        set(${dependency}_SOURCE_DIR ${vendored})
    else()
        FetchContent_Declare(${dependency}
            URL ${LARMOR_${upper}_URL}
            URL_HASH SHA256=${LARMOR_${upper}_SHA256}
            SOURCE_SUBDIR do-not-configure
            DOWNLOAD_EXTRACT_TIMESTAMP ON)
        FetchContent_MakeAvailable(${dependency})
    endif()
    message(STATUS "${LARMOR_${upper}_NAME}: ${${dependency}_SOURCE_DIR}")
endforeach()

add_library(larmor_eigen INTERFACE)
target_include_directories(larmor_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

# pocketfft (BSD 3-clause): FFTs of any length, for Gibbs-ringing removal. Larmor's own threads run
# it, so its thread pool is left out.
add_library(larmor_pocketfft INTERFACE)
target_include_directories(larmor_pocketfft SYSTEM INTERFACE ${pocketfft_SOURCE_DIR})
target_compile_definitions(larmor_pocketfft INTERFACE POCKETFFT_NO_MULTITHREADING)

include(cmake/blas.cmake)

if(LARMOR_BUILD_MODULE)
    # nanobind comes from the Python package in the uv environment, as nanobind's documentation
    # recommends, so its version is the one in uv.lock.
    find_package(Python 3.12 REQUIRED
        COMPONENTS Interpreter Development.Module Development.SABIModule)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT
        COMMAND_ERROR_IS_FATAL ANY)
    find_package(nanobind CONFIG REQUIRED)
endif()

# ---------------------------------------------------------------------------------------------------
# Targets

# The numerical library: plain C++ and Eigen, no Python. Each folder appends its own sources.
add_library(larmor_numerics STATIC)
target_include_directories(larmor_numerics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(larmor_numerics PUBLIC larmor_eigen PRIVATE larmor_options)
target_link_libraries(larmor_numerics PRIVATE larmor_pocketfft)
target_compile_features(larmor_numerics PUBLIC cxx_std_20)
set_target_properties(larmor_numerics PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src/common)
add_subdirectory(src/sh)
add_subdirectory(src/sd)
add_subdirectory(src/dwi)
add_subdirectory(src/tracking)
add_subdirectory(src/streamlines)
add_subdirectory(src/simulation)

if(LARMOR_BUILD_MODULE)
    add_subdirectory(src/python)
endif()

if(LARMOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
