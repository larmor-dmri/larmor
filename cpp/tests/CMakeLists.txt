# The unit tests of the numerical library: one doctest executable, run by ctest. It links only
# larmor_numerics, so the tests run without Python. Each part lists its own test files in its folder.
FetchContent_Declare(doctest
    URL https://github.com/doctest/doctest/archive/refs/tags/v2.5.3.tar.gz
    URL_HASH SHA256=174ebc4e769928959614789c5b4e9c3d0a0f81a62bb608756b127bfebfb21331
    SOURCE_SUBDIR do-not-configure
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(doctest)

add_executable(larmor_tests
    test_main.cpp
    test_blas.cpp
    test_maths.cpp
    test_parallel.cpp)
target_include_directories(larmor_tests SYSTEM PRIVATE ${doctest_SOURCE_DIR})
target_link_libraries(larmor_tests PRIVATE larmor_numerics larmor_options)

add_subdirectory(sh)
add_subdirectory(sd)
add_subdirectory(dwi)
add_subdirectory(tracking)
add_subdirectory(streamlines)
add_subdirectory(simulation)

add_test(NAME larmor_tests COMMAND larmor_tests)
