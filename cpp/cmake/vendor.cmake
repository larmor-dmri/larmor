# Downloads Eigen and pocketfft into cpp/third_party, checking their hashes, before a source
# distribution is built: `cmake -P cpp/cmake/vendor.cmake`. Only the headers and the licence files
# are kept. CMakeLists.txt then uses these folders instead of downloading.

include(${CMAKE_CURRENT_LIST_DIR}/third_party.cmake)
set(third_party ${CMAKE_CURRENT_LIST_DIR}/../third_party)
set(downloads ${third_party}/downloads)

function(vendor name url sha256)
    set(archive ${downloads}/${name}.tar.gz)
    file(DOWNLOAD ${url} ${archive} EXPECTED_HASH SHA256=${sha256} STATUS status)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "Downloading ${url} failed: ${status}")
    endif()
    file(REMOVE_RECURSE ${third_party}/${name})
    file(ARCHIVE_EXTRACT INPUT ${archive} DESTINATION ${third_party} PATTERNS ${ARGN})
    message(STATUS "Vendored ${name}")
endfunction()

vendor(${LARMOR_EIGEN_NAME} ${LARMOR_EIGEN_URL} ${LARMOR_EIGEN_SHA256}
    ${LARMOR_EIGEN_NAME}/Eigen ${LARMOR_EIGEN_NAME}/COPYING.*)
vendor(${LARMOR_POCKETFFT_NAME} ${LARMOR_POCKETFFT_URL} ${LARMOR_POCKETFFT_SHA256}
    ${LARMOR_POCKETFFT_NAME}/pocketfft_hdronly.h ${LARMOR_POCKETFFT_NAME}/LICENSE.md)
file(REMOVE_RECURSE ${downloads})
