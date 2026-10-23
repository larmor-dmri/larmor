# The Python module, the only code that uses nanobind and Python, with one file per part.
nanobind_add_module(${LARMOR_MODULE_NAME}
    STABLE_ABI
    NB_STATIC
    NB_SUPPRESS_WARNINGS # nanobind's headers are SYSTEM includes: their warnings do not count
    NOMINSIZE # optimise the bindings like the rest, not for size
    module.cpp
    sh.cpp
    sd.cpp
    dwi.cpp
    tracking.cpp
    streamlines.cpp
    simulation.cpp)
target_link_libraries(${LARMOR_MODULE_NAME} PRIVATE larmor_numerics larmor_options)
target_compile_definitions(${LARMOR_MODULE_NAME} PRIVATE LARMOR_MODULE_NAME=${LARMOR_MODULE_NAME})

# The wheel build (scikit-build-core) installs the module into the larmor package.
install(TARGETS ${LARMOR_MODULE_NAME} LIBRARY DESTINATION larmor)
