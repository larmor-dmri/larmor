// The Python module, which registers each part's bindings and the shared ones defined here.

#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "common/maths.hpp"
#include "common/parallel.hpp"
#include "common/types.hpp"
#include "python/arrays.hpp"
#include "python/bindings.hpp"

namespace nb = nanobind;

namespace larmor::python {

namespace {

// Rows per block of the parallel loop.
constexpr Index NORMS_BLOCK_SIZE = 4096;

// Returns the norm of each row of an (n, 3) array.
NumpyArray<double, 1> norms(const InArray<double, 2>& vectors_array, int n_threads) {
    const auto vectors = matrix_view(vectors_array); // checks contiguity
    require_columns(vectors_array, 3);
    require_threads(n_threads);

    Vector result(vectors.rows());
    {
        const nb::gil_scoped_release release;
        parallel_for_blocks(vectors.rows(), NORMS_BLOCK_SIZE, n_threads,
                            [&](Index begin, Index end) {
                                for (Index row = begin; row < end; ++row) {
                                    result(row) = norm3(vectors.row(row).transpose());
                                }
                            });
    }
    return to_numpy(std::move(result));
}

} // namespace

void register_common(nb::module_& module) {
    module.def("norms", &norms, nb::arg("v").noconvert(), nb::arg("n_threads"),
               "The norm of each row of a C-contiguous (n, 3) float64 array.\n"
               "\n"
               "Raises:\n"
               "    ValueError: If the array is not C-contiguous, does not have 3 columns, or\n"
               "        n_threads is 0.\n");
}

} // namespace larmor::python

// LARMOR_MODULE_NAME is the module's name, set by the build (_core).
NB_MODULE(LARMOR_MODULE_NAME, module) {
    module.doc() = "The compiled numerical code of Larmor.\n"
                   "\n"
                   "Private: the public API in larmor.sd, larmor.dwi, larmor.tractography\n"
                   "and larmor.simulation checks the inputs and calls these functions.\n";
    larmor::python::register_common(module);
    larmor::python::register_sh(module);
    larmor::python::register_sd(module);
    larmor::python::register_dwi(module);
    larmor::python::register_tracking(module);
    larmor::python::register_streamlines(module);
    larmor::python::register_simulation(module);
}
