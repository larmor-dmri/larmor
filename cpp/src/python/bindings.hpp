// The parts of the Python module, each registering its functions and classes.
#pragma once

#include <nanobind/nanobind.h>

namespace larmor::python {

/// Registers the shared helpers, such as norms.
void register_common(nanobind::module_& module);

/// Registers the spherical-harmonics and peak functions.
void register_sh(nanobind::module_& module);

/// Registers the spherical-deconvolution functions and classes.
void register_sd(nanobind::module_& module);

/// Registers the diffusion-weighted signal fits.
void register_dwi(nanobind::module_& module);

/// Registers the tracking functions and classes.
void register_tracking(nanobind::module_& module);

/// Registers the streamline functions, such as resampling.
void register_streamlines(nanobind::module_& module);

/// Registers the simulation engines and their compartments.
void register_simulation(nanobind::module_& module);

} // namespace larmor::python
