// Threading settings of the BLAS library that Eigen's matrix products call.
#pragma once

namespace larmor {

/// Keeps BLAS calls on the current thread single-threaded while it exists.
///
/// It restores the thread's previous setting when destroyed. On macOS before 15, or without
/// Accelerate, it does nothing.
class SingleThreadedBlas {
public:
    SingleThreadedBlas();
    ~SingleThreadedBlas();
    SingleThreadedBlas(const SingleThreadedBlas&) = delete;
    SingleThreadedBlas& operator=(const SingleThreadedBlas&) = delete;
    SingleThreadedBlas(SingleThreadedBlas&&) = delete;
    SingleThreadedBlas& operator=(SingleThreadedBlas&&) = delete;

private:
    // Unused when the build has no Accelerate to set.
    [[maybe_unused]] unsigned int previous_threading = 0;
    [[maybe_unused]] bool restore = false;
};

} // namespace larmor
