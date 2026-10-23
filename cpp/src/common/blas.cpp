#include "common/blas.hpp"

// The build defines LARMOR_BLAS_ACCELERATE when it links Accelerate. BLASSetThreading exists in the
// macOS 15 SDK and later.
#if defined(LARMOR_BLAS_ACCELERATE) && __has_include(<vecLib/thread_api.h>)
#include <vecLib/thread_api.h>
#define LARMOR_HAS_BLAS_SET_THREADING
#endif

namespace larmor {

SingleThreadedBlas::SingleThreadedBlas() {
#ifdef LARMOR_HAS_BLAS_SET_THREADING
    // The SDK may be newer than the macOS the module runs on.
    if (__builtin_available(macOS 15.0, *)) {
        previous_threading = BLASGetThreading();
        restore = BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0;
    }
#endif
}

SingleThreadedBlas::~SingleThreadedBlas() {
#ifdef LARMOR_HAS_BLAS_SET_THREADING
    if (__builtin_available(macOS 15.0, *)) {
        if (restore) {
            BLASSetThreading(static_cast<BLAS_THREADING>(previous_threading));
        }
    }
#endif
}

} // namespace larmor
