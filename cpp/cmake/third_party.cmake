# The header-only C++ libraries the module is built with, pinned by the hash of their archive.
# CMakeLists.txt downloads them at configure time; vendor.cmake puts them into cpp/third_party for
# the source distribution, so that a build from it needs no network.

set(LARMOR_EIGEN_NAME eigen-5.0.1)
set(LARMOR_EIGEN_URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz)
set(LARMOR_EIGEN_SHA256 e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec)

# pocketfft tags no releases, so a commit of its cpp branch is pinned.
set(LARMOR_POCKETFFT_NAME pocketfft-c90e55b3d529f8efa40ed01a20de22405f45fc65)
set(LARMOR_POCKETFFT_URL
    https://github.com/mreineck/pocketfft/archive/c90e55b3d529f8efa40ed01a20de22405f45fc65.tar.gz)
set(LARMOR_POCKETFFT_SHA256 2d90016da31a8cc1dd0ac0c89c1f2fb55df8f4ccc3f0386dcfbad09ac6d1c3ba)
