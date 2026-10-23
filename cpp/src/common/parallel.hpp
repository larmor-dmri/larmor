// Parallel loops over blocks of items on a fixed number of threads, started for one call.
#pragma once

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/blas.hpp"
#include "common/types.hpp"

namespace larmor {

/// Runs body(begin, end, scratch) over the items 0 to n_items - 1, in blocks of block_size
/// items, on n_threads threads.
///
/// Each thread takes the next free block until none is left, and calls make_scratch() once for
/// the buffers it passes to every block it runs. The calling thread is one of the n_threads, and
/// BLAS calls inside a body stay on their thread. A body that writes only the outputs of its own
/// items gives the same result for any number of threads. If a body or make_scratch throws, the
/// first exception is rethrown on the calling thread once all threads have finished.
template <typename MakeScratch, typename Body>
void parallel_for_blocks(Index n_items, Index block_size, int n_threads, MakeScratch make_scratch,
                         Body body) {
    if (n_items < 0 || block_size < 1 || n_threads < 1) {
        throw std::invalid_argument(
            "parallel_for_blocks needs n_items >= 0, block_size >= 1 and n_threads >= 1");
    }
    const Index n_blocks = (n_items + block_size - 1) / block_size;
    if (n_blocks == 0) {
        return;
    }

    std::atomic<Index> next_block{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    const auto record_error = [&] {
        const std::scoped_lock lock(error_mutex);
        if (!first_error) {
            first_error = std::current_exception();
        }
        failed.store(true);
    };

    const auto run_blocks = [&] {
        try {
            const SingleThreadedBlas single_threaded_blas;
            auto scratch = make_scratch();
            while (!failed.load(std::memory_order_relaxed)) {
                const Index block = next_block.fetch_add(1);
                if (block >= n_blocks) {
                    break;
                }
                const Index begin = block * block_size;
                const Index end = std::min(begin + block_size, n_items);
                body(begin, end, scratch);
            }
        } catch (...) {
            record_error();
        }
    };

    const Index n_workers = std::min<Index>(n_threads, n_blocks);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::vector<std::thread>::size_type>(n_workers - 1));
    try {
        for (Index worker = 1; worker < n_workers; ++worker) {
            threads.emplace_back(run_blocks);
        }
    } catch (...) {
        // A thread that cannot start is reported like a failing body, after the others finish.
        record_error();
    }
    run_blocks();
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

/// Runs body(begin, end) over the items 0 to n_items - 1, in blocks of block_size items, on
/// n_threads threads, for bodies that need no per-thread buffers.
template <typename Body>
void parallel_for_blocks(Index n_items, Index block_size, int n_threads, Body body) {
    struct NoScratch {};
    parallel_for_blocks(
        n_items, block_size, n_threads, [] { return NoScratch{}; },
        [&body](Index begin, Index end, NoScratch& /*scratch*/) { body(begin, end); });
}

} // namespace larmor
