#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include <doctest/doctest.h>

#include "common/parallel.hpp"
#include "common/types.hpp"

namespace {

using larmor::Index;

// A value per item that takes a few roundings, so a change of order would show.
std::vector<double> item_values(Index n_items, Index block_size, int n_threads) {
    std::vector<double> values(static_cast<std::vector<double>::size_type>(n_items));
    larmor::parallel_for_blocks(n_items, block_size, n_threads, [&](Index begin, Index end) {
        for (Index item = begin; item < end; ++item) {
            double sum = 0.0;
            for (Index term = 1; term <= 50; ++term) {
                sum += std::sin(static_cast<double>(item * term)) / static_cast<double>(term);
            }
            values.at(static_cast<std::vector<double>::size_type>(item)) = sum;
        }
    });
    return values;
}

} // namespace

TEST_CASE("parallel_for_blocks visits every item exactly once") {
    for (const int n_threads : {1, 2, 7}) {
        const Index n_items = 1003;
        std::vector<int> n_visits(n_items, 0);
        larmor::parallel_for_blocks(n_items, 10, n_threads, [&](Index begin, Index end) {
            CHECK(begin % 10 == 0);
            CHECK(end - begin <= 10);
            for (Index item = begin; item < end; ++item) {
                ++n_visits.at(static_cast<std::vector<int>::size_type>(item));
            }
        });
        CHECK(n_visits == std::vector<int>(n_items, 1));
    }
}

TEST_CASE("parallel_for_blocks gives the same output for any number of threads") {
    const std::vector<double> one_thread = item_values(5000, 64, 1);
    CHECK(item_values(5000, 64, 2) == one_thread);
    CHECK(item_values(5000, 64, 7) == one_thread);
}

TEST_CASE("parallel_for_blocks makes one scratch per thread and reuses it") {
    std::atomic<int> n_scratches{0};
    std::atomic<int> n_uses{0};
    larmor::parallel_for_blocks(
        1000, 10, 4,
        [&] {
            ++n_scratches;
            return std::vector<double>(8, 0.0);
        },
        [&](Index /*begin*/, Index /*end*/, std::vector<double>& scratch) {
            CHECK(scratch.size() == 8);
            ++n_uses;
        });
    CHECK(n_scratches.load() >= 1);
    CHECK(n_scratches.load() <= 4);
    CHECK(n_uses.load() == 100);
}

TEST_CASE("parallel_for_blocks with one thread runs on the calling thread") {
    const std::thread::id caller = std::this_thread::get_id();
    bool on_caller = true;
    larmor::parallel_for_blocks(100, 7, 1, [&](Index /*begin*/, Index /*end*/) {
        on_caller = on_caller && std::this_thread::get_id() == caller;
    });
    CHECK(on_caller);
}

TEST_CASE("parallel_for_blocks rethrows the first exception on the caller") {
    for (const int n_threads : {1, 2, 7}) {
        CHECK_THROWS_WITH_AS(larmor::parallel_for_blocks(1000, 10, n_threads,
                                                         [](Index begin, Index /*end*/) {
                                                             if (begin == 500) {
                                                                 throw std::runtime_error(
                                                                     "block 50");
                                                             }
                                                         }),
                             "block 50", std::runtime_error);
    }
}

TEST_CASE("parallel_for_blocks with no items calls nothing") {
    bool called = false;
    larmor::parallel_for_blocks(0, 10, 4, [&](Index /*begin*/, Index /*end*/) { called = true; });
    CHECK_FALSE(called);
}
