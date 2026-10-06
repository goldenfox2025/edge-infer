#pragma once

#include <cstddef>

// Test-only counters. Production code has no allocation instrumentation.
namespace test_alloc {

struct Counts {
  std::size_t host_allocations = 0;
  std::size_t host_frees = 0;
  std::size_t device_allocations = 0;
  std::size_t device_frees = 0;
};

extern thread_local Counts counts;
extern thread_local bool active;

class Scope {
 public:
  Scope() noexcept { counts = {}; active = true; }
  ~Scope() { active = false; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Counts finish() noexcept { active = false; return counts; }
};

}  // namespace test_alloc
