#include <iostream>
#include <stdexcept>

#include "execution/workspace_plan.hpp"

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_reuses_non_overlapping_buffers() {
    WorkspacePlanner planner;
    planner.add_request("residual", 1024, 0, 2);
    planner.add_request("attn_proj", 1024, 3, 4);

    WorkspacePlan plan = planner.build();
    expect(plan.total_bytes() == 1024, "non-overlapping buffers should reuse the same slot");
    expect(plan.at("residual").offset == plan.at("attn_proj").offset, "expected slot reuse for non-overlapping buffers");
}

void test_keeps_overlapping_buffers_separate() {
    WorkspacePlanner planner;
    planner.add_request("q_buf", 1024, 0, 3);
    planner.add_request("k_buf", 1024, 2, 4);

    WorkspacePlan plan = planner.build();
    expect(plan.total_bytes() == 2048, "overlapping buffers should not share storage");
    expect(plan.at("q_buf").offset != plan.at("k_buf").offset, "overlapping buffers must have distinct offsets");
}

void test_respects_alignment() {
    WorkspacePlanner planner;
    planner.add_request("small", 300, 0, 0, 256);
    planner.add_request("aligned", 512, 1, 1, 512);

    WorkspacePlan plan = planner.build();
    expect(plan.at("small").offset % 256 == 0, "small allocation must be aligned");
    expect(plan.at("aligned").offset % 512 == 0, "aligned allocation must respect 512-byte alignment");
}

void test_keeps_large_slot_for_later_reuse() {
    WorkspacePlanner planner;
    planner.add_request("large_1", 1024, 0, 0);
    planner.add_request("small_mid", 256, 1, 1);
    planner.add_request("large_2", 1024, 2, 2);

    WorkspacePlan plan = planner.build();
    expect(plan.total_bytes() == 1024,
           "temporary small value should not permanently fragment a larger reusable slot");
    expect(plan.at("large_1").offset == plan.at("small_mid").offset,
           "small value should reuse the existing large slot");
    expect(plan.at("large_1").offset == plan.at("large_2").offset,
           "later large value should reclaim the original large slot");
}

}  // namespace

int main() {
    test_reuses_non_overlapping_buffers();
    test_keeps_overlapping_buffers_separate();
    test_respects_alignment();
    test_keeps_large_slot_for_later_reuse();

    std::cout << "decode_workspace_plan_test passed" << std::endl;
    return 0;
}
