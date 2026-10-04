#include <iostream>
#include <functional>
#include <limits>
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

void expect_rejected(const std::function<void()>& operation,
                     const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, message);
}

void expect_valid_live_ranges(const WorkspacePlan& plan) {
    const auto& allocations = plan.allocations();
    for (size_t index = 0; index < allocations.size(); ++index) {
        const auto& value = allocations[index];
        expect(value.offset % value.alignment == 0, "allocation alignment differs");
        expect(value.offset <= plan.total_bytes() &&
                   value.bytes <= plan.total_bytes() - value.offset,
               "allocation exceeds arena capacity");
        for (size_t other = index + 1; other < allocations.size(); ++other) {
            const auto& right = allocations[other];
            const bool live_overlap = value.first_use <= right.last_use &&
                                      right.first_use <= value.last_use;
            const bool byte_overlap = value.offset < right.offset + right.bytes &&
                                      right.offset < value.offset + value.bytes;
            expect(!(live_overlap && byte_overlap),
                   "concurrently live values overlap in arena storage");
        }
    }
}

void test_equal_step_boundaries_remain_live() {
    WorkspacePlanner planner;
    planner.add_request("input", 512, 0, 4);
    planner.add_request("output", 512, 4, 8);
    planner.add_request("later", 512, 9, 10);
    const auto plan = planner.build();
    expect_valid_live_ranges(plan);
    expect(plan.at("input").offset != plan.at("output").offset,
           "input and output used by the same step must stay separate");
    expect(plan.total_bytes() == 1024,
           "dead storage must be reused after the shared step completes");
}

void test_alignment_with_live_reuse() {
    WorkspacePlanner planner;
    planner.add_request("small", 300, 0, 0, 256);
    planner.add_request("wide", 512, 1, 1, 512);
    planner.add_request("live_small", 128, 2, 4, 256);
    planner.add_request("live_wide", 300, 2, 4, 512);
    const auto plan = planner.build();
    expect_valid_live_ranges(plan);
    expect(plan.total_bytes() < 300 + 512 + 128 + 300,
           "aligned dead slots should be reused despite mixed capacities");
}

void test_rejects_ambiguous_names_and_extent_overflow() {
    expect_rejected([] {
        WorkspacePlanner planner;
        planner.add_request("duplicate", 64, 0, 1);
        planner.add_request("duplicate", 128, 2, 3);
        (void)planner.build();
    }, "duplicate workspace names must be rejected");
    expect_rejected([] {
        WorkspacePlanner planner;
        planner.add_request("near_limit", std::numeric_limits<size_t>::max() - 511,
                            0, 1, 256);
        planner.add_request("overflow", 512, 0, 1, 256);
        (void)planner.build();
    }, "arena extent addition must reject size_t overflow");
    expect_rejected([] {
        WorkspacePlanner planner;
        planner.add_request("near_limit", std::numeric_limits<size_t>::max(),
                            0, 1, 1);
        planner.add_request("overflow", 1, 1, 2, 256);
        (void)planner.build();
    }, "arena alignment must reject size_t overflow");
}

}  // namespace

int main() {
    test_reuses_non_overlapping_buffers();
    test_keeps_overlapping_buffers_separate();
    test_respects_alignment();
    test_keeps_large_slot_for_later_reuse();
    test_equal_step_boundaries_remain_live();
    test_alignment_with_live_reuse();
    test_rejects_ambiguous_names_and_extent_overflow();

    std::cout << "decode_workspace_plan_test passed" << std::endl;
    return 0;
}
