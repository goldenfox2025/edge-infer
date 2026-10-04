#include <iostream>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

#include "execution/workspace_liveness.hpp"

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_infers_lifetimes_from_steps() {
    WorkspaceLivenessBuilder builder;
    builder.add_value("a", {1, 256});
    builder.add_value("b", {1, 256});
    builder.add_value("c", {1, 256});

    builder.add_step({}, {"a"});
    builder.add_step({"a"}, {"b"});
    builder.add_step({"b"}, {"c"});

    WorkspacePlan plan = builder.build_plan<float>();
    expect(plan.total_bytes() == sizeof(float) * 256 * 2, "expected reuse after value dies");
    expect(plan.at("a").offset == plan.at("c").offset, "a and c should reuse the same slot");
    expect(plan.at("a").offset != plan.at("b").offset, "b overlaps with a's last use and must stay separate");
}

void test_decoder_trace_reuses_storage_without_corrupting_live_values() {
    WorkspaceLivenessBuilder builder;
    const std::vector<std::string> hidden_values{
        "residual", "hidden_attn", "q", "k", "v", "attention",
        "attention_projected", "residual_mid", "hidden_ffn", "ffn_out",
        "final_residual", "final_h"};
    for (const auto& name : hidden_values) builder.add_value(name, {4, 128});
    builder.add_value("gate", {4, 256});
    builder.add_value("up", {4, 256});
    builder.add_value("logits", {4, 64});
    const std::vector<WorkspaceStep> steps{
        {{}, {"residual"}},
        {{"residual"}, {"hidden_attn"}},
        {{"hidden_attn"}, {"q"}},
        {{"hidden_attn"}, {"k"}},
        {{"hidden_attn"}, {"v"}},
        {{"k", "v"}, {}},
        {{"q", "k", "v"}, {"attention"}},
        {{"attention"}, {"attention_projected"}},
        {{"residual", "attention_projected"}, {"residual_mid"}},
        {{"residual_mid"}, {"hidden_ffn"}},
        {{"hidden_ffn"}, {"gate"}},
        {{"hidden_ffn"}, {"up"}},
        {{"gate", "up"}, {"gate"}},
        {{"gate"}, {"ffn_out"}},
        {{"residual_mid", "ffn_out"}, {"final_residual"}},
        {{"final_residual"}, {"final_h"}},
        {{"final_h"}, {"logits"}},
        {{"logits"}, {}}};
    for (const auto& step : steps) builder.add_step(step.inputs, step.outputs);
    const auto plan = builder.build_plan<float>();
    const size_t independent_bytes = hidden_values.size() * 4 * 128 * sizeof(float) +
                                     2 * 4 * 256 * sizeof(float) + 4 * 64 * sizeof(float);
    expect(plan.total_bytes() < independent_bytes,
           "decoder trace must reclaim temporary activation storage");
    expect(plan.requested_bytes() == independent_bytes &&
               plan.unaliased_bytes() >= plan.requested_bytes() &&
               plan.reused_bytes() == plan.unaliased_bytes() - plan.total_bytes() &&
               plan.reused_bytes() > 0 && plan.slot_count() < plan.allocations().size(),
           "workspace metrics must describe actual reclaimed storage");
    const auto& allocations = plan.allocations();
    for (size_t index = 0; index < allocations.size(); ++index) {
        const auto& left = allocations[index];
        expect(left.offset % left.alignment == 0 &&
                   left.offset <= plan.total_bytes() &&
                   left.bytes <= plan.total_bytes() - left.offset,
               "decoder allocation extent or alignment is invalid");
        for (size_t other = index + 1; other < allocations.size(); ++other) {
            const auto& right = allocations[other];
            const bool live_overlap = left.first_use <= right.last_use &&
                                      right.first_use <= left.last_use;
            const bool byte_overlap = left.offset < right.offset + right.bytes &&
                                      right.offset < left.offset + left.bytes;
            expect(!(live_overlap && byte_overlap),
                   "simultaneously live decoder values share bytes");
        }
    }
    // Execute the planned lifetimes with byte patterns. Reading every input
    // before each write catches aliasing that comparing offsets alone can hide.
    std::vector<uint8_t> arena(plan.total_bytes());
    std::unordered_map<std::string, uint8_t> patterns;
    for (size_t step = 0; step < steps.size(); ++step) {
        for (const auto& input : steps[step].inputs) {
            const auto& allocation = plan.at(input);
            const auto pattern = patterns.at(input);
            expect(std::all_of(arena.begin() + allocation.offset,
                               arena.begin() + allocation.offset + allocation.bytes,
                               [&](uint8_t value) { return value == pattern; }),
                   "a live decoder input was overwritten by another value");
        }
        for (const auto& output : steps[step].outputs) {
            const auto& allocation = plan.at(output);
            const auto pattern = static_cast<uint8_t>((step + 1) * 7);
            std::fill_n(arena.begin() + allocation.offset, allocation.bytes, pattern);
            patterns[output] = pattern;
        }
    }
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

void test_rejects_unknown_unused_zero_and_overflow_values() {
    expect_rejected([] {
        WorkspaceLivenessBuilder builder;
        builder.add_value("known", {1});
        builder.add_step({"unknown"}, {"known"});
        (void)builder.build_plan<float>();
    }, "unknown value references must be rejected");
    expect_rejected([] {
        WorkspaceLivenessBuilder builder;
        builder.add_value("unused", {1});
        (void)builder.build_plan<float>();
    }, "declared values without execution uses must be rejected");
    expect_rejected([] {
        WorkspaceLivenessBuilder builder;
        builder.add_value("zero", {1, 0});
        builder.add_step({}, {"zero"});
        (void)builder.build_plan<float>();
    }, "zero-sized workspace dimensions must be rejected");
    expect_rejected([] {
        WorkspaceLivenessBuilder builder;
        builder.add_value("overflow", {std::numeric_limits<size_t>::max(), 2});
        builder.add_step({}, {"overflow"});
        (void)builder.build_plan<float>();
    }, "workspace shape byte multiplication must reject size_t overflow");
}

}  // namespace

int main() {
    test_infers_lifetimes_from_steps();
    test_decoder_trace_reuses_storage_without_corrupting_live_values();
    test_rejects_unknown_unused_zero_and_overflow_values();
    std::cout << "workspace_liveness_test passed" << std::endl;
    return 0;
}
