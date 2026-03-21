#include <iostream>
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

}  // namespace

int main() {
    test_infers_lifetimes_from_steps();
    std::cout << "workspace_liveness_test passed" << std::endl;
    return 0;
}
