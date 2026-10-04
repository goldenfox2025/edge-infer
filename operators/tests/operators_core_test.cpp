#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "operators/core/cpu_reference.hpp"

static_assert(std::is_trivially_copyable_v<op::ArrayView<float>>);
static_assert(!std::is_polymorphic_v<op::ArrayView<float>>);

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(float actual, float expected, const char* message) {
    expect(std::isfinite(actual) && std::fabs(actual - expected) < 1e-5f, message);
}

void test_elementwise_aliasing_and_empty_views() {
    float a[]{1.0f, -2.0f, 3.0f};
    const float b[]{4.0f, 2.0f, -1.0f};
    op::cpu::add<float>({a, 3}, {b, 3}, {a, 3});
    expect(a[0] == 5.0f && a[1] == 0.0f && a[2] == 2.0f, "in-place add");
    op::cpu::multiply<float>({a, 3}, {b, 3}, {a, 3});
    expect(a[0] == 20.0f && a[1] == 0.0f && a[2] == -2.0f, "in-place multiply");
    op::cpu::add<float>({}, {}, {});
    op::cpu::multiply<float>({}, {}, {});
    op::cpu::silu<float>({}, {});
}

void test_silu_in_place() {
    float values[]{0.0f, 1.0f, -1.0f};
    op::cpu::silu<float>({values, 3}, {values, 3});
    expect_near(values[0], 0.0f, "SiLU at zero");
    expect_near(values[1], 0.7310586f, "positive SiLU");
    expect_near(values[2], -0.2689414f, "negative SiLU");
}

void test_rms_norm_rows_in_place() {
    float values[]{1.0f, 3.0f, 0.0f, 0.0f};
    const float weights[]{2.0f, 0.5f};
    op::cpu::rms_norm<float>({values, 4}, {weights, 2}, {values, 4}, 2, 2, 1e-6f);
    expect_near(values[0], 0.8944271f, "first feature scale");
    expect_near(values[1], 0.6708203f, "second feature scale");
    expect(values[2] == 0.0f && values[3] == 0.0f, "zero row remains finite");
}

}  // namespace

int main() {
    test_elementwise_aliasing_and_empty_views();
    test_silu_in_place();
    test_rms_norm_rows_in_place();
    std::cout << "operators_core_test passed\n";
}
