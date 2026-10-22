#include <arithmetic/addition/addition.hpp>
#include <arithmetic/subtraction/subtraction.hpp>

#include <cassert>
#include <stdexcept>

int main() {
    using tsr::Tensor;

    Tensor<int> a({2, 2}, 3);
    Tensor<int> b({2, 2}, 2);

    // Safe addition returns a new tensor with element-wise sum.
    Tensor<int> add_result = tsr::add(a, b);
    assert(add_result.shape() == a.shape());
    for (size_t i = 0; i < add_result.size(); ++i) {
        assert(add_result.data()[i] == 5);
    }

    // Safe subtraction returns a new tensor with element-wise difference.
    Tensor<int> sub_result = tsr::subtract(a, b);
    assert(sub_result.shape() == a.shape());
    for (size_t i = 0; i < sub_result.size(); ++i) {
        assert(sub_result.data()[i] == 1);
    }

    // In-place variants mutate the first argument.
    Tensor<int> add_inplace_target({2, 2}, 3);
    tsr::add_inplace(add_inplace_target, b);
    for (size_t i = 0; i < add_inplace_target.size(); ++i) {
        assert(add_inplace_target.data()[i] == 5);
    }

    Tensor<int> sub_inplace_target({2, 2}, 3);
    tsr::subtract_inplace(sub_inplace_target, b);
    for (size_t i = 0; i < sub_inplace_target.size(); ++i) {
        assert(sub_inplace_target.data()[i] == 1);
    }

    // Unsafe variants should still compute correctly for valid matching shapes.
    Tensor<int> add_unsafe_result = tsr::add_unsafe(a, b);
    Tensor<int> sub_unsafe_result = tsr::subtract_unsafe(a, b);
    for (size_t i = 0; i < add_unsafe_result.size(); ++i) {
        assert(add_unsafe_result.data()[i] == 5);
        assert(sub_unsafe_result.data()[i] == 1);
    }

    // Safe variants throw when shapes mismatch.
    Tensor<int> mismatch({2, 3}, 1);
    bool add_threw = false;
    try {
        (void)tsr::add(a, mismatch);
    } catch (const std::invalid_argument&) {
        add_threw = true;
    }
    assert(add_threw);

    bool sub_threw = false;
    try {
        (void)tsr::subtract(a, mismatch);
    } catch (const std::invalid_argument&) {
        sub_threw = true;
    }
    assert(sub_threw);

    bool add_inplace_threw = false;
    Tensor<int> add_target_mismatch({2, 2}, 0);
    try {
        tsr::add_inplace(add_target_mismatch, mismatch);
    } catch (const std::invalid_argument&) {
        add_inplace_threw = true;
    }
    assert(add_inplace_threw);

    bool sub_inplace_threw = false;
    Tensor<int> sub_target_mismatch({2, 2}, 0);
    try {
        tsr::subtract_inplace(sub_target_mismatch, mismatch);
    } catch (const std::invalid_argument&) {
        sub_inplace_threw = true;
    }
    assert(sub_inplace_threw);

    // Rank-0 scalar arithmetic.
    Tensor<int> scalar_a;
    Tensor<int> scalar_b;
    scalar_a.at({}) = 10;
    scalar_b.at({}) = 6;

    Tensor<int> scalar_add = tsr::add(scalar_a, scalar_b);
    Tensor<int> scalar_sub = tsr::subtract(scalar_a, scalar_b);
    assert(scalar_add.at({}) == 16);
    assert(scalar_sub.at({}) == 4);

    return 0;
}
