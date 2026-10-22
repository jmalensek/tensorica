#include <tensorica/tensor/tensor.hpp>

#include <cassert>
#include <stdexcept>

int main() {
    using tsr::Tensor;

    // Scalar tensor behavior (rank-0).
    Tensor<int> scalar;
    assert(scalar.size() == 1);
    scalar.at({}) = 7;
    assert(scalar.at({}) == 7);

    bool scalar_rejects_indices = false;
    try {
        (void)scalar.at({0});
    } catch (const std::out_of_range&) {
        scalar_rejects_indices = true;
    }
    assert(scalar_rejects_indices);

    // Basic shape/size/indexing.
    Tensor<int> t({2, 3}, 1);
    assert(t.size() == 6);
    assert(t.shape().size() == 2);
    assert(t.at({1, 2}) == 1);

    t.at({1, 2}) = 9;
    assert(t.at({1, 2}) == 9);

    // Bounds checks in safe access.
    bool throws_oob = false;
    try {
        (void)t.at({2, 0});
    } catch (const std::out_of_range&) {
        throws_oob = true;
    }
    assert(throws_oob);

    // Unsafe access path should work for valid indices.
    assert(t({1, 2}) == 9);

    // Resize preserves prefix data in linear storage order.
    t.resize({3, 3});
    assert(t.size() == 9);
    assert(t.data()[0] == 1);

    // Fill updates every element.
    t.fill(4);
    for (const int v : t) {
        assert(v == 4);
    }

    // Zero-size dimension.
    Tensor<int> z({0, 5});
    assert(z.size() == 0);
    assert(z.empty());

    // clear() removes storage and should reject scalar indexing afterwards.
    scalar.clear();
    assert(scalar.empty());
    bool clear_rejects_index = false;
    try {
        (void)scalar.at({});
    } catch (const std::out_of_range&) {
        clear_rejects_index = true;
    }
    assert(clear_rejects_index);

    return 0;
}
