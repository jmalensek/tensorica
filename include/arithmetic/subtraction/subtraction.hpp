#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <tensorica/tensor/tensor.hpp>

namespace tsr {

// ------------------------------------------------------------------------------
// DECLARATIONS
// ------------------------------------------------------------------------------

// Tensor subtraction function with safety checks. Returns a new Tensor.
template<typename T> Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b);

// Tensor subtraction function without safety checks. Returns a new Tensor.
// Shape compatibility validation is left to the caller.
template<typename T> Tensor<T> subtract_unsafe(const Tensor<T>& a, const Tensor<T>& b);

// In-place tensor subtraction function with safety checks. Modifies the first tensor.
template<typename T> void subtract_inplace(Tensor<T>& a, const Tensor<T>& b);

// In-place tensor subtraction function without safety checks. Modifies the first tensor.
// Shape compatibility validation is left to the caller.
template<typename T> void subtract_inplace_unsafe(Tensor<T>& a, const Tensor<T>& b);

// ------------------------------------------------------------------------------
// IMPLEMENTATIONS
// ------------------------------------------------------------------------------

template<typename T>
Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b) {
    if(a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for subtraction.");
    }

    Tensor<T> result(a.shape());
    size_t total_size = a.size();

    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();
    auto result_data = result.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        result_data[i] = a_data[i] - b_data[i];
    }

    return result;
}

template<typename T>
Tensor<T> subtract_unsafe(const Tensor<T>& a, const Tensor<T>& b) {
    Tensor<T> result(a.shape());
    size_t total_size = a.size();

    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();
    auto result_data = result.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        result_data[i] = a_data[i] - b_data[i];
    }

    return result;
}

template<typename T>
void subtract_inplace(Tensor<T>& a, const Tensor<T>& b){
    if(a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for subtraction.");
    }

    size_t total_size = a.size();
    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        a_data[i] -= b_data[i];
    }
}

template<typename T>
void subtract_inplace_unsafe(Tensor<T>& a, const Tensor<T>& b){
    size_t total_size = a.size();
    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        a_data[i] -= b_data[i];
    }
}

} // namespace tsr