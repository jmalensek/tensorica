#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <tensorica/tensor/tensor.hpp>

namespace tsr {

// ------------------------------------------------------------------------------
// DECLARATIONS
// ------------------------------------------------------------------------------

// Tensor addition function with safety checks. Returns a new Tensor.
template<typename T, std::size_t Rank> 
Tensor<T, Rank> add(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b);

// Tensor addition function without safety checks. Returns a new Tensor.
// Shape compatibility validation is left to the caller.
template<typename T, std::size_t Rank> 
Tensor<T, Rank> add_unsafe(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b);

// In-place tensor addition function with safety checks. Modifies the first tensor.
template<typename T, std::size_t Rank> 
void add_inplace(Tensor<T, Rank>& a, const Tensor<T, Rank>& b);

// In-place tensor addition function without safety checks. Modifies the first tensor.
// Shape compatibility validation is left to the caller.
template<typename T, std::size_t Rank> 
void add_inplace_unsafe(Tensor<T, Rank>& a, const Tensor<T, Rank>& b);

// ------------------------------------------------------------------------------
// IMPLEMENTATIONS
// ------------------------------------------------------------------------------

template<typename T, std::size_t Rank>
Tensor<T, Rank> add(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
    if(a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }

    Tensor<T, Rank> result(a.shape());
    size_t total_size = a.size();

    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();
    auto result_data = result.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> add_unsafe(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {
    Tensor<T, Rank> result(a.shape());
    size_t total_size = a.size();

    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();
    auto result_data = result.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

template<typename T, std::size_t Rank>
void add_inplace(Tensor<T, Rank>& a, const Tensor<T, Rank>& b){
    if(a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }

    size_t total_size = a.size();
    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        a_data[i] += b_data[i];
    }
}

template<typename T, std::size_t Rank>
void add_inplace_unsafe(Tensor<T, Rank>& a, const Tensor<T, Rank>& b){
    size_t total_size = a.size();
    auto a_data = a.data_ptr();
    auto b_data = b.data_ptr();

    for(size_t i = 0; i < total_size; ++i) {
        a_data[i] += b_data[i];
    }
}

} // namespace tsr