#pragma once

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace tsr {

// UTILITY FUNCTIONS (Not part of the Tensor class)
// Computes the total number of elements in a tensor given its shape
inline size_t compute_size(const std::vector<size_t>& shape) {
    if(shape.empty()) return 1; // Scalar tensor has size 1

    size_t size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            return 0;
        }
        // Avoid overflow in the product.
        if (size > (std::numeric_limits<size_t>::max() / dim)) {
            throw std::overflow_error("Tensor size overflow.");
        }
        size *= dim;
    }
    return size;
}

// Computes the strides for each dimension of a tensor given its shape
inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    if(shape.empty()) return {}; // Scalar tensor has no strides

    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size() - 1; i-- > 0;) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template<typename T>
class Tensor {
private:
    std::vector<T> m_data;
    std::vector<size_t> m_shape;
    size_t m_size;
    std::vector<size_t> m_strides;

    void validate_indices(const std::vector<size_t>& indices) const {
        if (m_shape.empty()){
            if (m_size == 0) {
                throw std::out_of_range("Cannot index an empty tensor.");
            }
            if (indices.empty()) return;
            throw std::out_of_range("Scalar tensor does not accept indices.");
        }

        if (indices.size() != m_shape.size()) {
            throw std::out_of_range("Number of indices does not match tensor dimensions.");
        }

        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= m_shape[i]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
        }
    }

    void validate_indices(const std::initializer_list<size_t>& indices) const {
        if (m_shape.empty()){
            if (m_size == 0) {
                throw std::out_of_range("Cannot index an empty tensor.");
            }
            if (indices.size() == 0) return;
            throw std::out_of_range("Scalar tensor does not accept indices.");
        }

        if (indices.size() != m_shape.size()) {
            throw std::out_of_range("Number of indices does not match tensor dimensions.");
        }

        size_t i = 0;
        for (size_t index : indices) {
            if (index >= m_shape[i]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
            ++i;
        }
    }

    // Converts multi-dimensional vector of indices to a flat index in the underlying data vector, with safety checks
    size_t flat_index_safe(const std::vector<size_t>& indices) const {
        validate_indices(indices);

        size_t flat_index_value = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_index_value += indices[i] * m_strides[i];
        }
        return flat_index_value;
    }

    // Converts the multi-dimensional vector of indices to a flat index in the underlying data vector, without safety checks
    size_t flat_index_unsafe(const std::vector<size_t>& indices) const {
        size_t flat_index_value = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_index_value += indices[i] * m_strides[i];
        }
        return flat_index_value;
    }

    // Converts the initializer list of indices to a flat index in the underlying data vector, with safety checks
    size_t flat_index_safe(const std::initializer_list<size_t>& indices) const {
        validate_indices(indices);

        size_t flat_index_value = 0;
        size_t i = 0;
        for (size_t index : indices) {
            flat_index_value += index * m_strides[i];
            ++i;
        }
        return flat_index_value;
    }

    // Converts the initializer list of indices to a flat index in the underlying data vector, without safety checks
    size_t flat_index_unsafe(const std::initializer_list<size_t>& indices) const {
        size_t flat_index_value = 0;
        size_t i = 0;
        for (size_t index : indices) {
            flat_index_value += index * m_strides[i];
            ++i;
        }
        return flat_index_value;
    }

public:
    // CONSTRUCTORS
    Tensor() : m_data(1), m_shape({}), m_size(1), m_strides({}) {}
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const T& initial_value);
    Tensor(const std::initializer_list<size_t>& shape);
    Tensor(const std::initializer_list<size_t>& shape, const T& initial_value);

    // CONSTRUCTOR WRAPPERS
    static Tensor<T> zeros(const std::vector<size_t>& shape);
    static Tensor<T> zeros(const std::initializer_list<size_t>& shape);

    // GETTERS
    // Returns the shape of the tensor as a vector of sizes
    const std::vector<size_t>& shape() const noexcept;

    // Returns the total number of elements in the tensor
    size_t size() const noexcept;

    // Returns a reference to the underlying data vector
    std::vector<T>& data() noexcept;
    const std::vector<T>& data() const noexcept;

    // Returns a pointer to the underlying data array
    T* data_ptr() noexcept;
    const T* data_ptr() const noexcept;

    // Returns whether the tensor is empty (i.e., has no elements)
    bool empty() const noexcept;

    // ELEMENT ACCESS
    // Element access with a vector of indices functions with safety checks
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;

    // Element access with an initializer list of indices functions with safety checks
    T& at(const std::initializer_list<size_t>& indices);
    const T& at(const std::initializer_list<size_t>& indices) const;

    // Element access with a vector of indices without safety checks
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    // Element access with an initializer list of indices without safety checks
    T& operator()(const std::initializer_list<size_t>& indices);
    const T& operator()(const std::initializer_list<size_t>& indices) const;

    // MODIFIERS
    // Resizes the tensor to the new shape, preserving existing data where possible
    void resize(const std::vector<size_t>& new_shape);
    void resize(const std::initializer_list<size_t>& new_shape);

    // Fills the tensor with the specified value
    void fill(const T& value);

    // Clears the tensor, removing all elements and resetting its shape
    void clear();

    // ITERATORS
    // Returns an iterator to the beginning of the data
    typename std::vector<T>::iterator begin() noexcept;
    typename std::vector<T>::const_iterator begin() const noexcept;

    // Returns an iterator to the end of the data
    typename std::vector<T>::iterator end() noexcept;
    typename std::vector<T>::const_iterator end() const noexcept;
};

// ------------------------------------------------------------------------------
// CLASS METHODS IMPLEMENTATIONS
// ------------------------------------------------------------------------------

// CONSTRUCTORS
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape)
    : m_shape(shape), m_size(compute_size(shape)), m_strides(compute_strides(shape)) {
    m_data.resize(m_size);
}

template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, const T& initial_value)
    : m_shape(shape), m_size(compute_size(shape)), m_strides(compute_strides(shape)) {
    m_data.resize(m_size, initial_value);
}

template<typename T>
Tensor<T>::Tensor(const std::initializer_list<size_t>& shape)
    : m_shape(shape), m_size(compute_size(shape)), m_strides(compute_strides(shape)) {
    m_data.resize(m_size);
}

template<typename T>
Tensor<T>::Tensor(const std::initializer_list<size_t>& shape, const T& initial_value)
    : m_shape(shape), m_size(compute_size(shape)), m_strides(compute_strides(shape)) {
    m_data.resize(m_size, initial_value);
}

// CONSTRUCTOR WRAPPERS
template<typename T>
Tensor<T> Tensor<T>::zeros(const std::vector<size_t>& shape){
    return Tensor<T>(shape, T{});
}

template<typename T>
Tensor<T> Tensor<T>::zeros(const std::initializer_list<size_t>& shape){
    return Tensor<T>(shape, T{});
}

// GETTERS
template<typename T>
const std::vector<size_t>& Tensor<T>::shape() const noexcept {
    return m_shape;
}

template<typename T>
size_t Tensor<T>::size() const noexcept {
    return m_size;
}

template<typename T>
std::vector<T>& Tensor<T>::data() noexcept {
    return m_data;
}

template<typename T>
const std::vector<T>& Tensor<T>::data() const noexcept {
    return m_data;
}

template<typename T>
T* Tensor<T>::data_ptr() noexcept {
    return m_data.data();
}

template<typename T>
const T* Tensor<T>::data_ptr() const noexcept {
    return m_data.data();
}

template<typename T>
bool Tensor<T>::empty() const noexcept {
    return m_data.empty();
}

// ELEMENT ACCESS
template<typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
    return m_data[flat_index_safe(indices)];
}

template<typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
    return m_data[flat_index_safe(indices)];
}

template<typename T>
T& Tensor<T>::at(const std::initializer_list<size_t>& indices) {
    return m_data[flat_index_safe(indices)];
}

template<typename T>
const T& Tensor<T>::at(const std::initializer_list<size_t>& indices) const {
    return m_data[flat_index_safe(indices)];
}

template<typename T>
T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
    return m_data[flat_index_unsafe(indices)];
}

template<typename T>
const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
    return m_data[flat_index_unsafe(indices)];
}

template<typename T>
T& Tensor<T>::operator()(const std::initializer_list<size_t>& indices) {
    return m_data[flat_index_unsafe(indices)];
}

template<typename T>
const T& Tensor<T>::operator()(const std::initializer_list<size_t>& indices) const {
    return m_data[flat_index_unsafe(indices)];
}

// MODIFIERS
template<typename T>
void Tensor<T>::resize(const std::vector<size_t>& new_shape) {
    const size_t new_size = compute_size(new_shape);
    std::vector<T> new_data(new_size);

    const size_t copy_count = std::min(m_data.size(), new_data.size());
    for (size_t i = 0; i < copy_count; ++i) {
        new_data[i] = m_data[i];
    }

    m_shape = new_shape;
    m_size = new_size;
    m_strides = compute_strides(new_shape);
    m_data = std::move(new_data);
}

template<typename T>
void Tensor<T>::resize(const std::initializer_list<size_t>& new_shape) {
    resize(std::vector<size_t>(new_shape.begin(), new_shape.end()));
}

template<typename T>
void Tensor<T>::fill(const T& value){
    std::fill(m_data.begin(), m_data.end(), value);
}

template<typename T>
void Tensor<T>::clear() {
    m_data.clear();
    m_shape.clear();
    m_size = 0;
    m_strides.clear();
}

// ITERATORS
template<typename T>
typename std::vector<T>::iterator Tensor<T>::begin() noexcept {
    return m_data.begin();
}

template<typename T>
typename std::vector<T>::const_iterator Tensor<T>::begin() const noexcept {
    return m_data.begin();
}

template<typename T>
typename std::vector<T>::iterator Tensor<T>::end() noexcept {
    return m_data.end();
}

template<typename T>
typename std::vector<T>::const_iterator Tensor<T>::end() const noexcept {
    return m_data.end();
}

}  // namespace tsr