#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tsr {

// UTILITY FUNCTIONS (Not part of the Tensor class)
// Computes the total number of elements in a tensor given its shape
template<std::size_t Rank>
inline size_t compute_size(const std::array<size_t, Rank>& shape) {
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
template<std::size_t Rank>
inline std::array<size_t, Rank> compute_strides(const std::array<size_t, Rank>& shape) {
    std::array<size_t, Rank> strides{};
    if constexpr (Rank == 0) {
        return strides;
    }

    strides[Rank - 1] = 1;
    for (std::size_t i = Rank; i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

template<typename T, std::size_t Rank>
class Tensor {
private:
    std::vector<T> m_data;
    std::array<size_t, Rank> m_shape{};
    size_t m_size;
    std::array<size_t, Rank> m_strides{};

    // Validates that the provided array of indices is within the bounds of the tensor's shape
    void validate_indices(const std::array<size_t, Rank>& indices) const {
        if constexpr (Rank == 0) {
            if (m_size == 0) {
                throw std::out_of_range("Cannot index an empty tensor.");
            }
            if (indices.size() == 0) {
                return;
            }
            throw std::out_of_range("Scalar tensor does not accept indices.");
        } else {
            if (m_size == 0) {
                throw std::out_of_range("Cannot index an empty tensor.");
            }

            for (size_t i = 0; i < Rank; ++i) {
                if (indices[i] >= m_shape[i]) {
                    throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
                }
            }
        }
    }

    // Validates that the provided initializer list of indices is within the bounds of the tensor's shape
    void validate_indices(const std::initializer_list<size_t>& indices) const {
        if (Rank == 0) {
            if (m_size == 0) {
                throw std::out_of_range("Cannot index an empty tensor.");
            }
            if (indices.size() == 0) {
                return;
            }
            throw std::out_of_range("Scalar tensor does not accept indices.");
        }

        if (indices.size() != Rank) {
            throw std::out_of_range("Number of indices does not match tensor dimensions.");
        }

        if (m_size == 0) {
            throw std::out_of_range("Cannot index an empty tensor.");
        }

        size_t i = 0;
        for (size_t index : indices) {
            if (index >= m_shape[i]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
            ++i;
        }
    }

    // Converts multi-dimensional array of indices to a flat index in the underlying data vector, with safety checks
    size_t flat_index_safe(const std::array<size_t, Rank>& indices) const {
        validate_indices(indices);

        size_t flat_index_value = 0;
        for (size_t i = 0; i < Rank; ++i) {
            flat_index_value += indices[i] * m_strides[i];
        }
        return flat_index_value;
    }

    // Converts the multi-dimensional array of indices to a flat index in the underlying data vector, without safety checks
    size_t flat_index_unsafe(const std::array<size_t, Rank>& indices) const {
        size_t flat_index_value = 0;
        for (size_t i = 0; i < Rank; ++i) {
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
    Tensor() : m_size(1) {
        m_data.resize(m_size);
    }
    Tensor(const std::array<size_t, Rank>& shape);
    Tensor(const std::array<size_t, Rank>& shape, const T& initial_value);
    Tensor(const std::initializer_list<size_t>& shape);
    Tensor(const std::initializer_list<size_t>& shape, const T& initial_value);

    // CONSTRUCTOR WRAPPERS
    static Tensor<T, Rank> zeros(const std::array<size_t, Rank>& shape);
    static Tensor<T, Rank> zeros(const std::initializer_list<size_t>& shape);

    // GETTERS
    // Returns the shape of the tensor as a vector of sizes
    const std::array<size_t, Rank>& shape() const noexcept;

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
    T& at(const std::array<size_t, Rank>& indices);
    const T& at(const std::array<size_t, Rank>& indices) const;

    // Element access with an initializer list of indices functions with safety checks
    T& at(const std::initializer_list<size_t>& indices);
    const T& at(const std::initializer_list<size_t>& indices) const;

    // Element access with a vector of indices without safety checks
    T& operator()(const std::array<size_t, Rank>& indices);
    const T& operator()(const std::array<size_t, Rank>& indices) const;

    // Element access with an initializer list of indices without safety checks
    T& operator()(const std::initializer_list<size_t>& indices);
    const T& operator()(const std::initializer_list<size_t>& indices) const;

    // MODIFIERS
    // Resizes the tensor to the new shape, preserving existing data where possible
    void resize(const std::array<size_t, Rank>& new_shape);
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
template<typename T, std::size_t Rank>
Tensor<T, Rank>::Tensor(const std::array<size_t, Rank>& shape)
    : m_shape(shape), m_size(compute_size(shape)), m_strides(compute_strides(shape)) {
    m_data.resize(m_size);
}

template<typename T, std::size_t Rank>
Tensor<T, Rank>::Tensor(const std::array<size_t, Rank>& shape, const T& initial_value)
    : m_shape(shape), m_size(compute_size(shape)), m_strides(compute_strides(shape)) {
    m_data.resize(m_size, initial_value);
}

template<typename T, std::size_t Rank>
Tensor<T, Rank>::Tensor(const std::initializer_list<size_t>& shape)
    : m_shape(), m_size(1), m_strides() {
    if (shape.size() != Rank) {
        throw std::invalid_argument("Initializer list size does not match tensor rank.");
    }

    std::size_t i = 0;
    for (size_t dim : shape) {
        m_shape[i++] = dim;
    }
    m_size = compute_size(m_shape);
    m_strides = compute_strides(m_shape);
    m_data.resize(m_size);
}

template<typename T, std::size_t Rank>
Tensor<T, Rank>::Tensor(const std::initializer_list<size_t>& shape, const T& initial_value)
    : m_shape(), m_size(1), m_strides() {
    if (shape.size() != Rank) {
        throw std::invalid_argument("Initializer list size does not match tensor rank.");
    }

    std::size_t i = 0;
    for (size_t dim : shape) {
        m_shape[i++] = dim;
    }
    m_size = compute_size(m_shape);
    m_strides = compute_strides(m_shape);
    m_data.resize(m_size, initial_value);
}

// CONSTRUCTOR WRAPPERS
template<typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T, Rank>::zeros(const std::array<size_t, Rank>& shape) {
    return Tensor<T, Rank>(shape, T{});
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T, Rank>::zeros(const std::initializer_list<size_t>& shape) {
    return Tensor<T, Rank>(shape, T{});
}

// GETTERS
template<typename T, std::size_t Rank>
const std::array<size_t, Rank>& Tensor<T, Rank>::shape() const noexcept {
    return m_shape;
}

template<typename T, std::size_t Rank>
size_t Tensor<T, Rank>::size() const noexcept {
    return m_size;
}

template<typename T, std::size_t Rank>
std::vector<T>& Tensor<T, Rank>::data() noexcept {
    return m_data;
}

template<typename T, std::size_t Rank>
const std::vector<T>& Tensor<T, Rank>::data() const noexcept {
    return m_data;
}

template<typename T, std::size_t Rank>
T* Tensor<T, Rank>::data_ptr() noexcept {
    return m_data.data();
}

template<typename T, std::size_t Rank>
const T* Tensor<T, Rank>::data_ptr() const noexcept {
    return m_data.data();
}

template<typename T, std::size_t Rank>
bool Tensor<T, Rank>::empty() const noexcept {
    return m_data.empty();
}

// ELEMENT ACCESS
template<typename T, std::size_t Rank>
T& Tensor<T, Rank>::at(const std::array<size_t, Rank>& indices) {
    return m_data[flat_index_safe(indices)];
}

template<typename T, std::size_t Rank>
const T& Tensor<T, Rank>::at(const std::array<size_t, Rank>& indices) const {
    return m_data[flat_index_safe(indices)];
}

template<typename T, std::size_t Rank>
T& Tensor<T, Rank>::at(const std::initializer_list<size_t>& indices) {
    return m_data[flat_index_safe(indices)];
}

template<typename T, std::size_t Rank>
const T& Tensor<T, Rank>::at(const std::initializer_list<size_t>& indices) const {
    return m_data[flat_index_safe(indices)];
}

template<typename T, std::size_t Rank>
T& Tensor<T, Rank>::operator()(const std::array<size_t, Rank>& indices) {
    return m_data[flat_index_unsafe(indices)];
}

template<typename T, std::size_t Rank>
const T& Tensor<T, Rank>::operator()(const std::array<size_t, Rank>& indices) const {
    return m_data[flat_index_unsafe(indices)];
}

template<typename T, std::size_t Rank>
T& Tensor<T, Rank>::operator()(const std::initializer_list<size_t>& indices) {
    return m_data[flat_index_unsafe(indices)];
}

template<typename T, std::size_t Rank>
const T& Tensor<T, Rank>::operator()(const std::initializer_list<size_t>& indices) const {
    return m_data[flat_index_unsafe(indices)];
}

// MODIFIERS
template<typename T, std::size_t Rank>
void Tensor<T, Rank>::resize(const std::array<size_t, Rank>& new_shape) {
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

template<typename T, std::size_t Rank>
void Tensor<T, Rank>::resize(const std::initializer_list<size_t>& new_shape) {
    if (new_shape.size() != Rank) {
        throw std::invalid_argument("Initializer list size does not match tensor rank.");
    }

    std::array<size_t, Rank> shape{};
    std::size_t i = 0;
    for (size_t dim : new_shape) {
        shape[i++] = dim;
    }
    resize(shape);
}

template<typename T, std::size_t Rank>
void Tensor<T, Rank>::fill(const T& value) {
    std::fill(m_data.begin(), m_data.end(), value);
}

template<typename T, std::size_t Rank>
void Tensor<T, Rank>::clear() {
    m_data.clear();
    m_shape = {};
    m_size = 0;
    m_strides = {};
}

// ITERATORS
template<typename T, std::size_t Rank>
typename std::vector<T>::iterator Tensor<T, Rank>::begin() noexcept {
    return m_data.begin();
}

template<typename T, std::size_t Rank>
typename std::vector<T>::const_iterator Tensor<T, Rank>::begin() const noexcept {
    return m_data.begin();
}

template<typename T, std::size_t Rank>
typename std::vector<T>::iterator Tensor<T, Rank>::end() noexcept {
    return m_data.end();
}

template<typename T, std::size_t Rank>
typename std::vector<T>::const_iterator Tensor<T, Rank>::end() const noexcept {
    return m_data.end();
}

}  // namespace tsr