/**
 * @file tensor.hpp
 * @brief Core tensor library providing automatic differentiation and memory
 * management.
 *
 * This library implements a minimal tensor API with separate contiguous storage
 * for data and gradients.
 * - ParameterStore manages memory allocation and owns data/grad buffers.
 * - Tensor provides a lightweight view into the buffers with shape and offset
 * information.
 * - Supports automatic differentiation via a tape-based system.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

struct ParameterStore;

/**
 * @struct ParameterStoreStats
 * @brief Statistics for ParameterStore operations.
 *
 * Tracks performance metrics for memory management and gradient operations.
 */
struct ParameterStoreStats {
  size_t tensor_zero_calls =
      0;  ///< Number of times tensors were zero-initialized
  size_t tensor_zero_elems =
      0;  ///< Total elements zeroed during tensor creation
  double tensor_zero_ms = 0.0;      ///< Time spent zeroing tensors (ms)
  size_t zero_grad_calls = 0;       ///< Number of zero_grad() calls
  size_t zero_grad_elems = 0;       ///< Total elements zeroed in gradients
  double zero_grad_ms = 0.0;        ///< Time spent zeroing gradients (ms)
  size_t reserve_calls = 0;         ///< Number of reserve() calls
  size_t reserve_elements = 0;      ///< Maximum reserved elements
  size_t capacity_grow_events = 0;  ///< Number of buffer reallocations
  size_t peak_elements = 0;         ///< Peak memory usage in elements
};

/**
 * @enum TensorInit
 * @brief Initialization options for tensor data.
 */
enum class TensorInit {
  ZeroData,          ///< Initialize tensor data to zeros
  UninitializedData  ///< Leave tensor data uninitialized (faster)
};

/**
 * @enum OpType
 * @brief Types of operations recorded in the autograd tape.
 */
enum class OpType {
  Add,        ///< Element-wise addition
  Sub,        ///< Element-wise subtraction
  Mul,        ///< Element-wise multiplication
  Relu,       ///< Rectified Linear Unit activation
  Tanh,       ///< Hyperbolic tangent activation
  Sigmoid,    ///< Sigmoid activation
  Log,        ///< Natural logarithm
  Sum,        ///< Sum reduction to scalar
  Matmul,     ///< Matrix multiplication
  AddRowwise  ///< Add bias vector to each row
};

/**
 * @struct Tensor
 * @brief Lightweight tensor view with automatic differentiation support.
 *
 * A Tensor represents a multi-dimensional array stored in a ParameterStore.
 * It provides access to data and gradients, and supports autograd operations.
 */
struct Tensor {
  ParameterStore* store = nullptr;  ///< Pointer to the owning ParameterStore
  size_t offset = 0;                ///< Starting index into store buffers
  std::vector<int> shape;           ///< Tensor dimensions (contiguous layout)
  size_t numel = 0;  ///< Total number of elements (product of shape)

  Tensor() = default;

  /**
   * @brief Construct a Tensor view.
   * @param s Owning ParameterStore
   * @param off Offset into buffers
   * @param sh Shape vector
   * @param n Number of elements
   */
  Tensor(ParameterStore* s, size_t off, std::vector<int> sh, size_t n)
      : store(s), offset(off), shape(std::move(sh)), numel(n) {}

  /**
   * @brief Get mutable pointer to tensor data.
   * @return Pointer to data buffer
   */
  float* data();

  /**
   * @brief Get mutable pointer to gradient buffer.
   * @return Pointer to gradient buffer
   */
  float* grad();

  /**
   * @brief Get const pointer to tensor data.
   * @return Const pointer to data buffer
   */
  const float* data() const;

  /**
   * @brief Get const pointer to gradient buffer.
   * @return Const pointer to gradient buffer
   */
  const float* grad() const;

  /**
   * @brief Zero out the gradient buffer for this tensor.
   */
  void zero_grad();

  /**
   * @brief Fill tensor data with a constant value.
   * @param v Value to fill
   */
  void fill(float v);
};

/**
 * @struct TapeOp
 * @brief Represents an operation in the autograd computation graph.
 *
 * Each operation records its type, output tensor, and input tensors for
 * backward pass computation.
 */
struct TapeOp {
  OpType type;  ///< Type of operation
  Tensor out;   ///< Output tensor
  Tensor a;     ///< First input tensor (or only input for unary ops)
  Tensor b;     ///< Second input tensor (unused for unary ops)
};

/**
 * @class ParameterStore
 * @brief Memory manager for tensors with automatic differentiation support.
 *
 * Manages contiguous buffers for tensor data and gradients. Provides memory
 * allocation, reuse, and autograd functionality. Tracks statistics for
 * performance monitoring.
 */
struct ParameterStore {
  std::unique_ptr<float[]> data_buf;  ///< Buffer for tensor data
  std::unique_ptr<float[]> grad_buf;  ///< Buffer for gradients
  size_t capacity = 0;                ///< Current buffer capacity in elements
  size_t used = 0;                    ///< Currently used elements
  std::vector<TapeOp> tape;           ///< Operation tape for autograd
  ParameterStoreStats stats;          ///< Performance statistics
  bool stats_enabled = false;         ///< Whether to collect statistics
  std::mt19937 rng{5489u};            ///< Deterministic RNG for parameters

  // Internal tracking for parameter gradients
  size_t param_grad_offset = 0;
  size_t param_grad_span = 0;
  size_t param_grad_elements = 0;
  bool param_block_initialized = false;
  bool param_block_contiguous = true;

  /**
   * @brief Allocate space for a tensor.
   * @param count Number of elements to allocate
   * @return Starting offset in buffers
   */
  size_t allocate(size_t count);

  /**
   * @brief Pre-allocate capacity for efficiency.
   * @param total_elements Total elements to reserve
   */
  void reserve(size_t total_elements);

  /**
   * @brief Ensure sufficient capacity (internal).
   * @param required Minimum required elements
   */
  void ensure_capacity(size_t required);

  /**
   * @brief Mark current memory usage for later reset.
   * @return Current used count
   */
  size_t mark() const;

  /**
   * @brief Reset memory usage to a previous mark.
   * @param mark Value returned by mark()
   */
  void reset(size_t mark);

  /**
   * @brief Get current memory usage.
   * @return Number of used elements
   */
  size_t size() const { return used; }

  /**
   * @brief Get current buffer capacity.
   * @return Capacity in elements
   */
  size_t capacity_count() const { return capacity; }

  /**
   * @brief Get data pointer at offset.
   * @param offset Buffer offset
   * @return Mutable data pointer
   */
  float* data_ptr(size_t offset);

  /**
   * @brief Get const data pointer at offset.
   * @param offset Buffer offset
   * @return Const data pointer
   */
  const float* data_ptr(size_t offset) const;

  /**
   * @brief Get gradient pointer at offset.
   * @param offset Buffer offset
   * @return Mutable gradient pointer
   */
  float* grad_ptr(size_t offset);

  /**
   * @brief Get const gradient pointer at offset.
   * @param offset Buffer offset
   * @return Const gradient pointer
   */
  const float* grad_ptr(size_t offset) const;

  /**
   * @brief Create a new tensor.
   * @param shape Tensor dimensions
   * @param init Initialization type
   * @return New tensor
   */
  Tensor tensor(const std::vector<int>& shape,
                TensorInit init = TensorInit::UninitializedData);

  /**
   * @brief Create a learnable parameter tensor.
   * @param shape Parameter dimensions
   * @param scale Initialization scale
   * @param seed Random seed
   * @return Parameter tensor
   */
  Tensor parameter(const std::vector<int>& shape, float scale = 0.01f,
                   unsigned seed = 0);

  /**
   * @brief Seed the internal random number generator.
   * @param seed New RNG seed
   */
  void seed(unsigned seed);

  /**
   * @brief Enable or disable statistics collection.
   * @param enabled Whether to collect stats
   */
  void enable_stats(bool enabled = true);

  /**
   * @brief Reset statistics counters.
   */
  void reset_stats();

  /**
   * @brief Get current statistics.
   * @return Reference to stats struct
   */
  const ParameterStoreStats& get_stats() const;

  /**
   * @brief Check if statistics are enabled.
   * @return True if stats are active
   */
  bool stats_active() const { return stats_enabled; }

  /**
   * @brief Print statistics to stdout.
   */
  void print_stats() const;

  /**
   * @brief Zero all gradients.
   */
  void zero_grad();

  /**
   * @brief Clear the operation tape.
   */
  void clear_tape();

  /**
   * @brief Compute gradients via backpropagation.
   * @param loss Loss tensor to differentiate
   */
  void backward(const Tensor& loss);

 private:
  /**
   * @brief Register parameter allocation for gradient management.
   * @param offset Allocation offset
   * @param count Number of elements
   */
  void register_parameter_allocation(size_t offset, size_t count);
};

/**
 * @name Tensor Operations
 * @brief Basic tensor operations with autograd support.
 * @{
 */

/**
 * @brief Element-wise addition.
 * @param a First tensor
 * @param b Second tensor (same shape as a)
 * @param store ParameterStore for memory allocation
 * @return Result tensor
 */
Tensor add(const Tensor& a, const Tensor& b, ParameterStore& store);

/**
 * @brief Element-wise subtraction.
 * @param a First tensor
 * @param b Second tensor (same shape as a)
 * @param store ParameterStore for memory allocation
 * @return Result tensor
 */
Tensor sub(const Tensor& a, const Tensor& b, ParameterStore& store);

/**
 * @brief Element-wise multiplication.
 * @param a First tensor
 * @param b Second tensor (same shape as a)
 * @param store ParameterStore for memory allocation
 * @return Result tensor
 */
Tensor mul(const Tensor& a, const Tensor& b, ParameterStore& store);

/**
 * @brief Matrix multiplication.
 * @param a Left matrix [M,K]
 * @param b Right matrix [K,N]
 * @param store ParameterStore for memory allocation
 * @return Result matrix [M,N]
 */
Tensor matmul(const Tensor& a, const Tensor& b, ParameterStore& store);

/**
 * @brief Rectified Linear Unit activation.
 * @param x Input tensor
 * @param store ParameterStore for memory allocation
 * @return Activated tensor
 */
Tensor relu(const Tensor& x, ParameterStore& store);

/**
 * @brief Hyperbolic tangent activation.
 * @param x Input tensor
 * @param store ParameterStore for memory allocation
 * @return Activated tensor
 */
Tensor vtanh(const Tensor& x, ParameterStore& store);

/**
 * @brief Sigmoid activation.
 * @param x Input tensor
 * @param store ParameterStore for memory allocation
 * @return Activated tensor
 */
Tensor sigmoid(const Tensor& x, ParameterStore& store);

/**
 * @brief Natural logarithm.
 * @param x Input tensor
 * @param store ParameterStore for memory allocation
 * @return Logarithm tensor
 */
Tensor vlog(const Tensor& x, ParameterStore& store);

/**
 * @brief Sum reduction to scalar.
 * @param x Input tensor
 * @param store ParameterStore for memory allocation
 * @return Scalar tensor [1]
 */
Tensor sum(const Tensor& x, ParameterStore& store);

/**
 * @brief Add bias vector to each row.
 * @param X Matrix [N,H]
 * @param b Bias vector [H]
 * @param store ParameterStore for memory allocation
 * @return Result matrix [N,H]
 */
Tensor add_rowwise(const Tensor& X, const Tensor& b, ParameterStore& store);

/** @} */
