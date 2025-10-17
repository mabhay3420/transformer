/**
 * @file nn.hpp
 * @brief Neural network modules and loss functions.
 *
 * Provides a modular neural network API with automatic differentiation.
 * Includes common layers, activations, and loss functions.
 */

#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensor.hpp"

namespace nn {

/**
 * @class Module
 * @brief Base class for neural network layers and models.
 *
 * All neural network components inherit from Module. Provides forward pass
 * interface and parameter management.
 */
struct Module {
  virtual ~Module() = default;

  /**
   * @brief Forward pass through the module.
   * @param x Input tensor
   * @param store ParameterStore for memory allocation
   * @return Output tensor
   */
  virtual Tensor forward(const Tensor& x, ParameterStore& store) = 0;

  /**
   * @brief Function call operator for convenience.
   * @param x Input tensor
   * @param store ParameterStore for memory allocation
   * @return Output tensor
   */
  Tensor operator()(const Tensor& x, ParameterStore& store) {
    return forward(x, store);
  }

  /**
   * @brief Get list of learnable parameters.
   * @return Vector of parameter tensors
   */
  virtual std::vector<Tensor> params() { return {}; }
};

/**
 * @class Linear
 * @brief Fully connected linear layer.
 *
 * Performs affine transformation: y = xW^T + b (if bias enabled).
 */
struct Linear : public Module {
  int in_features;   ///< Number of input features
  int out_features;  ///< Number of output features
  bool use_bias;     ///< Whether to include bias term
  Tensor W;          ///< Weight matrix [out_features, in_features]
  Tensor b;          ///< Bias vector [out_features] (empty if no bias)

  /**
   * @brief Construct a linear layer.
   * @param in_f Number of input features
   * @param out_f Number of output features
   * @param store ParameterStore for parameter allocation
   * @param bias Whether to use bias (default true)
   * @param init_scale Weight initialization scale (default 0.5)
   * @param seed Random seed for initialization
   */
  Linear(int in_f, int out_f, ParameterStore& store, bool bias = true,
         float init_scale = 0.5f, unsigned seed = 0);

  /**
   * @brief Forward pass: y = xW^T + b
   * @param x Input tensor [batch_size, in_features]
   * @param store ParameterStore for computation
   * @return Output tensor [batch_size, out_features]
   */
  Tensor forward(const Tensor& x, ParameterStore& store) override;

  /**
   * @brief Get learnable parameters.
   * @return Vector containing W and optionally b
   */
  std::vector<Tensor> params() override;
};

/**
 * @class Tanh
 * @brief Hyperbolic tangent activation layer.
 */
struct Tanh : public Module {
  /**
   * @brief Apply tanh activation.
   * @param x Input tensor
   * @param store ParameterStore for computation
   * @return Activated tensor
   */
  Tensor forward(const Tensor& x, ParameterStore& store) override;
};

/**
 * @class Relu
 * @brief Rectified Linear Unit activation layer.
 */
struct Relu : public Module {
  /**
   * @brief Apply ReLU activation.
   * @param x Input tensor
   * @param store ParameterStore for computation
   * @return Activated tensor
   */
  Tensor forward(const Tensor& x, ParameterStore& store) override;
};

/**
 * @class Sigmoid
 * @brief Sigmoid activation layer.
 */
struct Sigmoid : public Module {
  /**
   * @brief Apply sigmoid activation.
   * @param x Input tensor
   * @param store ParameterStore for computation
   * @return Activated tensor
   */
  Tensor forward(const Tensor& x, ParameterStore& store) override;
};

/**
 * @class Sequential
 * @brief Container for sequential layer composition.
 *
 * Chains multiple modules together, passing output of one as input to the next.
 */
struct Sequential : public Module {
  std::vector<std::unique_ptr<Module>> layers;  ///< Sequence of layers

  Sequential() = default;

  /**
   * @brief Forward pass through all layers.
   * @param x Input tensor
   * @param store ParameterStore for computation
   * @return Output tensor after all layers
   */
  Tensor forward(const Tensor& x, ParameterStore& store) override;

  /**
   * @brief Get all parameters from all layers.
   * @return Concatenated parameter list
   */
  std::vector<Tensor> params() override;

  /**
   * @brief Add a layer to the sequence.
   * @param m Unique pointer to module
   */
  void push_back(std::unique_ptr<Module> m) { layers.push_back(std::move(m)); }

  /**
   * @brief Construct and add a layer in-place.
   * @tparam T Module type
   * @tparam Args Constructor arguments
   * @param args Arguments for T constructor
   * @return Reference to the added layer
   */
  template <typename T, typename... Args>
  T& emplace_back(Args&&... args) {
    static_assert(std::is_base_of_v<Module, T>,
                  "Sequential accepts Module-derived layers only");
    auto layer = std::make_unique<T>(std::forward<Args>(args)...);
    T& ref = *layer;
    layers.push_back(std::move(layer));
    return ref;
  }
};

/**
 * @brief Binary Cross-Entropy loss with logits.
 *
 * Computes BCE loss between logits and targets, with numerical stability.
 * @param logits Predicted logits [batch_size, num_classes]
 * @param targets Target labels [batch_size, num_classes] (0 or 1)
 * @param store ParameterStore for computation
 * @param eps Small epsilon for numerical stability (default 1e-6)
 * @return Mean loss scalar
 */
Tensor bce_with_logits_loss(const Tensor& logits, const Tensor& targets,
                            ParameterStore& store, float eps = 1e-6f);

}  // namespace nn
