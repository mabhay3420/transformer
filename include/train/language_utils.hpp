#pragma once

#include <vector>

#include "tensor.hpp"

namespace nn {
struct Sequential;
}  // namespace nn

namespace train {

// Compute average negative log-likelihood over consecutive next-token targets.
float evaluate_sequence_nll(nn::Sequential& model, ParameterStore& store,
                            Tensor& scratch_input,
                            const std::vector<int>& sequence,
                            int vocab_size);

// Measure next-token accuracy over a sequence using one-hot evaluation input.
float evaluate_sequence_accuracy(nn::Sequential& model, ParameterStore& store,
                                 Tensor& scratch_input,
                                 const std::vector<int>& sequence,
                                 int vocab_size);

// Sample a token index from logits by applying softmax and multinomial draw.
int sample_next_token(const Tensor& logits, int vocab_size);

}  // namespace train

