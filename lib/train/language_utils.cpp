#include "train/language_utils.hpp"

#include <algorithm>
#include <stdexcept>

#include "nn.hpp"
#include "probs.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"

namespace train {

float evaluate_sequence_nll(nn::Sequential& model, ParameterStore& store,
                            Tensor& scratch_input,
                            const std::vector<int>& sequence,
                            int vocab_size) {
  if (sequence.size() < 2) return 0.0f;
  scratch_input.fill(0.0f);
  float total = 0.0f;
  for (size_t i = 0; i + 1 < sequence.size(); ++i) {
    scratch_input.fill(0.0f);
    fill_one_hot(scratch_input, 0, sequence[i]);
    Tensor logits = model(scratch_input, store);
    const float* logits_ptr = logits.data();
    auto probs = softmax_from_logits(logits_ptr, vocab_size);
    float prob = std::max(probs[sequence[i + 1]], 1e-8f);
    total += -std::log(prob);
    store.clear_tape();
  }
  return total / static_cast<float>(sequence.size() - 1);
}

float evaluate_sequence_accuracy(nn::Sequential& model, ParameterStore& store,
                                 Tensor& scratch_input,
                                 const std::vector<int>& sequence,
                                 int vocab_size) {
  if (sequence.size() < 2) return 0.0f;
  scratch_input.fill(0.0f);
  int correct = 0;
  int total = 0;
  for (size_t i = 0; i + 1 < sequence.size(); ++i) {
    scratch_input.fill(0.0f);
    fill_one_hot(scratch_input, 0, sequence[i]);
    Tensor logits = model(scratch_input, store);
    const float* logits_ptr = logits.data();
    int predicted = argmax_from_logits(logits_ptr, vocab_size);
    if (predicted == sequence[i + 1]) ++correct;
    ++total;
    store.clear_tape();
  }
  if (total == 0) return 0.0f;
  return static_cast<float>(correct) / static_cast<float>(total);
}

int sample_next_token(const Tensor& logits, int vocab_size) {
  if (vocab_size <= 0) {
    throw std::invalid_argument("vocab_size must be positive");
  }
  const float* logits_ptr = logits.data();
  auto probs = softmax_from_logits(logits_ptr, vocab_size);
  MultinomialDistribution dist(probs);
  auto sampled = dist.sample(1);
  return sampled.empty() ? 0 : sampled[0];
}

}  // namespace train

