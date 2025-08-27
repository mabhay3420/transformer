#include "embednlp.hpp"

#include <iomanip>
#include <vector>

#include "bigramnn.hpp"
#include "dataloader.hpp"
#include "iostream"
#include "learning_rate.hpp"
#include "loss.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "neuron.hpp"
#include "nlohmann/json_fwd.hpp"
#include "optimizer.hpp"
#include "probs.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"

// Each Input = CONTEXT_LENGTH number of characters x EMBED_DIM number of float
// values target a single character
// Each example corresponds to CONTEXT_LENGTH * EMBED_DIM number of inputs and 1
// target
struct BigramMLPBatch {
  std::vector<std::vector<MemPoolIndex>> input;
  std::vector<MemPoolIndex> target;
};
/*
{
  Original: ABHAY
  Context: 3
  INPUT = [0,0,0...]
  TARGET
  ... -> A
  ..A -> B
  .AB -> H
  ABH -> A
  BHA -> Y
}
*/
BigramMLPData getBigramMLPData(std::vector<int> &data, int context_length,
                               int start_char_index) {
  BigramMLPData seqData;
  for (int i = 0; i < data.size(); i++) {
    std::vector<int> input(context_length, start_char_index);
    std::vector<int> target;
    for (int j = 0; j < context_length; j++) {
      auto index = i - (context_length - j);
      if (index < 0) continue;
      input[j] = data[index];
    }
    seqData.input.push_back(input);
    seqData.target.push_back(data[i]);
  }
  return seqData;
}
void EmbedNLP() {
  auto result = load_text_data("data/input.txt");
  std::set<char> unique_chars(result.begin(), result.end());
  int vocab_size = unique_chars.size();
  CharTokenizer tokenizer(unique_chars);
  auto data = tokenizer.encode(result);
  std::vector<int> train_data, val_data;
  split_data<int>(0.9f, data, train_data, val_data);

  // prepare train data

  int CONTEXT_LENGTH = 24;  // total context length
  int START_CHAR_INDEX = tokenizer.encode('.');
  int EMBED_DIM = 10;
  BigramMLPData trainSeqData =
      getBigramMLPData(train_data, CONTEXT_LENGTH, START_CHAR_INDEX);
  BigramMLPData valSeqData =
      getBigramMLPData(val_data, CONTEXT_LENGTH, START_CHAR_INDEX);
  std::cout << "Training data size: " << trainSeqData.input.size() << std::endl;
  std::cout << "Validation data size: " << valSeqData.input.size() << std::endl;
  auto mem_pool = new MemPool<Value>();

  //   auto n = Neuron(vocab_size, mem_pool, false);
  // auto n = Layer(EMBED_DIM * CONTEXT_LENGTH, 100, mem_pool, false, true,
  //                Activation::TANH);
  auto n = MLP(EMBED_DIM * CONTEXT_LENGTH, {100, 100, vocab_size}, mem_pool,
               false, true, Activation::TANH);
  auto params = n.params();
  std::vector<std::vector<MemPoolIndex>> embedding_matrix(
      vocab_size, std::vector<MemPoolIndex>(EMBED_DIM));
  // initially fill the embedding matrix with random values
  for (int i = 0; i < vocab_size; i++) {
    for (int j = 0; j < EMBED_DIM; j++) {
      embedding_matrix[i][j] =
          val(static_cast<float>(rand()) / RAND_MAX, mem_pool);
    }
  }
  mem_pool->set_persistent_boundary();
  auto getRandomBatchFn = [&](int batch_size) {
    std::vector<int> indices(batch_size);
    auto total_size = trainSeqData.input.size();
    for (int i = 0; i < batch_size; i++) {
      indices[i] = rand() % (total_size);
    }
    std::vector<std::vector<MemPoolIndex>> train_data_input;
    std::vector<MemPoolIndex> train_data_target;
    for (auto i : indices) {
      auto train_input = trainSeqData.input[i];
      std::vector<MemPoolIndex> train_input_embeddings_concatenated;
      for (auto c_i : train_input) {
        auto c_i_embedding = embedding_matrix[c_i];
        for (auto e_i : c_i_embedding) {
          train_input_embeddings_concatenated.push_back(e_i);
        }
      }
      train_data_input.push_back(train_input_embeddings_concatenated);
      train_data_target.push_back(val(trainSeqData.target[i], mem_pool));
    }
    return BigramMLPBatch{
        train_data_input,
        train_data_target,
    };
  };
  // auto TOTAL_EPOCH = train_data.size() / BATCH_SIZE;
  // auto TOTAL_EPOCH = 5000;
  auto BATCH_SIZE = 128;
  auto TOTAL_EPOCH = 1000;
  // 0.01 good
  auto LR0 = 0.01f;
  auto LR_GAMMA = 0.5f;
  auto LR_CLIFF = TOTAL_EPOCH - (TOTAL_EPOCH / 10);
  auto START_LR_EXP = -3.0f;
  auto END_LR_EXP = -1.0f;
  StepLRScheduler lr_scheduler(LR0, LR_CLIFF, LR_GAMMA);
  // ConstantLRScheduler lr_scheduler(LR0);
  // ExpLinspaceLRScheduler lr_scheduler(START_LR_EXP, END_LR_EXP, TOTAL_EPOCH);
  AdamWOptimizer<StepLRScheduler> optimizer(mem_pool, params, lr_scheduler);
  // AdamWOptimizer<ConstantLRScheduler> optimizer(mem_pool, params,
  // lr_scheduler); AdamWOptimizer<ExpLinspaceLRScheduler> optimizer(mem_pool,
  // params, lr_scheduler);

  vector<float> losses;
  vector<float> lri;
  std::cout << "Total epochs: " << TOTAL_EPOCH << std::endl;
  for (auto epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
    optimizer.zero_grad();
    mem_pool->deallocate_temp();
    std::vector<std::vector<MemPoolIndex>> predicted;
    std::vector<MemPoolIndex> expected;
    auto batch = getRandomBatchFn(BATCH_SIZE);
    for (int i = 0; i < batch.input.size(); i++) {
      auto r = n(batch.input[i]);
      auto probs = softmax(r, mem_pool);
      predicted.push_back(probs);
      expected.push_back(batch.target[i]);
    }
    auto loss = cross_entropy(predicted, expected, mem_pool);
    losses.push_back(mem_pool->get(loss)->data);
    lri.push_back(lr_scheduler.getLog());
    backprop(loss, mem_pool);
    optimizer.step();
    std::cout << "Epoch: " << epoch << " Loss: " << mem_pool->get(loss)->data
              << std::endl;
  }

  // a predicted sentence

  std::vector<int> predictedCharIndices(CONTEXT_LENGTH, START_CHAR_INDEX);
  auto totalToPredict = 100;
  for (int i = 0; i < totalToPredict; i++) {
    mem_pool->deallocate_temp();
    std::vector<MemPoolIndex> embedding;
    for (auto c_i : predictedCharIndices) {
      auto c_i_embedding = embedding_matrix[c_i];
      for (auto e_i : c_i_embedding) {
        embedding.push_back(e_i);
      }
    }
    auto predictedProbs_v = softmax(n(embedding), mem_pool);
    std::vector<float> predictedProbs;
    for (auto p : predictedProbs_v) {
      predictedProbs.push_back(mem_pool->get(p)->data);
    }
    MultinomialDistribution dist(predictedProbs);
    auto nextCharIndex = dist.sample(1)[0];
    std::cout << tokenizer.decode(nextCharIndex);
    for (int j = 0; j < CONTEXT_LENGTH - 1; j++) {
      predictedCharIndices[j] = predictedCharIndices[j + 1];
    }
    predictedCharIndices[CONTEXT_LENGTH - 1] = nextCharIndex;
  }

  json j = {
      {"losses", losses},
      {"lri", lri},
  };
  dumpJson(j, "data/losses_lri.json");
  // dumpJson(j, "data/losses.json");

  // //   Calculate validation loss
  // std::vector<std::vector<MemPoolIndex>> val_data_input;
  // std::vector<MemPoolIndex> val_data_target;
  // for (int i = 0; i < val_data.size(); i++) {
  //   auto input_one_hot = one_hot_encode(val_data[i], vocab_size, mem_pool);
  //   val_data_input.push_back(input_one_hot);
  //   val_data_target.push_back(val(val_data[i + 1], mem_pool));
  // }
  // int total = 0;
  // int correct = 0;
  // int total_val_size = val_data_input.size();
  // for (int i = 0; i < total_val_size; i++) {
  //   if (i % BATCH_SIZE == 0) {
  //     mem_pool->reset();
  //     std::cout << "Validating: " << i << "/" << val_data_input.size()
  //               << std::endl;
  //   }
  //   // mem_pool->reset();
  //   auto predicted_prob = softmax(n(val_data_input[i]), mem_pool);
  //   auto out = argmax(predicted_prob, mem_pool);
  //   int predicted_category = mem_pool->get(out)->data;
  //   int expected_category = mem_pool->get(val_data_target[i])->data;
  //   total++;
  //   correct += (predicted_category == expected_category);
  // }
  // auto accuracy = static_cast<float>(correct) / total;
  // std::cout << "Val Total : " << total << std::endl;
  // std::cout << "Val Correct: " << correct << std::endl;
  // std::cout << "Validation accuracy: " << accuracy << std::endl;
}