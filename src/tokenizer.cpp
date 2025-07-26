#include "tokenizer.hpp"
#include <stdexcept>

// must initialize with a set of unique characters
CharTokenizer::CharTokenizer(const std::set<char> &chars)
    : unique_chars(chars) {
  int id = 0;
  for (char c : unique_chars) {
    char_to_id[c] = id;
    id_to_char[id] = c;
    id++;
  }
}

void CharTokenizer::encode(const std::string &text,
                           std::vector<int> &encoded) const {
  encoded.clear();
  for (char c : text) {
    auto it = char_to_id.find(c);
    if (it != char_to_id.end()) {
      encoded.push_back(it->second);
    } else {
      throw std::out_of_range("Character not in tokenizer");
    }
  }
}

void CharTokenizer::decode(const std::vector<int> &encoded,
                           std::string &text) const {
  text.clear();
  for (int id : encoded) {
    auto it = id_to_char.find(id);
    if (it != id_to_char.end()) {
      text.push_back(it->second);
    } else {
      throw std::out_of_range("Encoded value out of range");
    }
  }
}
;