#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct CharTokenizer {
  std::set<char> unique_chars;
  std::unordered_map<char, int> char_to_id;
  std::unordered_map<int, char> id_to_char;

  // must initialize with a set of unique characters
  CharTokenizer(const std::set<char> &chars); 
  void encode(const std::string &text, std::vector<int> &encoded) const ;
  void decode(const std::vector<int> &encoded, std::string &text) const ;
};

#endif