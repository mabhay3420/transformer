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
  std::vector<int> encode(const std::string &text) const;
  int encode(char c) const;
  std::string decode(const std::vector<int> &encoded) const;
  char decode(const int &encoded) const;
};

#endif