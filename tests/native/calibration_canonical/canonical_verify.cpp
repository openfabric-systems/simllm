#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

struct Integer {
  std::string lexeme;
};

struct Value {
  using Array = std::vector<Value>;
  using Object = std::vector<std::pair<std::string, Value>>;
  using Data = std::variant<std::nullptr_t, bool, Integer, std::string, Array, Object>;

  explicit Value(Data value) : data(std::move(value)) {}
  Data data;
};

class ParseError : public std::runtime_error {
 public:
  ParseError(std::size_t offset, const std::string& message)
      : std::runtime_error("byte " + std::to_string(offset) + ": " + message) {}
};

class Parser {
 public:
  explicit Parser(std::string_view input) : input_(input) {
    for (std::size_t index = 0; index < input_.size(); ++index) {
      if (static_cast<unsigned char>(input_[index]) >= 0x80U) {
        throw ParseError(index, "non-ASCII input is outside the conformance subset");
      }
    }
  }

  Value Parse() {
    SkipWhitespace();
    Value result = ParseValue();
    SkipWhitespace();
    if (position_ != input_.size()) {
      throw ParseError(position_, "trailing input");
    }
    return result;
  }

 private:
  void SkipWhitespace() {
    while (position_ < input_.size()) {
      const char ch = input_[position_];
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
        return;
      }
      ++position_;
    }
  }

  char Peek() const {
    if (position_ >= input_.size()) {
      throw ParseError(position_, "unexpected end of input");
    }
    return input_[position_];
  }

  char Take() {
    const char ch = Peek();
    ++position_;
    return ch;
  }

  void Expect(char expected) {
    if (Take() != expected) {
      throw ParseError(position_ - 1, std::string("expected '") + expected + "'");
    }
  }

  bool Consume(char expected) {
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void ConsumeLiteral(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      throw ParseError(position_, "invalid literal");
    }
    position_ += literal.size();
  }

  Value ParseValue() {
    SkipWhitespace();
    switch (Peek()) {
      case 'n':
        ConsumeLiteral("null");
        return Value(nullptr);
      case 't':
        ConsumeLiteral("true");
        return Value(true);
      case 'f':
        ConsumeLiteral("false");
        return Value(false);
      case '"':
        return Value(ParseString());
      case '[':
        return Value(ParseArray());
      case '{':
        return Value(ParseObject());
      default:
        if (Peek() == '-' || (Peek() >= '0' && Peek() <= '9')) {
          return Value(ParseInteger());
        }
        throw ParseError(position_, "expected a JSON value");
    }
  }

  Integer ParseInteger() {
    const std::size_t start = position_;
    const bool negative = Consume('-');
    if (position_ >= input_.size()) {
      throw ParseError(position_, "incomplete integer");
    }
    if (input_[position_] == '0') {
      ++position_;
      if (negative) {
        throw ParseError(start, "negative zero is not canonical");
      }
      if (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
        throw ParseError(start, "leading zero is not canonical");
      }
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        throw ParseError(position_, "invalid integer");
      }
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < input_.size() &&
        (input_[position_] == '.' || input_[position_] == 'e' || input_[position_] == 'E')) {
      throw ParseError(position_, "floating-point numbers are forbidden");
    }
    return Integer{std::string(input_.substr(start, position_ - start))};
  }

  static int HexDigit(char ch) {
    if (ch >= '0' && ch <= '9') {
      return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
      return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
      return ch - 'A' + 10;
    }
    return -1;
  }

  char ParseUnicodeEscape() {
    unsigned value = 0;
    for (int index = 0; index < 4; ++index) {
      if (position_ >= input_.size()) {
        throw ParseError(position_, "incomplete Unicode escape");
      }
      const int digit = HexDigit(input_[position_++]);
      if (digit < 0) {
        throw ParseError(position_ - 1, "invalid Unicode escape");
      }
      value = (value << 4U) | static_cast<unsigned>(digit);
    }
    if (value > 0x7fU) {
      throw ParseError(position_ - 4, "non-ASCII string value is outside the subset");
    }
    return static_cast<char>(value);
  }

  std::string ParseString() {
    Expect('"');
    std::string value;
    while (true) {
      if (position_ >= input_.size()) {
        throw ParseError(position_, "unterminated string");
      }
      const unsigned char raw = static_cast<unsigned char>(Take());
      if (raw == '"') {
        return value;
      }
      if (raw < 0x20U) {
        throw ParseError(position_ - 1, "unescaped control character");
      }
      if (raw != '\\') {
        value.push_back(static_cast<char>(raw));
        continue;
      }
      if (position_ >= input_.size()) {
        throw ParseError(position_, "incomplete escape");
      }
      switch (Take()) {
        case '"':
          value.push_back('"');
          break;
        case '\\':
          value.push_back('\\');
          break;
        case '/':
          value.push_back('/');
          break;
        case 'b':
          value.push_back('\b');
          break;
        case 'f':
          value.push_back('\f');
          break;
        case 'n':
          value.push_back('\n');
          break;
        case 'r':
          value.push_back('\r');
          break;
        case 't':
          value.push_back('\t');
          break;
        case 'u':
          value.push_back(ParseUnicodeEscape());
          break;
        default:
          throw ParseError(position_ - 1, "invalid escape");
      }
    }
  }

  Value::Array ParseArray() {
    Expect('[');
    SkipWhitespace();
    Value::Array result;
    if (Consume(']')) {
      return result;
    }
    while (true) {
      result.push_back(ParseValue());
      SkipWhitespace();
      if (Consume(']')) {
        return result;
      }
      Expect(',');
      SkipWhitespace();
    }
  }

  Value::Object ParseObject() {
    Expect('{');
    SkipWhitespace();
    Value::Object result;
    std::set<std::string> keys;
    if (Consume('}')) {
      return result;
    }
    while (true) {
      if (Peek() != '"') {
        throw ParseError(position_, "object key must be a string");
      }
      std::string key = ParseString();
      if (!keys.insert(key).second) {
        throw ParseError(position_, "duplicate object key");
      }
      SkipWhitespace();
      Expect(':');
      result.emplace_back(std::move(key), ParseValue());
      SkipWhitespace();
      if (Consume('}')) {
        return result;
      }
      Expect(',');
      SkipWhitespace();
    }
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

void WriteString(std::ostream& output, std::string_view value) {
  static constexpr char kHex[] = "0123456789abcdef";
  output.put('"');
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (ch < 0x20U) {
          output << "\\u00" << kHex[(ch >> 4U) & 0x0fU] << kHex[ch & 0x0fU];
        } else {
          output.put(static_cast<char>(ch));
        }
    }
  }
  output.put('"');
}

void WriteCanonical(std::ostream& output, const Value& value);

void WriteArray(std::ostream& output, const Value::Array& values) {
  output.put('[');
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output.put(',');
    }
    WriteCanonical(output, values[index]);
  }
  output.put(']');
}

void WriteObject(std::ostream& output, const Value::Object& object) {
  std::vector<const std::pair<std::string, Value>*> members;
  members.reserve(object.size());
  for (const auto& member : object) {
    members.push_back(&member);
  }
  std::sort(members.begin(), members.end(), [](const auto* left, const auto* right) {
    return left->first < right->first;
  });
  output.put('{');
  for (std::size_t index = 0; index < members.size(); ++index) {
    if (index != 0) {
      output.put(',');
    }
    WriteString(output, members[index]->first);
    output.put(':');
    WriteCanonical(output, members[index]->second);
  }
  output.put('}');
}

void WriteCanonical(std::ostream& output, const Value& value) {
  std::visit(
      [&output](const auto& item) {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::nullptr_t>) {
          output << "null";
        } else if constexpr (std::is_same_v<T, bool>) {
          output << (item ? "true" : "false");
        } else if constexpr (std::is_same_v<T, Integer>) {
          output << item.lexeme;
        } else if constexpr (std::is_same_v<T, std::string>) {
          WriteString(output, item);
        } else if constexpr (std::is_same_v<T, Value::Array>) {
          WriteArray(output, item);
        } else {
          WriteObject(output, item);
        }
      },
      value.data);
}

constexpr std::array<std::uint32_t, 64> kSha256Constants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t RotateRight(std::uint32_t value, unsigned amount) {
  return (value >> amount) | (value << (32U - amount));
}

std::string Sha256(std::string_view input) {
  std::vector<std::uint8_t> message(input.begin(), input.end());
  const std::uint64_t bit_length = static_cast<std::uint64_t>(message.size()) * 8U;
  message.push_back(0x80U);
  while ((message.size() % 64U) != 56U) {
    message.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<std::uint8_t>((bit_length >> shift) & 0xffU));
  }

  std::array<std::uint32_t, 8> state = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                                 0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                                 0x1f83d9abU, 0x5be0cd19U};
  for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t base = offset + index * 4U;
      words[index] = (static_cast<std::uint32_t>(message[base]) << 24U) |
                     (static_cast<std::uint32_t>(message[base + 1]) << 16U) |
                     (static_cast<std::uint32_t>(message[base + 2]) << 8U) |
                     static_cast<std::uint32_t>(message[base + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t small0 = RotateRight(words[index - 15], 7U) ^
                                   RotateRight(words[index - 15], 18U) ^
                                   (words[index - 15] >> 3U);
      const std::uint32_t small1 = RotateRight(words[index - 2], 17U) ^
                                   RotateRight(words[index - 2], 19U) ^
                                   (words[index - 2] >> 10U);
      words[index] = words[index - 16] + small0 + words[index - 7] + small1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t big1 = RotateRight(e, 6U) ^ RotateRight(e, 11U) ^
                                 RotateRight(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + big1 + choose + kSha256Constants[index] + words[index];
      const std::uint32_t big0 = RotateRight(a, 2U) ^ RotateRight(a, 13U) ^
                                 RotateRight(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = big0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  std::ostringstream digest;
  digest << std::hex << std::setfill('0');
  for (const std::uint32_t word : state) {
    digest << std::setw(8) << word;
  }
  return digest.str();
}

std::string ReadFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open input file");
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    throw std::runtime_error("cannot read input file");
  }
  return contents.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: calibration_canonical_verify INPUT.json\n";
    return 2;
  }
  try {
    const Value value = Parser(ReadFile(argv[1])).Parse();
    std::ostringstream canonical;
    WriteCanonical(canonical, value);
    const std::string bytes = canonical.str();
    std::cout << bytes << '\n' << Sha256(bytes) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
