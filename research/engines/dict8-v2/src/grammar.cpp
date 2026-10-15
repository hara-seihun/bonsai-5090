#include "platform.h"
#include "grammar.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

namespace h5::grammar {
namespace {
using Json = nlohmann::ordered_json;
using Stack = std::vector<int>;
using State = std::vector<Stack>;
using Bytes = std::array<uint64_t, 4>;

struct Grammar {
    std::vector<std::vector<Stack>> rules;
    std::vector<Bytes> terminals;
    mutable std::mutex cache_mutex;
    struct Mask { std::vector<uint32_t> words; int allowed; };
    mutable std::unordered_map<std::string, Mask> masks;
    std::function<std::vector<int>(const std::string&)> canonical_encode;
    State string_body_pattern;
    int string_body_rule = -1;

    bool inside_string_body(const State& state) const {
        if (string_body_pattern.empty() || string_body_rule < 0) return false;
        const auto found = std::find_if(string_body_pattern.begin(), string_body_pattern.end(), [](const Stack& s) { return !s.empty(); });
        if (found == string_body_pattern.end()) return false;
        const auto& suffix = *found;
        for (const auto& stack : state) {
            if (stack.size() < suffix.size() || !std::equal(suffix.rbegin(), suffix.rend(), stack.rbegin())) continue;
            Stack seed(stack.begin(), stack.end() - suffix.size());
            seed.push_back(string_body_rule);
            if (close({std::move(seed)}) == state) return true;
        }
        return false;
    }

    int rule() { rules.emplace_back(); return int(rules.size()) - 1; }
    void add(int id, Stack rhs) { rules[id].push_back(std::move(rhs)); }
    int bytes(int lo, int hi) {
        Bytes b{};
        for (int c = lo; c <= hi; ++c) b[c / 64] |= uint64_t(1) << (c % 64);
        terminals.push_back(b);
        return -int(terminals.size());
    }
    int chars(const std::string& text) {
        Bytes b{};
        for (unsigned char c : text) b[c / 64] |= uint64_t(1) << (c % 64);
        terminals.push_back(b);
        return -int(terminals.size());
    }
    int literal(const std::string& text) {
                int r = rule();
        Stack rhs;
        for (unsigned char c : text) rhs.push_back(bytes(c, c));
        add(r, std::move(rhs));
        return r;
    }
    int choice(const std::vector<int>& symbols) {
        int r = rule();
        for (int symbol : symbols) add(r, {symbol});
        return r;
    }
    bool matches(int symbol, unsigned char c) const {
        const auto& b = terminals[-symbol - 1];
        return (b[c / 64] >> (c % 64)) & 1;
    }
    bool prune(int root) {
        std::vector<bool> productive(rules.size(), false);
        bool changed;
        do {
            changed = false;
            for (size_t i = 0; i < rules.size(); ++i) {
                if (productive[i]) continue;
                for (const auto& rhs : rules[i]) {
                    if (std::all_of(rhs.begin(), rhs.end(), [&](int s) { return s < 0 || productive[s]; })) {
                        productive[i] = true;
                        changed = true;
                        break;
                    }
                }
            }
        } while (changed);
        for (auto& alternatives : rules) {
            alternatives.erase(std::remove_if(alternatives.begin(), alternatives.end(), [&](const Stack& rhs) {
                return std::any_of(rhs.begin(), rhs.end(), [&](int s) { return s >= 0 && !productive[s]; });
            }), alternatives.end());
        }
        return productive[root];
    }
    State close(State seeds) const {
        std::set<Stack> visited;
        State result;
        for (size_t i = 0; i < seeds.size(); ++i) {
            Stack stack = std::move(seeds[i]);
            if (!visited.insert(stack).second) continue;
            if (stack.empty() || stack.back() < 0) {
                result.push_back(std::move(stack));
                continue;
            }
            int symbol = stack.back();
            stack.pop_back();
            for (const auto& rhs : rules[symbol]) {
                Stack next = stack;
                next.insert(next.end(), rhs.rbegin(), rhs.rend());
                seeds.push_back(std::move(next));
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }
    std::string deterministic_span(State state) const {
        std::string text;
        for (int n = 0; n < 4096; ++n) {
            if (state.empty() || state.front().empty()) break;
            Bytes legal{};
            for (const auto& stack : state) if (!stack.empty()) {
                const auto& terminal = terminals[-stack.back() - 1];
                for (int w = 0; w < 4; ++w) legal[w] |= terminal[w];
            }
            int count = 0, byte = -1;
            for (int w = 0; w < 4; ++w) {
                count += __builtin_popcountll(legal[w]);
                if (legal[w]) byte = 64 * w + __builtin_ctzll(legal[w]);
            }
            if (count != 1) break;
            text.push_back(char(byte)); state = step(state, byte);
        }
        return text;
    }
    State step(const State& state, unsigned char c) const {
        State next;
        for (const auto& stack : state) {
            if (!stack.empty() && matches(stack.back(), c)) {
                next.push_back(stack);
                next.back().pop_back();
            }
        }
        return close(std::move(next));
    }
};

bool accepting(const State& state) {
    return !state.empty() && state.front().empty();
}

bool equal(const Json& a, const Json& b) {
    if (a.is_object() && b.is_object()) {
        if (a.size() != b.size()) return false;
        for (auto it = a.begin(); it != a.end(); ++it)
            if (!b.contains(it.key()) || !equal(it.value(), b[it.key()])) return false;
        return true;
    }
    if (a.is_array() && b.is_array()) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) if (!equal(a[i], b[i])) return false;
        return true;
    }
    return a == b;
}

std::string state_key(const State& state, bool ended) {
    std::string key(1, ended ? '1' : '0');
    for (const auto& stack : state) {
        uint64_t size = stack.size();
        key.append(reinterpret_cast<const char*>(&size), sizeof(size));
        key.append(reinterpret_cast<const char*>(stack.data()), stack.size() * sizeof(int));
    }
    return key;
}

struct Builder {
    std::shared_ptr<Grammar> g = std::make_shared<Grammar>();
    std::string error;
    int ws, string, body, number, integer, boolean, null, generic;

    Builder() {
        ws = g->rule();
        g->add(ws, {});
        g->add(ws, {g->chars(" \t\r\n"), ws});
        int hex = g->chars("0123456789abcdefABCDEF");
        int scalar = g->rule();
        g->add(scalar, {g->bytes(0x20, 0x21)});
        g->add(scalar, {g->bytes(0x23, 0x5b)});
        g->add(scalar, {g->bytes(0x5d, 0x7f)});
        int cont = g->bytes(0x80, 0xbf);
        g->add(scalar, {g->bytes(0xc2, 0xdf), cont});
        g->add(scalar, {g->bytes(0xe0, 0xe0), g->bytes(0xa0, 0xbf), cont});
        g->add(scalar, {g->bytes(0xe1, 0xec), cont, cont});
        g->add(scalar, {g->bytes(0xed, 0xed), g->bytes(0x80, 0x9f), cont});
        g->add(scalar, {g->bytes(0xee, 0xef), cont, cont});
        g->add(scalar, {g->bytes(0xf0, 0xf0), g->bytes(0x90, 0xbf), cont, cont});
        g->add(scalar, {g->bytes(0xf1, 0xf3), cont, cont, cont});
        g->add(scalar, {g->bytes(0xf4, 0xf4), g->bytes(0x80, 0x8f), cont, cont});
        int slash = g->literal("\\");
        int u = g->literal("\\u");
        g->add(scalar, {slash, g->chars("\"\\/bfnrt")});
        g->add(scalar, {u, g->chars("0123456789abcefABCEF"), hex, hex, hex});
        g->add(scalar, {u, g->chars("dD"), g->chars("01234567"), hex, hex});
        g->add(scalar, {u, g->chars("dD"), g->chars("89abAB"), hex, hex,
                        u, g->chars("dD"), g->chars("cdefCDEF"), hex, hex});
        body = g->rule();
        g->add(body, {});
        g->add(body, {scalar, body});
        string = g->rule();
        int quote = g->literal("\"");
        g->add(string, {quote, body, quote});
        g->string_body_rule = body;
        g->string_body_pattern = g->close({{body}});

        int digit = g->bytes('0', '9');
        int digits = g->rule();
        g->add(digits, {});
        g->add(digits, {digit, digits});
        int positive = g->rule();
        g->add(positive, {g->literal("0")});
        g->add(positive, {g->bytes('1', '9'), digits});
        integer = g->rule();
        g->add(integer, {positive});
        g->add(integer, {g->literal("-"), positive});
        int fraction = g->rule();
        g->add(fraction, {});
        g->add(fraction, {g->literal("."), digit, digits});
        int sign = g->rule();
        g->add(sign, {});
        g->add(sign, {g->chars("+-")});
        int exponent = g->rule();
        g->add(exponent, {});
        g->add(exponent, {g->chars("eE"), sign, digit, digits});
        number = g->rule();
        g->add(number, {integer, fraction, exponent});
        boolean = g->choice({g->literal("true"), g->literal("false")});
        null = g->literal("null");
        generic = g->rule();
        int arr = array(generic, 0, -1);
        int obj = open_object(generic);
        for (int s : {string, number, boolean, null, arr, obj}) g->add(generic, {s});
    }

    bool fail(const std::string& path, const std::string& message) {
        if (error.empty()) error = path + ": " + message;
        return false;
    }
    static bool count(const Json& v) {
        return (v.is_number_unsigned() || v.is_number_integer()) && v.get<int64_t>() >= 0 && v.get<uint64_t>() <= 4096;
    }
    bool literal_value(const Json& value, const std::string& path, int depth) {
        if (depth > 128) return fail(path, "literal nesting exceeds 128");
        if (value.is_binary() || value.is_discarded() ||
            (value.is_number_float() && !std::isfinite(value.get<double>())))
            return fail(path, "literal is not a JSON value");
        if (value.is_array() || value.is_object())
            for (const auto& x : value) if (!literal_value(x, path, depth + 1)) return false;
        return true;
    }
    bool expand(const Json& root, const Json& s, Json& output, const std::string& path,
                std::vector<std::string>& references, int depth = 0) {
        if (depth > 128) return fail(path, "expanded schema nesting exceeds 128");
        if (!s.is_object()) { output = s; return true; }
        if (s.contains("$defs")) {
            if (!s["$defs"].is_object()) return fail(path, "$defs must be an object");
            for (const auto& definition : s["$defs"])
                if (!definition.is_object() && !definition.is_boolean()) return fail(path, "$defs entries must be schemas");
        }
        if (s.contains("$ref")) {
            if (!s["$ref"].is_string()) return fail(path, "$ref must be a string");
            const std::string ref = s["$ref"].get<std::string>();
            if (ref.empty() || ref[0] != '#' || (ref.size() > 1 && ref[1] != '/') || ref.find('%') != std::string::npos)
                return fail(path, "$ref supports only local JSON Pointer fragments without percent encoding");
            for (auto it = s.begin(); it != s.end(); ++it)
                if (it.key() != "$ref" && it.key() != "$defs" && it.key() != "$schema" && it.key() != "title" &&
                    it.key() != "description" && it.key() != "default" && it.key() != "examples" && it.key() != "$comment")
                    return fail(path, "$ref with sibling constraints is unsupported");
            if (std::find(references.begin(), references.end(), ref) != references.end())
                return fail(path, "recursive local $ref is unsupported: " + ref);
            const Json* target;
            try { target = &root.at(Json::json_pointer(ref.substr(1))); }
            catch (const nlohmann::json::exception&) { return fail(path, "unresolved local $ref: " + ref); }
            references.push_back(ref);
            const bool ok = expand(root, *target, output, path + "/$ref(" + ref + ")", references, depth + 1);
            references.pop_back();
            return ok;
        }
        output = s;
        output.erase("$defs");
        if (s.contains("properties") && s["properties"].is_object())
            for (auto it = s["properties"].begin(); it != s["properties"].end(); ++it)
                if (!expand(root, it.value(), output["properties"][it.key()], path + "/properties/" + it.key(), references, depth + 1)) return false;
        if (s.contains("items") && !expand(root, s["items"], output["items"], path + "/items", references, depth + 1)) return false;
        for (const auto& key : {"anyOf", "oneOf"}) if (s.contains(key) && s[key].is_array())
            for (size_t i = 0; i < s[key].size(); ++i)
                if (!expand(root, s[key][i], output[key][i], path + "/" + key + "/" + std::to_string(i), references, depth + 1)) return false;
        return true;
    }
    bool check(const Json& s, const std::string& path, int depth = 0) {
        if (depth > 128) return fail(path, "schema nesting exceeds 128");
        if (s.is_boolean()) return true;
        if (!s.is_object()) return fail(path, "schema must be an object or boolean");
        const std::set<std::string> known = {"type", "properties", "required", "additionalProperties", "items",
            "minItems", "maxItems", "enum", "const", "anyOf", "oneOf", "$schema", "title", "description", "default", "examples", "$comment"};
        const std::set<std::string> types = {"object", "array", "string", "number", "integer", "boolean", "null"};
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (!known.count(it.key())) return fail(path, "unsupported keyword '" + it.key() + "'");
        }
        if (s.contains("type")) {
            const auto& t = s["type"];
            if (t.is_string()) {
                if (!types.count(t.get<std::string>())) return fail(path, "unknown type");
            } else if (t.is_array() && !t.empty()) {
                std::set<std::string> seen;
                for (const auto& x : t) {
                    if (!x.is_string() || !types.count(x.get<std::string>()) || !seen.insert(x.get<std::string>()).second)
                        return fail(path, "type union must contain distinct known types");
                }
            } else return fail(path, "type must be a string or nonempty type array");
        }
        if (s.contains("properties")) {
            if (!s["properties"].is_object()) return fail(path, "properties must be an object");
            for (auto it = s["properties"].begin(); it != s["properties"].end(); ++it)
                if (!check(it.value(), path + "/properties/" + it.key(), depth + 1)) return false;
        }
        if (s.contains("required")) {
            if (!s["required"].is_array()) return fail(path, "required must be an array");
            std::set<std::string> names;
            for (const auto& name : s["required"]) {
                if (!name.is_string() || !names.insert(name.get<std::string>()).second)
                    return fail(path, "required must contain distinct strings");
            }
        }
        if (s.contains("additionalProperties") && !s["additionalProperties"].is_boolean())
            return fail(path, "schema-valued additionalProperties is unsupported");
        if ((s.contains("properties") || s.contains("required")) &&
            (!s.contains("additionalProperties") || s["additionalProperties"] != false))
            return fail(path, "declared properties/required require additionalProperties:false");
        if (s.contains("required")) {
            for (const auto& name : s["required"])
                if (!s.contains("properties") || !s["properties"].contains(name.get<std::string>()))
                    return fail(path, "required property is not declared");
        }
        if (s.contains("items") && !check(s["items"], path + "/items", depth + 1)) return false;
        for (const auto& key : {"minItems", "maxItems"})
            if (s.contains(key) && !count(s[key])) return fail(path, std::string(key) + " must be an integer in [0,4096]");
        if (s.contains("minItems") && s.contains("maxItems") && s["minItems"].get<int>() > s["maxItems"].get<int>())
            return fail(path, "minItems exceeds maxItems");
        if (s.contains("enum")) {
            if (!s["enum"].is_array() || s["enum"].empty()) return fail(path, "enum must be a nonempty array");
            for (const auto& value : s["enum"]) if (!literal_value(value, path + "/enum", 0)) return false;
        }
        if (s.contains("const") && !literal_value(s["const"], path + "/const", 0)) return false;
        for (const auto& union_key : {"anyOf", "oneOf"}) if (s.contains(union_key)) {
            const auto& alternatives = s[union_key];
            if (!alternatives.is_array() || alternatives.empty()) return fail(path, std::string(union_key) + " must be a nonempty array");
            for (const auto& key : {"type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "enum", "const", "anyOf", "oneOf"})
                if (key != std::string(union_key) && s.contains(key)) return fail(path, std::string(union_key) + " with sibling constraints is unsupported");
            for (size_t i = 0; i < alternatives.size(); ++i)
                if (!check(alternatives[i], path + "/" + union_key + "/" + std::to_string(i), depth + 1)) return false;
            if (std::string(union_key) == "oneOf")
                for (size_t i = 0; i < alternatives.size(); ++i)
                    for (size_t j = i + 1; j < alternatives.size(); ++j)
                        if (!disjoint(alternatives[i], alternatives[j]))
                            return fail(path + "/oneOf", "alternatives " + std::to_string(i) + " and " + std::to_string(j) +
                                " overlap or cannot be proven prefix-disjoint; use distinct types, finite values, or required discriminators");
        }
        return true;
    }

    bool has_type(const Json& value, const std::string& type) const {
        if (type == "null") return value.is_null();
        if (type == "boolean") return value.is_boolean();
        if (type == "string") return value.is_string();
        if (type == "array") return value.is_array();
        if (type == "object") return value.is_object();
        if (type == "number") return value.is_number();
        if (type == "integer") return value.is_number() &&
            (value.is_number_integer() || (std::isfinite(value.get<double>()) && std::floor(value.get<double>()) == value.get<double>()));
        return false;
    }
    bool valid(const Json& s, const Json& value) const {
        if (s.is_boolean()) return s.get<bool>();
        if (s.contains("const") && !equal(value, s["const"])) return false;
        if (s.contains("enum") && std::none_of(s["enum"].begin(), s["enum"].end(), [&](const Json& v) { return equal(value, v); })) return false;
        if (s.contains("type")) {
            const auto& t = s["type"];
            if (t.is_string() && !has_type(value, t.get<std::string>())) return false;
            if (t.is_array() && std::none_of(t.begin(), t.end(), [&](const Json& x) { return has_type(value, x.get<std::string>()); })) return false;
        }
        if (s.contains("anyOf") && std::none_of(s["anyOf"].begin(), s["anyOf"].end(), [&](const Json& x) { return valid(x, value); })) return false;
        if (s.contains("oneOf") && std::count_if(s["oneOf"].begin(), s["oneOf"].end(), [&](const Json& x) { return valid(x, value); }) != 1) return false;
        if (value.is_array()) {
            if (s.contains("minItems") && value.size() < s["minItems"].get<size_t>()) return false;
            if (s.contains("maxItems") && value.size() > s["maxItems"].get<size_t>()) return false;
            if (s.contains("items")) for (const auto& x : value) if (!valid(s["items"], x)) return false;
        }
        if (value.is_object()) {
            if (s.contains("required")) for (const auto& k : s["required"]) if (!value.contains(k.get<std::string>())) return false;
            for (auto it = value.begin(); it != value.end(); ++it) {
                if (s.contains("properties") && s["properties"].contains(it.key())) {
                    if (!valid(s["properties"][it.key()], it.value())) return false;
                } else if (s.contains("additionalProperties") && s["additionalProperties"] == false) return false;
            }
        }
        return true;
    }

    static unsigned type_mask(const Json& s) {
        if (s.is_boolean()) return s.get<bool>() ? 63 : 0;
        if (!s.contains("type")) return 63;
        auto bit = [](const std::string& t) -> unsigned {
            if (t == "object") return 1;
            if (t == "array") return 2;
            if (t == "string") return 4;
            if (t == "number" || t == "integer") return 8;
            if (t == "boolean") return 16;
            if (t == "null") return 32;
            return 0;
        };
        if (s["type"].is_string()) return bit(s["type"].get<std::string>());
        unsigned mask = 0;
        for (const auto& t : s["type"]) mask |= bit(t.get<std::string>());
        return mask;
    }
    bool finite_values(const Json& s, std::vector<Json>& values) const {
        if (s.is_boolean()) return !s.get<bool>();
        if (s.contains("const")) {
            if (valid(s, s["const"])) values.push_back(s["const"]);
            return true;
        }
        if (!s.contains("enum")) return false;
        for (const auto& v : s["enum"]) if (valid(s, v)) values.push_back(v);
        return true;
    }
    // Prove semantic separation, not merely different canonical spellings.
    // Unknown intersections are rejected, so union compilation is exact oneOf.
    bool disjoint(const Json& a, const Json& b) const {
        std::vector<Json> values;
        if (finite_values(a, values))
            return std::none_of(values.begin(), values.end(), [&](const Json& v) { return valid(b, v); });
        if (finite_values(b, values))
            return std::none_of(values.begin(), values.end(), [&](const Json& v) { return valid(a, v); });
        for (const auto& key : {"anyOf", "oneOf"}) {
            if (a.is_object() && a.contains(key))
                return std::all_of(a[key].begin(), a[key].end(), [&](const Json& x) { return disjoint(x, b); });
            if (b.is_object() && b.contains(key))
                return std::all_of(b[key].begin(), b[key].end(), [&](const Json& x) { return disjoint(a, x); });
        }
        const unsigned shared = type_mask(a) & type_mask(b);
        if (!shared) return true;
        if (shared == 1) {
            auto separates = [&](const Json& left, const Json& right) {
                if (!left.is_object() || !left.contains("required") || !right.is_object()) return false;
                for (const auto& name : left["required"]) {
                    const std::string k = name.get<std::string>();
                    if (!right.contains("properties") || !right["properties"].contains(k)) {
                        if (right.contains("additionalProperties") && right["additionalProperties"] == false) return true;
                    } else if (disjoint(left["properties"][k], right["properties"][k])) return true;
                }
                return false;
            };
            return separates(a, b) || separates(b, a);
        }
        if (shared == 2) {
            auto bound = [](const Json& s, const char* key, int absent) {
                return s.is_object() && s.contains(key) ? s[key].get<int>() : absent;
            };
            const int min_a = bound(a, "minItems", 0), min_b = bound(b, "minItems", 0);
            if (min_a > bound(b, "maxItems", 4097) || min_b > bound(a, "maxItems", 4097)) return true;
            if (std::max(min_a, min_b) > 0 && a.is_object() && b.is_object() && a.contains("items") && b.contains("items"))
                return disjoint(a["items"], b["items"]);
        }
        return false;
    }

    int array(int item, int minimum, int maximum) {
        int r = g->rule();
        int tail = g->rule();
        int comma = g->literal(",");
        int close = g->literal("]");
        if (maximum < 0) {
            g->add(tail, {close});
            g->add(tail, {comma, ws, item, ws, tail});
        } else {
            // The continuation is indexed by the number already emitted.
            std::vector<int> tails(size_t(maximum) + 1);
            tails[maximum] = tail;
            for (int n = maximum - 1; n >= 0; --n) tails[n] = g->rule();
            for (int n = 0; n <= maximum; ++n) {
                if (n >= minimum) g->add(tails[n], {close});
                if (n < maximum) g->add(tails[n], {comma, ws, item, ws, tails[n + 1]});
            }
            if (minimum == 0) g->add(r, {g->literal("["), ws, close});
            if (maximum > 0) g->add(r, {g->literal("["), ws, item, ws, tails[1]});
            return r;
        }
        Stack rhs{g->literal("["), ws};
        if (minimum == 0) g->add(r, {rhs[0], ws, close});
        int required = std::max(minimum, 1);
        for (int n = 0; n < required; ++n) {
            if (n) rhs.insert(rhs.end(), {comma, ws});
            rhs.insert(rhs.end(), {item, ws});
        }
        rhs.push_back(tail);
        g->add(r, std::move(rhs));
        return r;
    }
    int open_object(int item) {
        int r = g->rule(), tail = g->rule();
        int close = g->literal("}"), colon = g->literal(":"), comma = g->literal(",");
        g->add(r, {g->literal("{"), ws, close});
        g->add(r, {g->literal("{"), ws, string, ws, colon, ws, item, ws, tail});
        g->add(tail, {close});
        g->add(tail, {comma, ws, string, ws, colon, ws, item, ws, tail});
        return r;
    }
    int object(const Json& s) {
        if (!s.contains("additionalProperties") || s["additionalProperties"] != false) return open_object(generic);
        int r = g->rule();
        std::vector<std::pair<std::string, int>> props;
        std::set<std::string> required;
        if (s.contains("required")) for (const auto& k : s["required"]) required.insert(k.get<std::string>());
        if (s.contains("properties")) for (auto it = s["properties"].begin(); it != s["properties"].end(); ++it)
            props.emplace_back(it.key(), build(it.value()));
        std::vector<std::array<int, 2>> suffix(props.size() + 1);
        for (auto& pair : suffix) pair = {g->rule(), g->rule()};
        int close = g->literal("}"), comma = g->literal(","), colon = g->literal(":");
        for (int emitted = 0; emitted < 2; ++emitted) g->add(suffix.back()[emitted], {close});
        for (size_t n = 0; n < props.size(); ++n) {
            int key = g->literal(Json(props[n].first).dump());
            for (int emitted = 0; emitted < 2; ++emitted) {
                if (!required.count(props[n].first)) g->add(suffix[n][emitted], {suffix[n + 1][emitted]});
                Stack rhs;
                if (emitted) rhs.insert(rhs.end(), {comma, ws});
                rhs.insert(rhs.end(), {key, ws, colon, ws, props[n].second, ws, suffix[n + 1][1]});
                g->add(suffix[n][emitted], std::move(rhs));
            }
        }
        g->add(r, {g->literal("{"), ws, suffix[0][0]});
        return r;
    }
    int build(const Json& s) {
        if (s.is_boolean()) return s.get<bool>() ? generic : g->rule();
        if (s.contains("enum") || s.contains("const")) {
            int r = g->rule();
            auto add = [&](const Json& value) { if (valid(s, value)) g->add(r, {g->literal(value.dump())}); };
            if (s.contains("const")) add(s["const"]);
            else for (const auto& value : s["enum"]) add(value);
            return r;
        }
        for (const auto& key : {"anyOf", "oneOf"}) if (s.contains(key)) {
            int r = g->rule();
            for (const auto& x : s[key]) g->add(r, {build(x)});
            return r;
        }
        std::vector<std::string> types;
        if (s.contains("type")) {
            if (s["type"].is_string()) types.push_back(s["type"].get<std::string>());
            else for (const auto& t : s["type"]) types.push_back(t.get<std::string>());
        } else types = {"object", "array", "string", "number", "boolean", "null"};
        int r = g->rule();
        for (const auto& t : types) {
            int symbol;
            if (t == "object") symbol = object(s);
            else if (t == "array") symbol = array(s.contains("items") ? build(s["items"]) : generic,
                s.contains("minItems") ? s["minItems"].get<int>() : 0,
                s.contains("maxItems") ? s["maxItems"].get<int>() : -1);
            else if (t == "string") symbol = string;
            else if (t == "number") symbol = number;
            else if (t == "integer") symbol = integer;
            else if (t == "boolean") symbol = boolean;
            else symbol = null;
            g->add(r, {symbol});
        }
        return r;
    }
};
} // namespace

struct Vocabulary::Impl {
    struct Node {
        std::vector<std::pair<unsigned char, int>> children;
        std::vector<int> tokens;
    };
    std::vector<std::string> pieces;
    std::vector<bool> eos;
    std::string error;
    std::vector<Node> trie{1};
    std::vector<int> boundary_tokens;
    std::vector<uint32_t> string_content_mask;
    int string_content_count = 0;
};

Vocabulary::Vocabulary(std::vector<std::string> pieces, std::vector<int> eos) {
    auto impl = std::make_shared<Impl>();
    impl->pieces = std::move(pieces);
    impl->eos.resize(impl->pieces.size());
    for (int id : eos) {
        if (id < 0 || size_t(id) >= impl->eos.size()) impl->error = "EOS token ID is outside vocabulary";
        else impl->eos[id] = true;
    }
    for (size_t id = 0; id < impl->pieces.size(); ++id) {
        if (impl->eos[id] || impl->pieces[id].empty()) continue;
        int node = 0;
        for (unsigned char c : impl->pieces[id]) {
            auto& children = impl->trie[node].children;
            auto it = std::find_if(children.begin(), children.end(), [&](auto edge) { return edge.first == c; });
            if (it != children.end()) node = it->second;
            else {
                int next = int(impl->trie.size());
                children.emplace_back(c, next);
                impl->trie.emplace_back();
                node = next;
            }
        }
        impl->trie[node].tokens.push_back(int(id));
    }
    // Pure string content cannot see the schema continuation. Prepare its exact
    // lexical mask once; only quote-containing tokens need the full parser.
    Builder lexical;
    const State initial = lexical.g->string_body_pattern;
    std::vector<State> states{initial};
    std::vector<std::array<int, 256>> transitions(1);
    transitions[0].fill(-2);
    std::unordered_map<std::string, int> ids{{state_key(initial, false), 0}};
    impl->string_content_mask.assign((impl->pieces.size() + 31) / 32, 0);
    for (size_t id = 0; id < impl->pieces.size(); ++id) {
        if (impl->eos[id] || impl->pieces[id].empty()) continue;
        if (impl->pieces[id].find('"') != std::string::npos) { impl->boundary_tokens.push_back(int(id)); continue; }
        int current = 0;
        for (unsigned char c : impl->pieces[id]) {
            int next = transitions[current][c];
            if (next == -2) {
                State advanced = lexical.g->step(states[current], c);
                next = -1;
                if (!advanced.empty()) {
                    auto insertion = ids.emplace(state_key(advanced, false), int(states.size()));
                    next = insertion.first->second;
                    if (insertion.second) {
                        states.push_back(std::move(advanced)); transitions.emplace_back(); transitions.back().fill(-2);
                    }
                }
                transitions[current][c] = next;
            }
            current = next;
            if (current < 0) break;
        }
        if (current >= 0) {
            impl->string_content_mask[id / 32] |= uint32_t(1) << (id % 32); ++impl->string_content_count;
        }
    }
    impl_ = std::move(impl);
}
size_t Vocabulary::size() const { return impl_->pieces.size(); }

struct Matcher::Impl {
    std::shared_ptr<const Grammar> grammar;
    std::shared_ptr<const Vocabulary> vocabulary;
    State state;
    bool ended = false;
};
Matcher::Matcher(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Matcher::~Matcher() = default;
Matcher::Matcher(const Matcher& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}
Matcher& Matcher::operator=(const Matcher& other) {
    if (this != &other) impl_ = std::make_unique<Impl>(*other.impl_);
    return *this;
}
Matcher::Matcher(Matcher&&) noexcept = default;
Matcher& Matcher::operator=(Matcher&&) noexcept = default;
bool Matcher::complete() const { return accepting(impl_->state); }
bool Matcher::accept(int token) {
    const auto& v = *impl_->vocabulary->impl_;
    if (token < 0 || size_t(token) >= v.pieces.size()) return false;
    if (impl_->ended) return v.eos[token];
    if (v.eos[token]) {
        if (!complete()) return false;
        impl_->ended = true;
        return true;
    }
    if (v.pieces[token].empty()) return false;
    if (impl_->grammar->canonical_encode) {
        std::vector<uint32_t> mask((v.pieces.size() + 31) / 32);
        if (this->mask(mask.data()) <= 0 || !(mask[token / 32] & (uint32_t(1) << (token % 32)))) return false;
    }
        State next = impl_->state;
    for (unsigned char c : v.pieces[token]) {
        next = impl_->grammar->step(next, c);
        if (next.empty()) break;
    }
        if (next.empty()) return false;
    impl_->state = std::move(next);
    return true;
}
int Matcher::mask(uint32_t* words) const {
    const auto& g = *impl_->grammar;
    const auto& v = *impl_->vocabulary->impl_;
    const size_t nwords = (v.pieces.size() + 31) / 32;
    std::string key = state_key(impl_->state, impl_->ended);
    {
        std::lock_guard<std::mutex> lock(g.cache_mutex);
        auto found = g.masks.find(key);
        if (found != g.masks.end()) {
            std::copy(found->second.words.begin(), found->second.words.end(), words);
            return found->second.allowed;
        }
    }
    std::vector<uint32_t> result(nwords, 0);
    int allowed = 0;
    if (!impl_->ended && g.canonical_encode) {
        const std::string span = g.deterministic_span(impl_->state);
        if (!span.empty()) {
            std::vector<int> encoded;
            try { encoded = g.canonical_encode(span); } catch (const std::exception&) { return -1; }
            if (encoded.empty()) return -1;
            std::string reproduced;
            for (int token : encoded) {
                if (token < 0 || size_t(token) >= v.pieces.size() || v.eos[token] || v.pieces[token].empty()) return -1;
                reproduced += v.pieces[token];
            }
            if (reproduced != span) return -1;
            const int token = encoded.front();
            result[token / 32] |= uint32_t(1) << (token % 32);
            std::copy(result.begin(), result.end(), words);
            std::lock_guard<std::mutex> lock(g.cache_mutex);
            if (g.masks.size() < 4096) g.masks.emplace(std::move(key), Grammar::Mask{std::move(result), 1});
            return 1;
        }
    }
    auto allow = [&](int id) {
        const uint32_t bit = uint32_t(1) << (id % 32);
        if (!(result[id / 32] & bit)) { result[id / 32] |= bit; ++allowed; }
    };
    if (complete()) for (size_t id = 0; id < v.eos.size(); ++id) if (v.eos[id]) allow(int(id));
    const bool byte_path = std::any_of(impl_->state.begin(), impl_->state.end(), [](const Stack& stack) {
        return !stack.empty();
    });
    if (!impl_->ended && byte_path) {
        // Determinize only states reached by vocabulary pieces. Thousands of
        // trie edges inside a string share the same byte transitions.
        std::vector<State> states{impl_->state};
        std::vector<std::array<int, 256>> transitions(1);
        transitions[0].fill(-2);
        std::unordered_map<std::string, int> ids{{state_key(impl_->state, false), 0}};
        auto transition = [&](int current, unsigned char c) {
            int next = transitions[current][c];
            if (next != -2) return next;
            State advanced = g.step(states[current], c);
            next = -1;
            if (!advanced.empty()) {
                auto insertion = ids.emplace(state_key(advanced, false), int(states.size()));
                next = insertion.first->second;
                if (insertion.second) {
                    states.push_back(std::move(advanced)); transitions.emplace_back(); transitions.back().fill(-2);
                }
            }
            transitions[current][c] = next;
            return next;
        };
        const bool string_body = g.inside_string_body(impl_->state);
        if (string_body) {
            result = v.string_content_mask;
            allowed = v.string_content_count;
            for (int token : v.boundary_tokens) {
                int current = 0;
                for (unsigned char c : v.pieces[token]) {
                    current = transition(current, c);
                    if (current < 0) break;
                }
                if (current >= 0) allow(token);
            }
        }
        std::vector<std::pair<int, int>> pending;
        if (!string_body) pending.emplace_back(0, 0);
        while (!pending.empty()) {
            auto current = pending.back();
            pending.pop_back();
            int node = current.first, state_id = current.second;
            for (int id : v.trie[node].tokens) allow(id);
            for (auto edge : v.trie[node].children) {
                const int next_id = transition(state_id, edge.first);
                if (next_id >= 0) pending.emplace_back(edge.second, next_id);
            }
        }
    }
        std::copy(result.begin(), result.end(), words);
    {
        std::lock_guard<std::mutex> lock(g.cache_mutex);
        // Cache is shared by forks and bounded independently of output length.
        if (g.masks.size() < 4096) g.masks.emplace(std::move(key), Grammar::Mask{std::move(result), allowed});
    }
    return allowed;
}

struct CompilerAccess {
    static Result run(const Json& schema, std::shared_ptr<const Vocabulary> vocabulary, const CompileOptions& options) {
        if (!vocabulary) return {nullptr, "vocabulary is unset"};
        if (vocabulary->size() == 0) return {nullptr, "vocabulary is empty"};
        if (!vocabulary->impl_->error.empty()) return {nullptr, vocabulary->impl_->error};
        Builder builder;
        if (options.mode == Mode::CanonicalFixed) {
            if (!options.encode) return {nullptr, "canonical_fixed requires tokenizer encoder"};
            // The ordinary byte grammar remains the decoding law at every
            // ambiguous byte state. Canonical tokenization only replaces a
            // already-forced byte span; no boundary competes with payload tokens.
            const auto probe = options.encode("{}");
            std::string actual;
            for (int token : probe) {
                if (token < 0 || size_t(token) >= vocabulary->size() || vocabulary->impl_->eos[token] || vocabulary->impl_->pieces[token].empty())
                    return {nullptr, "canonical encoder returned disabled/out-of-vocabulary token"};
                actual += vocabulary->impl_->pieces[token];
            }
            if (actual != "{}") return {nullptr, "canonical encoder token pieces do not reproduce fixed literal"};
            builder.g->canonical_encode = options.encode;
        } else if (options.mode != Mode::Ordinary) return {nullptr, "unknown grammar compile mode"};
        Json expanded;
        std::vector<std::string> references;
        if (!builder.expand(schema, schema, expanded, "#", references) || !builder.check(expanded, "#"))
            return {nullptr, builder.error};
        int root = builder.g->rule();
        builder.g->add(root, {builder.ws, builder.build(expanded), builder.ws});
                if (!builder.g->prune(root)) return {nullptr, "schema has no permitted output"};
        auto impl = std::make_unique<Matcher::Impl>();
        impl->grammar = builder.g;
        impl->vocabulary = std::move(vocabulary);
        impl->state = builder.g->close({{root}});
        return {std::unique_ptr<Matcher>(new Matcher(std::move(impl))), ""};
    }
};
Result compile(const Json& schema, std::shared_ptr<const Vocabulary> vocabulary) {
    return compile(schema, std::move(vocabulary), CompileOptions{});
}
Result compile(const Json& schema, std::shared_ptr<const Vocabulary> vocabulary, const CompileOptions& options) {
    try { return CompilerAccess::run(schema, std::move(vocabulary), options); }
    catch (const nlohmann::json::exception& e) { return {nullptr, std::string("invalid schema JSON: ") + e.what()}; }
    catch (const std::exception& e) { return {nullptr, std::string("grammar compilation failed: ") + e.what()}; }
}
} // namespace h5::grammar
