#include "platform.h"
#include "gpu_grammar.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace h5::gpu_grammar {
namespace {
using Json = nlohmann::ordered_json;
using ByteSet = std::array<uint64_t, 4>;
using Subset = std::vector<uint32_t>;

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

struct Edge { ByteSet bytes; uint32_t destination; };
struct Node { std::vector<uint32_t> epsilon; std::vector<Edge> edges; };
struct Fragment { uint32_t first, last; };

struct Compiler {
    CompileError error = CompileError::None;
    std::string detail;
    std::vector<Node> nodes;
    uint32_t budget;
    size_t nfa_budget;
    size_t expansion_nodes = 0;

    explicit Compiler(uint32_t limit) : budget(limit), nfa_budget(std::min<uint64_t>(uint64_t(limit)*16, 1048576)) {}
    bool fail(const std::string& path, const std::string& message, CompileError kind = CompileError::InvalidSchema) {
        if (error == CompileError::None) { error = kind; detail = path + ": " + message; }
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
        if (++expansion_nodes > nfa_budget) return fail(path, "schema expansion exceeds construction budget", CompileError::StateBudget);
        if (depth > 128) return fail(path, "expanded schema nesting exceeds 128");
        if (!s.is_object()) { output = s; return true; }
        if (s.contains("$defs")) {
            if (!s["$defs"].is_object()) return fail(path, "$defs must be an object");
            for (const auto& definition : s["$defs"])
                if (!definition.is_object() && !definition.is_boolean()) return fail(path, "$defs entries must be schemas");
        }
        if (s.contains("$ref")) {
            if (!s["$ref"].is_string()) return fail(path, "$ref must be a string", CompileError::InvalidReference);
            const std::string ref = s["$ref"].get<std::string>();
            if (ref.empty() || ref[0] != '#' || (ref.size() > 1 && ref[1] != '/') || ref.find('%') != std::string::npos)
                return fail(path, "$ref supports only local JSON Pointer fragments without percent encoding", CompileError::InvalidReference);
            for (auto it = s.begin(); it != s.end(); ++it)
                if (it.key() != "$ref" && it.key() != "$defs" && it.key() != "$schema" && it.key() != "title" &&
                    it.key() != "description" && it.key() != "default" && it.key() != "examples" && it.key() != "$comment")
                    return fail(path, "$ref with sibling constraints is unsupported", CompileError::UnsupportedKeyword);
            if (std::find(references.begin(), references.end(), ref) != references.end())
                return fail(path, "recursive local $ref is unsupported: " + ref, CompileError::InvalidReference);
            const Json* target;
            try { target = &root.at(Json::json_pointer(ref.substr(1))); }
            catch (const nlohmann::json::exception&) { return fail(path, "unresolved local $ref: " + ref, CompileError::InvalidReference); }
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
            if (!known.count(it.key())) return fail(path, "unsupported keyword '" + it.key() + "'", CompileError::UnsupportedKeyword);
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
            return fail(path, "schema-valued additionalProperties is unsupported", CompileError::UnsupportedKeyword);
        if ((s.contains("properties") || s.contains("required")) &&
            (!s.contains("additionalProperties") || s["additionalProperties"] != false))
            return fail(path, "declared properties/required require additionalProperties:false", CompileError::UnsupportedKeyword);
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
                if (key != std::string(union_key) && s.contains(key)) return fail(path, std::string(union_key) + " with sibling constraints is unsupported", CompileError::UnsupportedKeyword);
            for (size_t i = 0; i < alternatives.size(); ++i)
                if (!check(alternatives[i], path + "/" + union_key + "/" + std::to_string(i), depth + 1)) return false;
            if (std::string(union_key) == "oneOf")
                for (size_t i = 0; i < alternatives.size(); ++i)
                    for (size_t j = i + 1; j < alternatives.size(); ++j)
                        if (!disjoint(alternatives[i], alternatives[j]))
                            return fail(path + "/oneOf", "alternatives " + std::to_string(i) + " and " + std::to_string(j) +
                                " overlap or cannot be proven disjoint; use distinct types, finite values, or required discriminators", CompileError::OverlappingOneOf);
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

    uint32_t node() {
        if (error != CompileError::None) return INVALID;
        if (nodes.size() >= nfa_budget) {
            fail("#", "NFA construction exceeds budget", CompileError::StateBudget);
            return INVALID;
        }
        nodes.emplace_back();
        return static_cast<uint32_t>(nodes.size()-1);
    }
    Fragment blank() { return {node(), node()}; }
    void epsilon(uint32_t from, uint32_t to) {
        if (from != INVALID && to != INVALID) nodes[from].epsilon.push_back(to);
    }
    Fragment empty() {
        auto f = blank(); epsilon(f.first, f.last); return f;
    }
    Fragment chars(const std::string& text) {
        auto f = blank();
        ByteSet bits{};
        for (unsigned char c : text) bits[c/64] |= uint64_t(1) << (c%64);
        if (f.first != INVALID && f.last != INVALID) nodes[f.first].edges.push_back({bits, f.last});
        return f;
    }
    Fragment range(unsigned lo, unsigned hi) {
        std::string text;
        for (unsigned c = lo; c <= hi; ++c) text.push_back(static_cast<char>(c));
        return chars(text);
    }
    Fragment literal(const std::string& text) {
        auto first = node(), last = first;
        for (unsigned char c : text) {
            auto next = node();
            if (last != INVALID && next != INVALID) {
                ByteSet bits{}; bits[c/64] = uint64_t(1) << (c%64);
                nodes[last].edges.push_back({bits, next});
            }
            last = next;
        }
        return {first, last};
    }
    Fragment sequence(std::vector<Fragment> pieces) {
        if (pieces.empty()) return empty();
        for (size_t i = 1; i < pieces.size(); ++i) epsilon(pieces[i-1].last, pieces[i].first);
        return {pieces.front().first, pieces.back().last};
    }
    Fragment choice(std::vector<Fragment> pieces) {
        auto f = blank();
        for (auto p : pieces) { epsilon(f.first, p.first); epsilon(p.last, f.last); }
        return f;
    }
    Fragment star(Fragment body) {
        auto f = blank(); epsilon(f.first, f.last); epsilon(f.first, body.first);
        epsilon(body.last, body.first); epsilon(body.last, f.last); return f;
    }
    Fragment optional(Fragment body) {
        auto f = blank(); epsilon(f.first, f.last); epsilon(f.first, body.first);
        epsilon(body.last, f.last); return f;
    }
    Fragment whitespace() { return star(chars(" \t\r\n")); }
    Fragment hex() { return chars("0123456789abcdefABCDEF"); }
    Fragment continuation() { return range(0x80, 0xbf); }
    Fragment string_value() {
        auto scalar = choice({
            range(0x20, 0x21), range(0x23, 0x5b), range(0x5d, 0x7f),
            sequence({range(0xc2, 0xdf), continuation()}),
            sequence({literal("\xe0"), range(0xa0, 0xbf), continuation()}),
            sequence({range(0xe1, 0xec), continuation(), continuation()}),
            sequence({literal("\xed"), range(0x80, 0x9f), continuation()}),
            sequence({range(0xee, 0xef), continuation(), continuation()}),
            sequence({literal("\xf0"), range(0x90, 0xbf), continuation(), continuation()}),
            sequence({range(0xf1, 0xf3), continuation(), continuation(), continuation()}),
            sequence({literal("\xf4"), range(0x80, 0x8f), continuation(), continuation()}),
            sequence({literal("\\"), chars("\"\\/bfnrt")}),
            sequence({literal("\\u"), chars("0123456789abcefABCEF"), hex(), hex(), hex()}),
            sequence({literal("\\u"), chars("dD"), chars("01234567"), hex(), hex()}),
            sequence({literal("\\u"), chars("dD"), chars("89abAB"), hex(), hex(),
                      literal("\\u"), chars("dD"), chars("cdefCDEF"), hex(), hex()})
        });
        return sequence({literal("\""), star(scalar), literal("\"")});
    }
    Fragment number_value(bool integer) {
        auto digit = [&] { return range('0', '9'); };
        auto integral = choice({literal("0"), sequence({range('1', '9'), star(digit())})});
        auto signed_integral = sequence({optional(literal("-")), integral});
        if (integer) return signed_integral;
        return sequence({signed_integral,
            optional(sequence({literal("."), digit(), star(digit())})),
            optional(sequence({chars("eE"), optional(chars("+-")), digit(), star(digit())}))});
    }
    Fragment array_value(const Json& s) {
        if (!s.contains("items")) {
            fail("#", "arrays require finite-depth items schemas", CompileError::UnsupportedKeyword);
            return {INVALID, INVALID};
        }
        int minimum = s.contains("minItems") ? s["minItems"].get<int>() : 0;
        int maximum = s.contains("maxItems") ? s["maxItems"].get<int>() : -1;
        auto f = blank();
        auto opening = sequence({literal("["), whitespace()});
        auto closing = literal("]");
        epsilon(f.first, opening.first); epsilon(closing.last, f.last);
        auto boundary = opening.last;
        if (minimum == 0) epsilon(boundary, closing.first);
        int count_limit = maximum < 0 ? std::max(minimum, 1) : maximum;
        uint32_t final_item = INVALID;
        for (int n = 1; n <= count_limit && error == CompileError::None; ++n) {
            auto item = build(s["items"]);
            final_item = item.first;
            auto piece = n == 1 ? sequence({item, whitespace()}) :
                sequence({literal(","), whitespace(), item, whitespace()});
            epsilon(boundary, piece.first);
            boundary = piece.last;
            if (n >= minimum) epsilon(boundary, closing.first);
        }
        if (maximum < 0 && error == CompileError::None) {
            auto separator = sequence({literal(","), whitespace()});
            epsilon(boundary, separator.first); epsilon(separator.last, final_item);
        }
        return f;
    }
    Fragment object_value(const Json& s) {
        if (!s.contains("additionalProperties") || s["additionalProperties"] != false) {
            fail("#", "objects require additionalProperties:false", CompileError::UnsupportedKeyword);
            return {INVALID, INVALID};
        }
        auto f = blank();
        std::set<std::string> required;
        if (s.contains("required")) for (const auto& k : s["required"]) required.insert(k.get<std::string>());
        size_t count = s.contains("properties") ? s["properties"].size() : 0;
        std::vector<std::array<uint32_t, 2>> boundaries;
        for (size_t i = 0; i <= count && error == CompileError::None; ++i) boundaries.push_back({node(), node()});
        if (error != CompileError::None) return {INVALID, INVALID};
        auto opening = sequence({literal("{"), whitespace()});
        epsilon(f.first, opening.first); epsilon(opening.last, boundaries[0][0]);
        auto closing = literal("}");
        epsilon(boundaries.back()[0], closing.first); epsilon(boundaries.back()[1], closing.first);
        epsilon(closing.last, f.last);
        size_t index = 0;
        if (s.contains("properties")) for (auto it = s["properties"].begin(); it != s["properties"].end(); ++it, ++index) {
            if (error != CompileError::None) break;
            auto property = sequence({literal(Json(it.key()).dump()), whitespace(), literal(":"),
                                      whitespace(), build(it.value()), whitespace()});
            epsilon(boundaries[index][0], property.first);
            auto comma = sequence({literal(","), whitespace()});
            epsilon(boundaries[index][1], comma.first); epsilon(comma.last, property.first);
            epsilon(property.last, boundaries[index+1][1]);
            if (!required.count(it.key())) {
                epsilon(boundaries[index][0], boundaries[index+1][0]);
                epsilon(boundaries[index][1], boundaries[index+1][1]);
            }
        }
        return f;
    }
    Fragment build(const Json& s) {
        if (error != CompileError::None) return {INVALID, INVALID};
        if (s.is_boolean()) {
            if (!s.get<bool>()) return blank();
            fail("#", "unrestricted JSON has unbounded nesting", CompileError::UnsupportedKeyword);
            return {INVALID, INVALID};
        }
        if (s.contains("enum") || s.contains("const")) {
            std::vector<Fragment> values;
            auto add = [&](const Json& v) { if (valid(s, v)) values.push_back(literal(v.dump())); };
            if (s.contains("const")) add(s["const"]);
            else for (const auto& v : s["enum"]) {
                if (error != CompileError::None) break;
                add(v);
            }
            return choice(std::move(values));
        }
        for (const auto& key : {"anyOf", "oneOf"}) if (s.contains(key)) {
            std::vector<Fragment> alternatives;
            for (const auto& v : s[key]) {
                if (error != CompileError::None) break;
                alternatives.push_back(build(v));
            }
            return choice(std::move(alternatives));
        }
        std::vector<std::string> types;
        if (!s.contains("type")) {
            fail("#", "schemas without type or finite values permit unbounded JSON nesting", CompileError::UnsupportedKeyword);
            return {INVALID, INVALID};
        }
        if (s["type"].is_string()) types.push_back(s["type"].get<std::string>());
        else for (const auto& t : s["type"]) types.push_back(t.get<std::string>());
        std::vector<Fragment> alternatives;
        for (const auto& t : types) {
            if (t == "object") alternatives.push_back(object_value(s));
            else if (t == "array") alternatives.push_back(array_value(s));
            else if (t == "string") alternatives.push_back(string_value());
            else if (t == "number") alternatives.push_back(number_value(false));
            else if (t == "integer") alternatives.push_back(number_value(true));
            else if (t == "boolean") alternatives.push_back(choice({literal("true"), literal("false")}));
            else if (t == "null") alternatives.push_back(literal("null"));
        }
        return choice(std::move(alternatives));
    }
    bool determinize(Fragment root, Plan& plan) {
        if (error != CompileError::None) return false;
        std::vector<Subset> reverse(nodes.size());
        for (uint32_t i = 0; i < nodes.size(); ++i) {
            for (auto j : nodes[i].epsilon) reverse[j].push_back(i);
            for (const auto& edge : nodes[i].edges) reverse[edge.destination].push_back(i);
        }
        std::vector<uint8_t> live(nodes.size(), 0);
        Subset todo{root.last}; live[root.last] = 1;
        for (size_t i = 0; i < todo.size(); ++i) for (auto j : reverse[todo[i]])
            if (!live[j]) { live[j] = 1; todo.push_back(j); }
        if (!live[root.first]) return fail("#", "schema has no permitted output");

        std::vector<uint32_t> stamps(nodes.size(), 0);
        uint32_t epoch = 0;
        auto close = [&](Subset seed) {
            if (++epoch == 0) { std::fill(stamps.begin(), stamps.end(), 0); epoch = 1; }
            Subset closure;
            for (auto i : seed) if (live[i] && stamps[i] != epoch) {
                stamps[i] = epoch; closure.push_back(i);
            }
            for (size_t n = 0; n < closure.size(); ++n) for (auto i : nodes[closure[n]].epsilon)
                if (live[i] && stamps[i] != epoch) { stamps[i] = epoch; closure.push_back(i); }
            std::sort(closure.begin(), closure.end());
            return closure;
        };
        std::map<Subset, uint32_t> ids;
        std::vector<Subset> states;
        auto intern = [&](Subset subset) -> uint32_t {
            if (subset.empty()) return INVALID;
            auto found = ids.find(subset);
            if (found != ids.end()) return found->second;
            if (states.size() >= budget) {
                fail("#", "DFA determinization exceeds state budget", CompileError::StateBudget);
                return INVALID;
            }
            uint32_t id = static_cast<uint32_t>(states.size());
            ids.emplace(subset, id); states.push_back(std::move(subset));
            plan.accepting.push_back(std::binary_search(states.back().begin(), states.back().end(), root.last));
            plan.transitions.resize((size_t(id)+1)*256, INVALID);
            return id;
        };
        plan.initial = intern(close({root.first}));
        for (size_t row = 0; row < states.size() && error == CompileError::None; ++row) {
            std::array<Subset, 256> destinations;
            for (auto i : states[row]) for (const auto& edge : nodes[i].edges) {
                if (!live[edge.destination]) continue;
                for (unsigned word = 0; word < 4; ++word) {
                    uint64_t bits = edge.bytes[word];
                    while (bits) {
                        unsigned bit = static_cast<unsigned>(__builtin_ctzll(bits)); bits &= bits-1;
                        destinations[word*64+bit].push_back(edge.destination);
                    }
                }
            }
            for (unsigned byte = 0; byte < 256 && error == CompileError::None; ++byte) {
                auto id = intern(close(std::move(destinations[byte])));
                plan.transitions[row*256+byte] = id;
            }
        }
        return error == CompileError::None;
    }
};
}

CompileResult compile(const Json& schema, uint32_t state_budget) {
    Compiler compiler(state_budget);
    if (state_budget == 0 || state_budget > 1048576)
        return {{}, CompileError::StateBudget, "state budget must be in [1,1048576]"};
    try {
        Json expanded;
        std::vector<std::string> references;
        if (!compiler.expand(schema, schema, expanded, "#", references) || !compiler.check(expanded, "#"))
            return {{}, compiler.error, compiler.detail};
        auto value = compiler.build(expanded);
        auto root = compiler.sequence({compiler.whitespace(), value, compiler.whitespace()});
        Plan plan;
        if (!compiler.determinize(root, plan)) return {{}, compiler.error, compiler.detail};
        return {std::move(plan), CompileError::None, {}};
    } catch (const nlohmann::json::type_error& e) {
        return {{}, e.id == 316 ? CompileError::InvalidUtf8 : CompileError::InvalidSchema, e.what()};
    } catch (const nlohmann::json::exception& e) {
        return {{}, CompileError::InvalidSchema, e.what()};
    } catch (const std::bad_alloc&) {
        return {{}, CompileError::StateBudget, "compiler allocation failed"};
    }
}
}

