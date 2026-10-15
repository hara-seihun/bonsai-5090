#include "gpu_grammar.h"
#include "grammar.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using Json = nlohmann::ordered_json;
using h5::gpu_grammar::CompileError;
using h5::gpu_grammar::Plan;
using h5::gpu_grammar::INVALID;

static void require(bool ok, const std::string& label) {
    if (!ok) {
        std::cerr << "FAIL: " << label << '\n';
        std::exit(1);
    }
}

static std::string hex(const std::string& bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : bytes) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}

struct Suite {
    std::shared_ptr<const h5::grammar::Vocabulary> vocabulary;
    uint64_t compiled = 0, states = 0, bytes = 0, prefixes = 0, outputs = 0, errors = 0;
    double compile_ms = 0;

    Suite() {
        std::vector<std::string> pieces;
        for (int byte = 0; byte < 256; ++byte) pieces.emplace_back(1, char(byte));
        pieces.emplace_back("<EOS>");
        vocabulary = std::make_shared<h5::grammar::Vocabulary>(std::move(pieces), std::vector<int>{256});
    }

    Plan compile(const std::string& label, const Json& schema) {
        auto start = std::chrono::steady_clock::now();
        auto result = h5::gpu_grammar::compile(schema);
        double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        require(result.ok(), label + " compile error=" + std::to_string(int(result.error)) + ": " + result.detail);
        require(result.detail.empty(), label + " successful compile contains an error detail");
        const Plan& plan = result.plan;
        require(plan.states() > 0 && plan.initial < plan.states(), label + " valid initial state");
        require(plan.transitions.size() == size_t(plan.states()) * 256, label + " 256 transitions per state");
        require(plan.states() <= 32768, label + " obeys state budget");
        for (uint8_t accepting : plan.accepting) require(accepting <= 1, label + " boolean acceptance");
        for (uint32_t next : plan.transitions)
            require(next == INVALID || next < plan.states(), label + " transition stays inside plan");
        ++compiled;
        states += plan.states();
        bytes += plan.bytes();
        compile_ms += elapsed;
        std::cout << Json{{"case", label}, {"states", plan.states()}, {"bytes", plan.bytes()}, {"compile_ms", elapsed}}.dump() << '\n';
        return std::move(result.plan);
    }

    void mask(const std::string& label, const Plan& plan, uint32_t state,
              const h5::grammar::Matcher& cpu, size_t offset) {
        std::array<uint32_t, 9> words;
        words.fill(~uint32_t(0));
        const int count = cpu.mask(words.data());
        int observed = 0;
        for (uint32_t token = 0; token <= 256; ++token) {
            const bool cpu_allowed = ((words[token / 32] >> (token % 32)) & 1) != 0;
            const bool plan_allowed = token == 256 ? plan.accepting[state] != 0
                : plan.transitions[size_t(state) * 256 + token] != INVALID;
            require(cpu_allowed == plan_allowed, label + " mask mismatch at prefix " + std::to_string(offset)
                    + " token " + std::to_string(token) + " state " + std::to_string(state));
            observed += cpu_allowed;
        }
        require(count == observed && (words.back() & ~uint32_t(1)) == 0, label + " mask count/padding");
        require(cpu.complete() == bool(plan.accepting[state]), label + " completion at prefix " + std::to_string(offset));
        ++prefixes;
    }

    void output(const std::string& label, const Json& schema, const Plan& plan,
                const std::string& text, bool expected) {
        auto result = h5::grammar::compile(schema, vocabulary);
        require(bool(result), label + " CPU compile: " + result.error);
        auto& cpu = *result.matcher;
        uint32_t state = plan.initial;
        bool viable = true;
        for (size_t offset = 0; offset <= text.size(); ++offset) {
            mask(label, plan, state, cpu, offset);
            if (offset == text.size()) break;
            const unsigned char byte = text[offset];
            const uint32_t next = plan.transitions[size_t(state) * 256 + byte];
            const bool accepted = cpu.accept(byte);
            require(accepted == (next != INVALID), label + " byte acceptance at " + std::to_string(offset));
            if (!accepted) { viable = false; break; }
            state = next;
        }
        const bool actual = viable && plan.accepting[state];
        require(actual == expected, label + (expected ? " rejected valid output: " : " accepted invalid output: ") + hex(text));
        if (viable) require(cpu.accept(256) == actual, label + " EOS matches acceptance");
        ++outputs;
    }

    void examples(const std::string& label, const Json& schema,
                  const std::vector<std::string>& valid, const std::vector<std::string>& invalid) {
        const Plan plan = compile(label, schema);
        for (const auto& text : valid) output(label, schema, plan, text, true);
        for (const auto& text : invalid) output(label, schema, plan, text, false);
    }

    void error(const std::string& label, const Json& schema, CompileError expected, uint32_t budget = 32768) {
        auto result = h5::gpu_grammar::compile(schema, budget);
        require(!result.ok() && result.error == expected, label + " expected error=" + std::to_string(int(expected))
                + " actual=" + std::to_string(int(result.error)) + ": " + result.detail);
        require(!result.detail.empty(), label + " error has diagnostic");
        require(result.plan.accepting.empty() && result.plan.transitions.empty(), label + " error does not expose partial plan");
        ++errors;
    }
};

int main(int argc, char** argv) {
    if (argc > 2) { std::cerr << "usage: gpu-grammar-compile-test [GRAMMAR_CASES_JSON]\n"; return 2; }
    Suite suite;
    const std::string path = argc == 2 ? argv[1] : "tools/grammar_cases.json";
    std::ifstream input(path);
    require(bool(input), "open fixtures: " + path);
    Json fixtures;
    try { input >> fixtures; }
    catch (const Json::exception& e) { require(false, std::string("fixture JSON: ") + e.what()); }
    require(fixtures.contains("cases") && fixtures["cases"].is_array() && !fixtures["cases"].empty(), "fixture cases array");
    for (const auto& item : fixtures.at("cases")) {
        const std::string label = item.at("id").get<std::string>();
        const auto& schema = item.at("schema");
        const Plan plan = suite.compile(label, schema);
        suite.output(label, schema, plan, item.at("expected").dump(), true);
    }
    const size_t fixture_count = fixtures.at("cases").size();

    const Json string_schema = {{"type", "string"}};
    std::vector<std::string> bad_strings = {"\"\\uD800\"", "\"\\uDC00\"", "\"\\uD800\\u0041\"", "\"\\uD83D\\uD800\"",
        "\"\\x\"", "\"\n\"", "\"\\u12\"", "\"", "\"abc", "\"ok\"x", std::string("\"\0\"", 3)};
    for (const auto& raw : {std::string("\xc0\xaf", 2), std::string("\xe0\x80\xaf", 3), std::string("\xed\xa0\x80", 3),
         std::string("\xf0\x80\x80\xaf", 4), std::string("\xf4\x90\x80\x80", 4), std::string("\xf5\x80\x80\x80", 4),
         std::string("\x80", 1), std::string("\xe2\x82", 2)}) bad_strings.push_back("\"" + raw + "\"");
    suite.examples("contract/string", string_schema,
        {"\"\"", "\"é\"", "\"山田😀\"", "\"\xe0\xa0\x80\"", "\"\xed\x9f\xbf\"", "\"\xf0\x90\x80\x80\"", "\"\xf4\x8f\xbf\xbf\"",
         "\"\\uD83D\\uDE00\"", "\"\\ud7ff\\ue000\"", "\"\\u0000\"",
         "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"", " \n\"hello\"\r\t"}, bad_strings);
    suite.examples("contract/number", {{"type", "number"}},
        {"0", "-0", "123", "-2.5", "1e+3", "0.01E-10", " 12 \n"},
        {"01", "+1", "1.", "1e+", "NaN", "Infinity", "--2", "1 2", "", "-", ".5"});
    suite.examples("contract/integer", {{"type", "integer"}}, {"0", "-42", "1234567890", "\t7\n"},
        {"4.2", "4e0", "01", "-", "1 2"});
    suite.examples("contract/optional-ordered-object", Json::parse(R"({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"},"c":{"type":"boolean"}},"required":["b"],"additionalProperties":false})"),
        {R"({"b":1})", R"({"a":"x","b":-2})", R"({"b":0,"c":true})", R"({"a":"é","b":3,"c":false})", " { \"b\" : 1 } \n"},
        {"{}", R"({"c":true,"b":1})", R"({"b":1,"a":"x"})", R"({"b":1,"b":2})", R"({"b":1,"extra":0})", R"({"a":"x"})"});
    suite.examples("contract/array-bounds", Json::parse(R"({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2})"),
        {"[1]", "[-2,3]", " [ 1 , 2 ] "}, {"[]", "[1,2,3]", "[1,]", "[true]", "[1.0]", "["});
    suite.examples("contract/array-unbounded", Json::parse(R"({"type":"array","items":{"type":"string"}})"),
        {"[]", R"(["a","b","c","d"])", R"(["😀"] )"}, {"[null]", "[,]", R"(["a",])"});
    suite.examples("contract/empty-array", Json::parse(R"({"type":"array","items":false})"), {"[]", " [ ] "}, {"[null]"});
    suite.examples("contract/anyOf-overlap", Json::parse(R"({"anyOf":[{"enum":["car","cat"]},{"enum":["cat","cart"]}]})"),
        {R"("car")", R"("cat")", R"("cart")"}, {R"("cap")", R"("cats")"});
    suite.examples("contract/anyOf-number-overlap", Json::parse(R"({"anyOf":[{"type":"number"},{"type":"integer"}]})"),
        {"0", "-42", "1.2", "1e+2"}, {"01", "1e+"});
    suite.examples("contract/oneOf-disjoint-types", Json::parse(R"({"oneOf":[{"type":"string"},{"type":"null"},{"type":"boolean"}]})"),
        {R"("x")", "null", "true", "false"}, {"1", "[]", "{}"});
    suite.examples("contract/oneOf-finite", Json::parse(R"({"oneOf":[{"enum":["red","green"]},{"enum":["blue","black"]}]})"),
        {R"("red")", R"("blue")", R"("black")"}, {R"("pink")", R"("blu")"});
    suite.examples("contract/oneOf-array-length", Json::parse(R"({"oneOf":[{"type":"array","items":{"type":"integer"},"maxItems":0},{"type":"array","minItems":1,"maxItems":2,"items":{"type":"integer"}}]})"),
        {"[]", "[1]", "[1,2]"}, {"[true]", "[1,2,3]"});
    suite.examples("contract/ref-json-pointer", Json::parse(R"({"$defs":{"a/b~c":{"const":"yes"}},"$ref":"#/$defs/a~1b~0c"})"),
        {R"("yes")"}, {R"("no")"});
    suite.examples("contract/finite-const", Json::parse(R"({"const":{"answer":42,"text":"é"}})"),
        {R"({"answer":42,"text":"é"})"}, {R"({"text":"é","answer":42})", R"({"answer":43,"text":"é"})"});

    for (const auto& schema : {R"({"type":"string","pattern":"a"})", R"({"type":"number","minimum":0})",
         R"({"type":"string","maxLength":10})", R"({"type":"object","additionalProperties":{}})"})
        suite.error("unsupported-keyword", Json::parse(schema), CompileError::UnsupportedKeyword);
    for (const auto& schema : {"{}", "true", R"({"type":"object"})", R"({"type":"array"})", R"({"type":"array","items":true})"})
        suite.error("open-generic-schema", Json::parse(schema), CompileError::UnsupportedKeyword);
    for (const auto& schema : {R"({"$ref":"#/$defs/missing"})", R"({"$ref":"https://example.test/schema"})",
         R"({"$ref":"#"})", R"({"$defs":{"x":{"$ref":"#/$defs/x"}},"$ref":"#/$defs/x"})",
         R"({"$defs":{"x":{"$ref":"#/$defs/y"},"y":{"$ref":"#/$defs/x"}},"$ref":"#/$defs/x"})"})
        suite.error("invalid-reference", Json::parse(schema), CompileError::InvalidReference);
    for (const auto& schema : {R"({"oneOf":[{"type":"number"},{"type":"integer"}]})",
         R"({"oneOf":[{"enum":["red","green"]},{"const":"green"}]})",
         R"({"oneOf":[{"const":1},{"const":1.0}]})"})
        suite.error("overlapping-oneOf", Json::parse(schema), CompileError::OverlappingOneOf);
    for (const auto& schema : {"123", "false", R"({"type":"bad"})", R"({"anyOf":[]})", R"({"minItems":2,"maxItems":1,"type":"array","items":{"type":"integer"}})"})
        suite.error("invalid-schema", Json::parse(schema), CompileError::InvalidSchema);
    suite.error("invalid-UTF8-const", Json{{"const", std::string("\xc0\xaf", 2)}}, CompileError::InvalidUtf8);
    suite.error("invalid-UTF8-key", Json{{"type", "object"}, {"properties", Json{{std::string("\xed\xa0\x80", 3), Json{{"type", "string"}}}}}, {"additionalProperties", false}}, CompileError::InvalidUtf8);
    suite.error("zero-budget", string_schema, CompileError::StateBudget, 0);
    suite.error("excessive-budget", string_schema, CompileError::StateBudget, 1048577);
    suite.error("small-budget", Json{{"const", "long-literal"}}, CompileError::StateBudget, 2);
    const Json tight_schema = {{"const", true}};
    const Plan tight_plan = suite.compile("contract/exact-budget", tight_schema);
    require(h5::gpu_grammar::compile(tight_schema, tight_plan.states()).ok(), "exact state budget permits plan");
    suite.error("one-short-budget", tight_schema, CompileError::StateBudget, tight_plan.states() - 1);

    std::cout << Json{{"result", "pass"}, {"fixture_cases", fixture_count}, {"compiled", suite.compiled},
        {"states", suite.states}, {"bytes", suite.bytes}, {"compile_ms", suite.compile_ms},
        {"mask_prefixes", suite.prefixes}, {"mask_bits", suite.prefixes * 257},
        {"outputs", suite.outputs}, {"error_contracts", suite.errors}}.dump() << '\n';
}
