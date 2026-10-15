#include "grammar.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using h5::grammar::Matcher;
using h5::grammar::Vocabulary;
using h5::grammar::compile;
using Json = nlohmann::ordered_json;

static void require(bool condition, const std::string& label) {
    if (!condition) { std::cerr << "FAIL: " << label << '\n'; std::exit(1); }
}

struct Fixture {
    std::vector<std::string> pieces;
    std::shared_ptr<const Vocabulary> vocabulary;
    int eos;
    Fixture() {
        for (int c = 0; c < 256; ++c) pieces.emplace_back(1, char(c));
        for (const auto& text : {"{\"name\":\"", "Alice\",\"rows\":[{\"ok\":true}", "]}",
             "{\"rows\":[]}", "true}", "\\uD83D\\uDE00", "\\uD800\"", "01", "1e+", "1e+3", "null", "false", "\"x\"",
             "{}garbage", "\"\\u0000\"", "[1,2]", " ", "\"é\"", "\"😀\"", "\"\n\"", "\"name\":", "[]", "\"}",
             "red\"", "green\"", "pink\"", "cat\",\"payload\":\"", "car\",\"payload\":0}"}) pieces.emplace_back(text);
        pieces.push_back("");
        eos = int(pieces.size());
        pieces.push_back("<EOS>");
        pieces.push_back("<EOS2>");
        vocabulary = std::make_shared<Vocabulary>(pieces, std::vector<int>{eos, eos + 1});
    }
    int id(const std::string& piece) const {
        for (size_t i = 0; i < pieces.size(); ++i) if (pieces[i] == piece) return int(i);
        require(false, "piece exists: " + piece); return -1;
    }
    std::unique_ptr<Matcher> matcher(const Json& schema) const {
        auto result = compile(schema, vocabulary);
        require(bool(result), "compile: " + result.error);
        require(result.error.empty(), "successful result has no error");
        return std::move(result.matcher);
    }
    bool feed(Matcher& m, const std::string& text) const {
        for (unsigned char c : text) if (!m.accept(int(c))) return false;
        return true;
    }
    void check_mask(const Matcher& m) const {
        std::vector<uint32_t> words((pieces.size() + 31) / 32, ~uint32_t(0));
        int allowed = m.mask(words.data());
        int observed = 0;
        for (size_t i = 0; i < pieces.size(); ++i) {
            Matcher branch(m);
            bool accepted = branch.accept(int(i));
            bool masked = (words[i / 32] >> (i % 32)) & 1;
            require(accepted == masked, "mask equals complete piece acceptance at " + std::to_string(i));
            observed += masked;
        }
        require(allowed == observed, "mask count");
        if (pieces.size() % 32) require((words.back() >> (pieces.size() % 32)) == 0, "unused mask bits zero");
        std::vector<uint32_t> cached(words.size());
        require(m.mask(cached.data()) == allowed && words == cached, "identical state cached mask");
    }
    void next_bytes(const Matcher& m, const std::string& expected) const {
        std::vector<uint32_t> words((pieces.size() + 31) / 32);
        require(m.mask(words.data()) > 0, "prefix has viable tokens");
        for (int c = 0; c < 256; ++c)
            require(bool((words[c / 32] >> (c % 32)) & 1) == (expected.find(char(c)) != std::string::npos),
                "immediate byte pruning at " + std::to_string(c));
        check_mask(m);
    }
    void output(const Json& schema, const std::string& text, bool expected) const {
        auto m = matcher(schema);
        bool actual = feed(*m, text) && m->complete();
        require(actual == expected, std::string(expected ? "accept " : "reject ") + text);
    }
};

int main(int argc, char** argv) {
    Fixture f;
    Json record = Json::parse(R"({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false})");
    Json schema = Json::parse(R"({"type":"object","properties":{"name":{"type":"string"},"rows":{"type":"array","items":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false},"minItems":1,"maxItems":2},"comment":{"type":["string","null"]}},"required":["rows"],"additionalProperties":false})");
    f.output(schema, R"({"name":"Alice","rows":[{"ok":true}]})", true);
    f.output(schema, R"({"rows":[{"ok":false},{"ok":true}],"comment":null})", true);
    f.output(schema, " \n { \"rows\" : [ { \"ok\" : true } ] } \t", true);
    for (const auto& text : {R"({})", R"({"rows":[]})", R"({"rows":[{}]})", R"({"rows":[{"ok":true},{"ok":true},{"ok":true}]})",
         R"({"rows":[{"ok":true}],"extra":1})", R"({"rows":[{"ok":true}],"name":"wrong order"})", R"({"name":"x","name":"y","rows":[{"ok":true}]})"}) f.output(schema, text, false);
    auto m = f.matcher(schema);
    f.check_mask(*m);
    require(m->accept(f.id("{\"name\":\"")), "token crosses key and opening string");
    f.check_mask(*m);
    require(m->accept(f.id("Alice\",\"rows\":[{\"ok\":true}")), "token crosses string, array, nested record");
    f.check_mask(*m);
    require(m->accept(f.id("]}")) && m->complete(), "token crosses array and outer object close");
    f.check_mask(*m);
    require(m->accept(f.eos), "EOS only after accepting");
    require(m->accept(f.eos + 1), "EOS padding is idempotent");
    require(!m->accept(' '), "no bytes after EOS");
    f.check_mask(*m);

    auto fork = f.matcher(record);
    require(f.feed(*fork, "{\"ok\":"), "fork prefix");
    Matcher a(*fork), b(*fork);
    require(f.feed(a, "true}") && a.complete(), "fork A");
    require(f.feed(b, "false}") && b.complete(), "fork B");
    require(!fork->complete() && !fork->accept(f.eos), "original prefix unchanged");
    Matcher assigned = a;
    assigned = *fork;
    require(f.feed(assigned, "false}") && assigned.complete(), "copy assignment independent");
    auto transactional = f.matcher(Json::parse(R"({"type":"object","additionalProperties":false})"));
    require(!transactional->accept(f.id("{}garbage")), "invalid suffix rejects entire token");
    require(f.feed(*transactional, "{}") && transactional->complete(), "failed token leaves prefix unchanged");
    require(!transactional->accept(-1) && !transactional->accept(99999), "invalid token IDs");

    Json str = {{"type", "string"}};
    for (const auto& text : {"\"é\"", "\"😀\"", "\"\\uD83D\\uDE00\"", "\"\\u0000\"", "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"", "\"\\ud7ff\\ue000\""}) f.output(str, text, true);
    for (const auto& text : {"\"\\uD800\"", "\"\\uDC00\"", "\"\\uD800\\u0041\"", "\"\\x\"", "\"\n\""}) f.output(str, text, false);
    for (const auto& bytes : {std::string("\xc0\xaf", 2), std::string("\xed\xa0\x80", 3), std::string("\xf4\x90\x80\x80", 4), std::string("\x80", 1)}) f.output(str, "\"" + bytes + "\"", false);
    auto unicode = f.matcher(str);
    require(f.feed(*unicode, std::string("\"\xf0", 2)), "partial UTF8 prefix accepted");
    require(!unicode->complete(), "partial UTF8 incomplete");
    f.check_mask(*unicode);
    require(f.feed(*unicode, std::string("\x9f\x98\x80\"", 4)) && unicode->complete(), "UTF8 split tokens");
    auto escape = f.matcher(str);
    require(f.feed(*escape, "\"\\uD8"), "partial surrogate escape accepted");
    f.check_mask(*escape);
    require(f.feed(*escape, "3D\\uDE00\"") && escape->complete(), "escape split tokens");

    Json num = {{"type", "number"}};
    for (const auto& text : {"0", "-0", "123", "-2.5", "1e+3", "0.01E-10"}) f.output(num, text, true);
    for (const auto& text : {"01", "+1", "1.", "1e+", "NaN", "Infinity", "--2", "1 2"}) f.output(num, text, false);
    auto n = f.matcher(num);
    require(f.feed(*n, "1e+"), "number viable incomplete exponent");
    require(!n->complete() && !n->accept(f.eos), "no EOS for incomplete exponent");
    f.check_mask(*n);
    f.output({{"type", "integer"}}, "-42", true);
    f.output({{"type", "integer"}}, "4.2", false);
    f.output({{"type", "integer"}}, "4e0", false);

    Json en = Json::parse(R"({"type":"string","enum":["red","green","blue"]})");
    f.output(en, "\"green\"", true);
    f.output(en, "\"pink\"", false);
    f.output(Json::parse(R"({"type":["string","null"],"enum":["x",null,12]})"), "null", true);
    f.output(Json::parse(R"({"type":["string","null"],"enum":["x",null,12]})"), "12", false);
    f.output(Json::parse(R"({"const":{"answer":42}})"), "{\"answer\":42}", true);
    f.output(Json::parse(R"({"const":{"a":1,"b":2},"enum":[{"b":2,"a":1}]})"), "{\"a\":1,\"b\":2}", true);
    f.output(Json::parse(R"({"anyOf":[{"type":"null"},{"type":"array","items":{"enum":[1,2]},"maxItems":2}]})"), "[1,2]", true);
    f.output(Json::parse(R"({"anyOf":[false,{"const":true}]})"), "true", true);
    auto enum_prefix = f.matcher(en);
    require(enum_prefix->accept('"'), "enum opening prefix");
    f.next_bytes(*enum_prefix, "rgb");
    require(enum_prefix->accept(f.id("red\"")) && enum_prefix->complete(), "whole enum suffix token");
    auto enum_fork = f.matcher(en);
    require(f.feed(*enum_fork, "\"gr"), "enum shared prefix");
    f.next_bytes(*enum_fork, "e");
    require(!enum_fork->accept(f.id("pink\"")), "dead enum token rejected immediately");
    f.next_bytes(*enum_fork, "e");
    Json tagged = Json::parse(R"({"anyOf":[
        {"type":"object","properties":{"kind":{"const":"cat"},"payload":{"type":"string"}},"required":["kind","payload"],"additionalProperties":false},
        {"type":"object","properties":{"kind":{"const":"car"},"payload":{"const":0}},"required":["kind","payload"],"additionalProperties":false}]})");
    auto union_prefix = f.matcher(tagged);
    require(f.feed(*union_prefix, "{\"kind\":\"ca"), "anyOf shared prefix");
    f.next_bytes(*union_prefix, "tr");
    require(union_prefix->accept('t'), "anyOf branch commits at distinguishing byte");
    f.next_bytes(*union_prefix, "\"");
    require(f.feed(*union_prefix, "\",\"payload\":"), "anyOf selected continuation");
    f.next_bytes(*union_prefix, " \t\r\n\"");
    require(!union_prefix->accept('0'), "eliminated anyOf branch cannot reappear");
    require(f.feed(*union_prefix, "\"hello\"}") && union_prefix->complete(), "anyOf live payload");
    tagged["oneOf"] = tagged["anyOf"];
    tagged.erase("anyOf");
    for (const auto& text : {R"({"kind":"cat","payload":"hello"})", R"({"kind":"car","payload":0})"}) f.output(tagged, text, true);
    f.output(tagged, R"({"kind":"cat","payload":0})", false);
    auto exclusive_prefix = f.matcher(tagged);
    require(f.feed(*exclusive_prefix, "{\"kind\":\""), "oneOf common structural prefix");
    f.next_bytes(*exclusive_prefix, "c");
    Matcher exclusive_other(*exclusive_prefix);
    require(exclusive_prefix->accept(f.id("cat\",\"payload\":\"")), "oneOf packed discriminator plus payload opening");
    f.check_mask(*exclusive_prefix);
    require(f.feed(*exclusive_prefix, "x\"}") && exclusive_prefix->complete(), "oneOf string branch");
    require(exclusive_other.accept(f.id("car\",\"payload\":0}")) && exclusive_other.complete(), "oneOf numeric packed branch");
    f.output(Json::parse(R"({"oneOf":[{"type":"string"},{"type":"null"},{"type":"boolean"}]})"), "null", true);
    f.output(Json::parse(R"({"oneOf":[{"enum":["red","green"]},{"enum":["blue","black"]}]})"), "\"blue\"", true);
    f.output(Json::parse(R"({"oneOf":[false,{"const":true}]})"), "true", true);
    f.output(Json::parse(R"({"oneOf":[{"type":"array","maxItems":0},{"type":"array","minItems":1,"items":{"type":"integer"}}]})"), "[1]", true);
    Json refs = Json::parse(R"({"$defs":{"label":{"enum":["active","inactive"]},"record":{"type":"object","properties":{"status":{"$ref":"#/$defs/label"}},"required":["status"],"additionalProperties":false}},"type":"array","items":{"$ref":"#/$defs/record"},"minItems":1,"maxItems":2})");
    f.output(refs, R"([{"status":"active"}])", true);
    f.output(refs, R"([{"status":"unknown"}])", false);
    auto ref_prefix = f.matcher(refs);
    require(f.feed(*ref_prefix, "[{\"status\":\"in"), "local ref enum prefix");
    f.next_bytes(*ref_prefix, "a");
    f.output(Json::parse(R"({"$defs":{"a/b~c":{"const":"yes"}},"$ref":"#/$defs/a~1b~0c"})"), "\"yes\"", true);
    Json union_refs = {{"$defs", {{"cat", tagged["oneOf"][0]}, {"car", tagged["oneOf"][1]}}},
                       {"oneOf", Json::array({Json{{"$ref", "#/$defs/cat"}}, Json{{"$ref", "#/$defs/car"}}})}};
    f.output(union_refs, R"({"kind":"car","payload":0})", true);
    f.output(Json::parse(R"({"type":"array","items":false})"), "[]", true);
    f.output(Json::parse(R"({"type":"array","items":false})"), "[null]", false);
    f.output(Json::parse(R"({"type":"object","properties":{"never":false,"ok":{"const":true}},"additionalProperties":false})"), "{\"ok\":true}", true);
    for (const auto& text : {"null", "{}", "[1,{\"a\":[true,false,null]},\"😀\"]", "{\"x\":-1.2e3}"}) f.output(Json::object(), text, true);
    f.output(Json::parse(R"({"properties":{"x":{"type":"integer"}},"additionalProperties":false})"), "null", true);

    for (const auto& text : {R"({"type":"string","pattern":"a"})", R"({"type":"number","minimum":0})", R"({"$ref":"#/$defs/x"})",
         R"({"type":"array","items":[{"type":"string"}]})", R"({"type":"object","properties":{"x":{"format":"date"}},"additionalProperties":false})",
         R"({"properties":{"x":{}},"additionalProperties":true})", R"({"type":"bad"})", R"({"type":["string","string"]})",
         R"({"required":["x"],"additionalProperties":false})", R"({"required":"x"})", R"({"maxItems":-1})", R"({"maxItems":5000})",
         R"({"minItems":2,"maxItems":1})", R"({"enum":[]})", R"({"enum":[1],"type":"string"})", R"({"const":true,"type":"null"})",
         R"({"anyOf":[]})", R"({"anyOf":[true],"type":"string"})", R"({"additionalProperties":{}})",
         R"({"oneOf":[]})", R"({"oneOf":[{"type":"number"},{"type":"integer"}]})",
         R"({"oneOf":[{"enum":["red","green"]},{"const":"green"}]})",
         R"({"oneOf":[{"const":1},{"const":1.0}]})",
         R"({"oneOf":[{"const":{"a":1,"b":2}},{"const":{"b":2,"a":1}}]})",
         R"({"oneOf":[true,{"type":"string"}]})", R"({"oneOf":[false],"type":"string"})",
         R"({"anyOf":[true],"oneOf":[true]})",
         R"({"oneOf":[{"properties":{"tag":{"const":"a"}},"additionalProperties":false},{"properties":{"tag":{"const":"b"}},"additionalProperties":false}]})",
         R"({"oneOf":[{"type":"object","properties":{"tag":{"const":"a"}},"additionalProperties":false},{"type":"object","properties":{"tag":{"const":"b"}},"additionalProperties":false}]})",
         R"({"oneOf":[{"type":"array","items":{"type":"string"}},{"type":"array","items":{"type":"number"}}]})",
         R"({"$ref":"#/$defs/missing"})", R"({"$ref":"https://example.test/schema"})", R"({"$ref":42})",
         R"({"$defs":[]})", R"({"$defs":{"bad":1}})", R"({"$defs":{"x":{"type":"string"}},"$ref":"#/$defs/x","type":"null"})",
         R"({"$defs":{"x":{"$ref":"#/$defs/x"}},"$ref":"#/$defs/x"})", R"({"$ref":"#"})",
         R"({"$defs":{"x y":true},"$ref":"#/$defs/x%20y"})", "false", "123"}) {
        auto result = compile(Json::parse(text), f.vocabulary);
        require(!result.matcher && !result.error.empty(), std::string("explicit schema error: ") + text);
    }
    require(!compile(Json::object(), nullptr), "unset vocabulary error");
    require(!compile(Json::object(), std::make_shared<Vocabulary>(std::vector<std::string>{"x"}, std::vector<int>{-1})), "invalid EOS vocabulary error");
    require(!compile(Json{{"const", std::numeric_limits<double>::quiet_NaN()}}, f.vocabulary), "non-JSON NaN literal error");
    require(!compile(Json::object(), std::make_shared<Vocabulary>(std::vector<std::string>{}, std::vector<int>{})), "empty vocabulary error");

    // Hybrid canonicalization changes only already byte-deterministic states.
    h5::grammar::CompileOptions canonical;
    canonical.mode = h5::grammar::Mode::CanonicalFixed;
    canonical.encode = [&](const std::string& text) {
        std::vector<int> tokens;
        for (unsigned char c : text) tokens.push_back(c);
        return tokens;
    };
    auto cm = compile(Json::parse(R"({"type":"object","properties":{"name":{"type":"string"},"ok":{"const":true}},"required":["name","ok"],"additionalProperties":false})"), f.vocabulary, canonical);
    require(bool(cm), "canonical fixed compiles");
    f.check_mask(*cm.matcher);
    std::vector<uint32_t> one((f.pieces.size()+31)/32);
    require(cm.matcher->mask(one.data()) > 1 && cm.matcher->accept(f.id("{\"name\":\"")), "ambiguous initial grammar permits full cross-boundary token");
    f.check_mask(*cm.matcher);
    require(cm.matcher->accept(f.id("null")), "dynamic string token remains full vocabulary");
    require(f.feed(*cm.matcher, "\",\"ok\":true}") && cm.matcher->complete(), "canonical root complete");
    f.check_mask(*cm.matcher);
    require(cm.matcher->accept(' '), "ordinary accepting-state whitespace remains legal");
    require(!compile(str, f.vocabulary, h5::grammar::CompileOptions{h5::grammar::Mode::CanonicalFixed,{}}), "canonical encoder required");
    auto bad_encoder = canonical;
    bad_encoder.encode = [](const std::string&) { return std::vector<int>{-1}; };
    require(!compile(str, f.vocabulary, bad_encoder), "canonical disabled encoder token error");
    bad_encoder.encode = [](const std::string&) { return std::vector<int>{'x'}; };
    require(!compile(str, f.vocabulary, bad_encoder), "canonical non-reproducing encoder error");
    auto merged_encoder = canonical;
    merged_encoder.encode = [&](const std::string& text) {
        if (text == "{\"name\":\"" || text == "\"name\":" || text == "[]" || text == "\"}") return std::vector<int>{f.id(text)};
        return canonical.encode(text);
    };
    auto merged = compile(Json::parse(R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false})"), f.vocabulary, merged_encoder);
    require(bool(merged), "joint fixed span compiles");
    require(merged.matcher->accept(f.id("{\"name\":\"")), "ordinary natural opening token retained");
    require(merged.matcher->accept(f.id("null")), "joint boundary permits dynamic payload");
    merged.matcher->mask(one.data());
    require(one['"'/32] & (1u << ('"'%32)), "closing quote is exposed alongside natural packed tail");
    require(merged.matcher->accept(f.id("\"}")) && merged.matcher->complete(), "natural closing quote plus tail token retained");
    f.check_mask(*merged.matcher);
    auto fixed = compile(Json{{"const","name"}}, f.vocabulary, canonical);
    auto control = f.matcher(Json{{"const","name"}});
    require(fixed.matcher->mask(one.data()) > 1 && fixed.matcher->accept('"'), "ambiguous whitespace versus literal opening remains ordinary");
    require(control->accept('"'), "ordinary forced-prefix control");
    require(fixed.matcher->mask(one.data()) == 1 && fixed.matcher->accept('n') && control->accept('n'), "forced token reproduces ordinary byte grammar");
    auto branches = compile(Json::parse(R"({"type":"array","items":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}})"), f.vocabulary, merged_encoder);
    require(bool(branches), "canonical record array compiles");
    require(branches.matcher->mask(one.data()) > 1 && branches.matcher->accept('['), "ambiguous opener keeps ordinary vocabulary");
    std::vector<uint32_t> ordinary_mask(one.size());
    auto ordinary_array = f.matcher(Json::parse(R"({"type":"array","items":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}})"));
    require(ordinary_array->accept('['), "array branch control prefix");
    require(branches.matcher->mask(one.data()) == ordinary_array->mask(ordinary_mask.data()) && one == ordinary_mask, "ambiguous array boundary mask bit-for-bit ordinary");
    require(f.feed(*branches.matcher,"{\"ok\":true}]") && branches.matcher->complete(), "nonempty canonical record path");
    auto hybrid_string = compile(str, f.vocabulary, canonical);
    auto ordinary_string = f.matcher(str);
    require(hybrid_string.matcher->accept('"') && ordinary_string->accept('"'), "payload branch prefix");
    require(hybrid_string.matcher->mask(one.data()) == ordinary_string->mask(ordinary_mask.data()) && one == ordinary_mask, "ambiguous payload mask bit-for-bit ordinary");

    const std::string case_path = argc > 1 ? argv[1] : "tools/grammar_cases.json";
    std::ifstream case_file(case_path);
    require(bool(case_file), "open benchmark fixtures: " + case_path);
    Json cases = Json::parse(case_file);
    for (const auto& fixture : cases["cases"]) {
        auto result = compile(fixture["schema"], f.vocabulary);
        require(bool(result), "benchmark schema " + fixture["id"].get<std::string>() + ": " + result.error);
        require(f.feed(*result.matcher, fixture["expected"].dump()) && result.matcher->complete(),
                "benchmark expected output permitted: " + fixture["id"].get<std::string>());
    }

    // A full-size vocabulary demonstrates shared-state cache reuse independently
    // of tokenizer libraries or GPU availability.
    std::vector<std::string> pieces;
    pieces.reserve(128002);
    for (int i = 0; i < 128000; ++i) pieces.push_back("word" + std::to_string(i));
    pieces.push_back("\"");
    pieces.push_back("");
    auto vocab = std::make_shared<Vocabulary>(std::move(pieces), std::vector<int>{128001});
    auto result = compile(str, vocab);
    require(bool(result), "large-vocab schema compiles");
    require(result.matcher->accept(128000), "open large-vocab string");
    std::vector<uint32_t> words((vocab->size() + 31) / 32);
    auto start = std::chrono::steady_clock::now();
    require(result.matcher->mask(words.data()) == 128001, "all word tokens and closing quote allowed");
    auto first = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) require(result.matcher->mask(words.data()) == 128001, "cached full vocabulary mask");
    auto last = std::chrono::steady_clock::now();
    std::cout << "grammar-test passed; 128002-token cold mask "
              << std::chrono::duration<double, std::milli>(first - start).count()
              << " ms, cached mean " << std::chrono::duration<double, std::micro>(last - first).count() / 100 << " us\n";
}
