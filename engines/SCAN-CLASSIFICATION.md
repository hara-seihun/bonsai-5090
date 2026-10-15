# Preliminary secret-scan classification

Six generic-api-key matches in the manager's preliminary scan were inspected against source context; all are safe false positives. No credential was removed or allowlist installed. The final whole-repository scan remains separate.

| Path / original line | Classification and retained rationale |
|---|---|
| `halo-5090/tools/two_path_code_check.py:63` | JavaScript synthetic edge-case result property `inherited_key_case_passed`; checks a word-count function with literal `constructor constructor __proto__`. No authentication material. |
| `halo-6000/docs/hybrid-forward-b/forward-comparison.json:6299,6300` | `oracle_token_digest` and `candidate_token_digest`: identical64-hex SHA256 digests of teacher-forced argmax token IDs. The following note explicitly scopes this to teacher forcing. Public benchmark integrity data, no bearer/token credential. |
| `halo-6000/docs/results/native-intrinsic-v1.json:206,245` | `tokens_sha256`:64-hex checksum of the frozen `tokens.i32` input fixture, alongside shape/vocabulary/version. A content hash, not an access token. |
| `halo-6000/src/intrinsic.cpp:30` | Compile-time `TokenHash` is the same fixture SHA256 and validates the intrinsic evaluator's exact input. Adjacent `ConfigId` is another content identifier. Neither authenticates or grants access. |

Line numbers above refer to original source bytes before any portability substitutions. SHA256 digests and tested property names are preserved because changing them would corrupt evidence or the test.
