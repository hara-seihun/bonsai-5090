#pragma once

namespace h5 {
enum class PathError { None, Geometry, SharedInput, DuplicateBranch };
struct PathChoice { PathError error; int path, matched; };
inline int accepted_prefix(const int* in, const int* out, int depth) {
  int a = 0; while (a < depth - 1 && out[a] == in[a + 1]) a++; return a;
}
inline PathChoice choose_two_path(const int* in, const int* out, int depth, int fork) {
  if (depth < 2 || depth > 8 || fork < 0 || fork >= depth - 1) return {PathError::Geometry, 0, 0};
  for (int k = 0; k <= fork; k++)
    if (in[k] != in[depth + k]) return {PathError::SharedInput, 0, 0};
  if (in[fork + 1] == in[depth + fork + 1]) return {PathError::DuplicateBranch, 0, 0};
  const int primary = accepted_prefix(in, out, depth);
  if (primary < fork || out[fork] != in[depth + fork + 1]) return {PathError::None, 0, primary};
  // The duplicated shared prefix is one logical tree node: primary owns its target.
  // Alternate eligibility uses that exact target, never its duplicate's potentially different argmax.
  int alternate = fork + 1;
  while (alternate < depth - 1 && out[depth + alternate] == in[depth + alternate + 1]) alternate++;
  return {PathError::None, 1, alternate};
}
inline int selected_target(const int* out, int depth, int fork, PathChoice c, int row) {
  return out[c.path && row > fork ? depth + row : row];
}
} // namespace h5
