#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>
inline int h5_ctz(unsigned int bits) {
  unsigned long index;
  return _BitScanForward(&index, bits) ? static_cast<int>(index) : 32;
}
inline int h5_ctzll(unsigned long long bits) {
  unsigned long index;
  return _BitScanForward64(&index, bits) ? static_cast<int>(index) : 64;
}
#define __builtin_ctz h5_ctz
#define __builtin_ctzll h5_ctzll
#define __builtin_popcount __popcnt
#define __builtin_popcountll __popcnt64
inline int setenv(const char* key, const char* value, int overwrite) {
  return !overwrite && std::getenv(key) ? 0 : _putenv_s(key, value);
}
inline int unsetenv(const char* key) { return _putenv_s(key, ""); }
inline void* h5_map_file(const std::string& path, size_t& size) {
  HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("open " + path);
  LARGE_INTEGER length;
  if (!GetFileSizeEx(file, &length) || length.QuadPart <= 0) {
    CloseHandle(file); throw std::runtime_error("file size " + path);
  }
  size = static_cast<size_t>(length.QuadPart);
  HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  CloseHandle(file);
  if (!mapping) throw std::runtime_error("file mapping " + path);
  void* base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(mapping);
  if (!base) throw std::runtime_error("map view " + path);
  return base;
}
inline void h5_unmap_file(void* base, size_t) { UnmapViewOfFile(base); }
inline uint64_t h5_thread_ns() {
  FILETIME created, exited, kernel, user;
  if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user))
    throw std::runtime_error("graph probe: thread clock failed");
  ULARGE_INTEGER k, u;
  k.LowPart=kernel.dwLowDateTime; k.HighPart=kernel.dwHighDateTime;
  u.LowPart=user.dwLowDateTime; u.HighPart=user.dwHighDateTime;
  return (k.QuadPart + u.QuadPart) * 100;
}
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
inline void* h5_map_file(const std::string& path, size_t& size) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open " + path);
  struct stat st;
  if (fstat(fd, &st) || st.st_size <= 0) { close(fd); throw std::runtime_error("file size " + path); }
  size = st.st_size;
  void* base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (base == MAP_FAILED) throw std::runtime_error("mmap " + path);
  return base;
}
inline void h5_unmap_file(void* base, size_t size) { munmap(base, size); }
inline uint64_t h5_thread_ns() {
  timespec t;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t))
    throw std::runtime_error("graph probe: thread clock failed");
  return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
#endif
