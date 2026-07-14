#pragma once

#include <filesystem>
#include <stdexcept>

#if defined(__linux__)
  #include <unistd.h> // readlink
#endif

namespace iganet_elasticity::utils::paths {

namespace fs = std::filesystem;

inline bool looks_like_repo_root(const fs::path &path) {
    return fs::exists(path / "CMakeLists.txt") &&
           fs::exists(path / "src") &&
           fs::exists(path / "results") &&
           fs::exists(path / "filedata");
}

inline fs::path find_repo_root_upwards(fs::path start) {
    start = fs::absolute(start);
    if (fs::is_regular_file(start)) {
        start = start.parent_path();
    }

    for (fs::path current = start; !current.empty(); current = current.parent_path()) {
        if (looks_like_repo_root(current)) {
            return current;
        }
        if (current == current.root_path()) {
            break;
        }
    }

    throw std::runtime_error("Could not locate elasticity repo root from path: " +
                             start.string());
}

/// Get absolute path to current executable (Linux via /proc/self/exe).
/// Throws if not supported / fails.
inline fs::path exe_path() {
#if defined(__linux__)
    char buf[4096];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) throw std::runtime_error("Failed to read /proc/self/exe");
    buf[len] = '\0';
    return fs::path(buf);
#else
    throw std::runtime_error("exe_path() only implemented for Linux (/proc/self/exe).");
#endif
}

/// Repo root of the elasticity repository.
/// This is determined robustly by searching upwards for repository markers,
/// so it also works for out-of-source build directories.
inline fs::path repo_root_from_build_exe() {
    try {
        return find_repo_root_upwards(fs::current_path());
    } catch (const std::exception &) {
        // fall through and try the executable location next
    }

    return find_repo_root_upwards(exe_path());
}

} // namespace iganet_elasticity::utils::paths
