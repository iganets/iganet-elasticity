#pragma once

#include <filesystem>
#include <stdexcept>

#if defined(__linux__)
  #include <unistd.h> // readlink
#endif

namespace iganet_elasticity::utils::paths {

namespace fs = std::filesystem;

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

/// Repo root assuming binary is located in <repo>/build/<binary>.
/// If exe is not inside a folder named "build", falls back to current working directory.
inline fs::path repo_root_from_build_exe() {
    fs::path exe = exe_path();         // .../repo/build/iganet_lin_elasticity_2D
    fs::path dir = exe.parent_path();  // .../repo/build

    if (dir.filename() == "build") {
        return dir.parent_path();      // .../repo
    }
    return fs::current_path();
}

} // namespace iganet_elasticity::utils::paths
