#pragma once

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace iganet_elasticity::utils::config {

/// Require nested key via "a.b.c". Throws std::runtime_error with a helpful message.
inline const nlohmann::json& require(const nlohmann::json& j, const std::string& path) {
    const nlohmann::json* cur = &j;
    size_t start = 0;

    while (true) {
        size_t dot = path.find('.', start);
        std::string key = (dot == std::string::npos)
            ? path.substr(start)
            : path.substr(start, dot - start);

        if (!cur->is_object() || !cur->contains(key)) {
            throw std::runtime_error("Missing required config key: " + path);
        }

        cur = &((*cur)[key]);

        if (dot == std::string::npos)
            break;

        start = dot + 1;
    }

    return *cur;
}

} // namespace iganet_elasticity::utils::config
