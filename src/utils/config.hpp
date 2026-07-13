#pragma once

#include <nlohmann/json.hpp>
#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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

enum class optimizer_type { lbfgs, adam };

struct optimizer_config {
    optimizer_type type{optimizer_type::lbfgs};
};

struct boundary_value_2d {
    int side{0};
    double x{0.0};
    double y{0.0};
};

struct patch_config_2d {
    int patch_id{0};
    std::array<double, 2> body_force{0.0, 0.0};
    std::vector<boundary_value_2d> force_sides;
    std::vector<boundary_value_2d> diri_sides;
    std::vector<int> tfbc_sides;
};

struct boundary_value_3d {
    int side{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct patch_config_3d {
    int patch_id{0};
    std::array<double, 3> body_force{0.0, 0.0, 0.0};
    std::vector<boundary_value_3d> force_sides;
    std::vector<boundary_value_3d> diri_sides;
    std::vector<int> tfbc_sides;
};

struct spline_space_config {
    int nr_ctrl_pts{0};
    int degree{0};
};

inline std::vector<boundary_value_2d>
read_boundary_values_2d(const nlohmann::json& values) {
    std::vector<boundary_value_2d> result;
    for (const auto& value : values) {
        result.push_back(boundary_value_2d{
            value.at(0).get<int>(),
            value.at(1).get<double>(),
            value.at(2).get<double>()});
    }
    return result;
}

inline std::vector<boundary_value_3d>
read_boundary_values_3d(const nlohmann::json& values) {
    std::vector<boundary_value_3d> result;
    for (const auto& value : values) {
        result.push_back(boundary_value_3d{
            value.at(0).get<int>(),
            value.at(1).get<double>(),
            value.at(2).get<double>(),
            value.at(3).get<double>()});
    }
    return result;
}

inline optimizer_config load_optimizer_config(const nlohmann::json& j) {
    optimizer_config cfg;
    if (!j.contains("optimizer")) {
        return cfg;
    }

    const auto& oj = j["optimizer"];
    if (!oj.contains("type")) {
        return cfg;
    }

    const auto type = oj["type"].get<std::string>();
    if (type == "lbfgs" || type == "LBFGS") {
        cfg.type = optimizer_type::lbfgs;
    } else if (type == "adam" || type == "Adam" || type == "ADAM") {
        cfg.type = optimizer_type::adam;
    } else {
        throw std::runtime_error(
            "Unsupported optimizer.type '" + type + "'. Expected 'lbfgs' or 'adam'.");
    }

    return cfg;
}

inline spline_space_config load_solution_spline_config(const nlohmann::json& j) {
    if (j.contains("solution_spline")) {
        return spline_space_config{
            require(j, "solution_spline.nr_ctrl_pts").get<int>(),
            require(j, "solution_spline.degree").get<int>()};
    }

    return spline_space_config{
        require(j, "spline.nr_ctrl_pts").get<int>(),
        require(j, "spline.degree").get<int>()};
}

inline spline_space_config load_geometry_spline_config(const nlohmann::json& j) {
    if (j.contains("geometry_spline")) {
        return spline_space_config{
            require(j, "geometry_spline.nr_ctrl_pts").get<int>(),
            require(j, "geometry_spline.degree").get<int>()};
    }

    return load_solution_spline_config(j);
}

inline patch_config_2d load_legacy_single_patch_config_2d(const nlohmann::json& j) {
    patch_config_2d cfg;
    cfg.patch_id = 0;

    if (j.contains("body_force")) {
        const auto& bf = require(j, "body_force");
        cfg.body_force = {bf.at(0).get<double>(), bf.at(1).get<double>()};
    }

    if (j.contains("boundary_conditions")) {
        const auto& bc = j["boundary_conditions"];
        if (bc.contains("force_sides")) {
            cfg.force_sides = read_boundary_values_2d(bc["force_sides"]);
        }
        if (bc.contains("diri_sides")) {
            cfg.diri_sides = read_boundary_values_2d(bc["diri_sides"]);
        }
        if (bc.contains("tfbc_sides")) {
            cfg.tfbc_sides = bc["tfbc_sides"].get<std::vector<int>>();
        }
    }

    return cfg;
}

inline patch_config_2d load_single_patch_config_2d(const nlohmann::json& j) {
    if (!j.contains("single_patch_2D")) {
        return load_legacy_single_patch_config_2d(j);
    }

    const auto& sp = j["single_patch_2D"];
    patch_config_2d cfg;
    cfg.patch_id = 0;

    if (sp.contains("body_force")) {
        const auto& bf = sp["body_force"];
        cfg.body_force = {bf.at(0).get<double>(), bf.at(1).get<double>()};
    }

    if (sp.contains("boundary_conditions")) {
        const auto& bc = sp["boundary_conditions"];
        if (bc.contains("force_sides")) {
            cfg.force_sides = read_boundary_values_2d(bc["force_sides"]);
        }
        if (bc.contains("diri_sides")) {
            cfg.diri_sides = read_boundary_values_2d(bc["diri_sides"]);
        }
        if (bc.contains("tfbc_sides")) {
            cfg.tfbc_sides = bc["tfbc_sides"].get<std::vector<int>>();
        }
    }

    return cfg;
}

inline std::vector<patch_config_2d> load_patch_configs_2d(const nlohmann::json& j) {
    if (j.contains("multipatch_2D") && j["multipatch_2D"].contains("patches")) {
        std::vector<patch_config_2d> result;
        for (const auto& entry : j["multipatch_2D"]["patches"]) {
            patch_config_2d cfg;
            cfg.patch_id = require(entry, "patch_id").get<int>();

            if (entry.contains("body_force")) {
                const auto& bf = entry["body_force"];
                cfg.body_force = {bf.at(0).get<double>(), bf.at(1).get<double>()};
            }

            if (entry.contains("boundary_conditions")) {
                const auto& bc = entry["boundary_conditions"];
                if (bc.contains("force_sides")) {
                    cfg.force_sides = read_boundary_values_2d(bc["force_sides"]);
                }
                if (bc.contains("diri_sides")) {
                    cfg.diri_sides = read_boundary_values_2d(bc["diri_sides"]);
                }
                if (bc.contains("tfbc_sides")) {
                    cfg.tfbc_sides = bc["tfbc_sides"].get<std::vector<int>>();
                }
            }

            result.push_back(std::move(cfg));
        }
        return result;
    }

    if (!j.contains("patches_2d")) {
        return {load_single_patch_config_2d(j)};
    }

    std::vector<patch_config_2d> result;
    for (const auto& entry : j["patches_2d"]) {
        patch_config_2d cfg;
        cfg.patch_id = require(entry, "patch_id").get<int>();

        if (entry.contains("body_force")) {
            const auto& bf = entry["body_force"];
            cfg.body_force = {bf.at(0).get<double>(), bf.at(1).get<double>()};
        }

        if (entry.contains("boundary_conditions")) {
            const auto& bc = entry["boundary_conditions"];
            if (bc.contains("force_sides")) {
                cfg.force_sides = read_boundary_values_2d(bc["force_sides"]);
            }
            if (bc.contains("diri_sides")) {
                cfg.diri_sides = read_boundary_values_2d(bc["diri_sides"]);
            }
            if (bc.contains("tfbc_sides")) {
                cfg.tfbc_sides = bc["tfbc_sides"].get<std::vector<int>>();
            }
        }

        result.push_back(std::move(cfg));
    }

    return result;
}

inline std::vector<patch_config_3d> load_patch_configs_3d(const nlohmann::json& j) {
    const auto* patches = [&]() -> const nlohmann::json* {
        if (j.contains("multipatch_3D") && j["multipatch_3D"].contains("patches")) {
            return &j["multipatch_3D"]["patches"];
        }
        if (j.contains("patches_3d")) {
            return &j["patches_3d"];
        }
        return nullptr;
    }();

    if (patches == nullptr) {
        return {};
    }

    std::vector<patch_config_3d> result;
    for (const auto& entry : *patches) {
        patch_config_3d cfg;
        cfg.patch_id = require(entry, "patch_id").get<int>();

        if (entry.contains("body_force")) {
            const auto& bf = entry["body_force"];
            cfg.body_force = {
                bf.at(0).get<double>(),
                bf.at(1).get<double>(),
                bf.at(2).get<double>()};
        }

        if (entry.contains("boundary_conditions")) {
            const auto& bc = entry["boundary_conditions"];
            if (bc.contains("force_sides")) {
                cfg.force_sides = read_boundary_values_3d(bc["force_sides"]);
            }
            if (bc.contains("diri_sides")) {
                cfg.diri_sides = read_boundary_values_3d(bc["diri_sides"]);
            }
            if (bc.contains("tfbc_sides")) {
                cfg.tfbc_sides = bc["tfbc_sides"].get<std::vector<int>>();
            }
        }

        result.push_back(std::move(cfg));
    }

    return result;
}

} // namespace iganet_elasticity::utils::config
