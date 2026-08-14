#include <iganet/iganet.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <utils/config.hpp>
#include <utils/paths.hpp>

using iganet_elasticity::utils::config::require;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;

namespace {

constexpr int kNumBonePatches = 16;
constexpr double kInterfaceTolerance = 1e-7;
constexpr double kPDEWeight = 1.0;
constexpr double kBoundaryConditionWeight = 1.0;
constexpr double kInterfaceTractionWeight = 10.0;
constexpr int64_t kInteriorCollocationStride = 2;
constexpr int64_t kBoundaryCollocationStride = 2;
constexpr int64_t kInterfaceCollocationStride = 2;
constexpr int64_t kLossPrintClosureStride = 10;

struct SideVector3 {
    int side = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct PatchBoundaryConditions {
    std::vector<SideVector3> diri_sides;
    std::vector<SideVector3> force_sides;
    std::vector<int> tfbc_sides;
};

struct PatchConfig {
    int id = 0;
    std::array<int64_t, 3> ncoeffs{0, 0, 0};
    PatchBoundaryConditions boundary_conditions;
};

struct PatchInterfaceConfig {
    int patch_a = 0;
    int side_a = 0;
    int patch_b = 0;
    int side_b = 0;
};

struct TopDisplacementConfig {
    bool enabled = true;
    double value = 1.0;
    int patch = -1;
    int side = -1;
};

struct ItdPatch {
    int id = 0;
    std::array<int64_t, 3> ncoeffs{0, 0, 0};
    std::array<int, 3> degrees{0, 0, 0};
    std::array<std::vector<double>, 3> knots;
    std::vector<std::array<double, 3>> coeffs;
};

struct FaceDescriptor {
    int patch = 0;
    int side = 0;
};

struct FaceEvalPoints {
    std::array<torch::Tensor, 3> xi;
    std::array<torch::Tensor, 2> eta;
};

struct ControlPointRef {
    int patch = 0;
    int64_t local_index = 0;
};

struct InterfaceOrderCache {
    torch::Tensor order_a;
    torch::Tensor order_b;
};

struct GeometryDerivativeCache {
    iganet::utils::BlockTensor<torch::Tensor, 3, 3, 3> hess;
    iganet::utils::BlockTensor<torch::Tensor, 3, 3> jac_inv;
};

struct SplineDerivativeBasisCache {
    std::array<torch::Tensor, 3> grad;
    std::array<std::array<torch::Tensor, 3>, 3> hess;
};

torch::Tensor strided_indices(int64_t size, int64_t stride, torch::Device device) {
    if (stride <= 1 || size <= 1) {
        return torch::arange(size, torch::TensorOptions().dtype(torch::kInt64).device(device));
    }

    auto indices = torch::arange(0, size, stride,
                                 torch::TensorOptions().dtype(torch::kInt64).device(device));
    const auto last = size - 1;
    if (indices[-1].template item<int64_t>() != last) {
        indices = torch::cat({
            indices,
            torch::tensor({last}, torch::TensorOptions().dtype(torch::kInt64).device(device))
        });
    }
    return indices;
}

template <std::size_t N>
iganet::utils::TensorArray<N> subsample_tensor_array(
    const iganet::utils::TensorArray<N>& values, int64_t stride) {
    if (stride <= 1 || values.empty() || values[0].size(0) <= 2) {
        return values;
    }

    const auto idx = strided_indices(values[0].size(0), stride, values[0].device());
    iganet::utils::TensorArray<N> reduced;
    for (std::size_t i = 0; i < N; ++i) {
        reduced[i] = values[i].index_select(0, idx);
    }
    return reduced;
}

FaceEvalPoints subsample_face_eval_points(const FaceEvalPoints& points,
                                          int64_t stride) {
    if (stride <= 1 || points.xi[0].size(0) <= 2) {
        return points;
    }

    const auto idx = strided_indices(points.xi[0].size(0), stride, points.xi[0].device());
    FaceEvalPoints reduced = points;
    for (auto& xi : reduced.xi) {
        xi = xi.index_select(0, idx);
    }
    for (auto& eta : reduced.eta) {
        eta = eta.index_select(0, idx);
    }
    return reduced;
}

template <std::size_t, typename T>
using repeat_type = T;

template <typename T, std::size_t... Is>
auto repeat_tuple_impl(std::index_sequence<Is...>)
    -> std::tuple<repeat_type<Is, T>...>;

template <typename T, std::size_t N>
using repeat_tuple_t = decltype(repeat_tuple_impl<T>(std::make_index_sequence<N>{}));

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open ITD geometry file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::vector<std::string> tokenize_itd(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());

    for (char c : text) {
        if (c == '[' || c == ']') {
            cleaned.push_back(' ');
        } else {
            cleaned.push_back(c);
        }
    }

    std::istringstream stream(cleaned);
    std::vector<std::string> tokens;
    for (std::string token; stream >> token;) {
        tokens.push_back(token);
    }
    return tokens;
}

double parse_double_token(const std::string& token) {
    std::size_t parsed = 0;
    double value = std::stod(token, &parsed);
    if (parsed != token.size()) {
        throw std::runtime_error("Invalid numeric token in ITD file: " + token);
    }
    return value;
}

int parse_int_token(const std::string& token) {
    return static_cast<int>(std::llround(parse_double_token(token)));
}

std::vector<ItdPatch> load_itd_patches(const std::filesystem::path& path) {
    const auto tokens = tokenize_itd(read_text_file(path));
    std::vector<ItdPatch> patches;

    std::size_t pos = 0;
    while (pos < tokens.size()) {
        if (tokens[pos] != "TRIVAR") {
            ++pos;
            continue;
        }

        if (pos + 9 >= tokens.size() || tokens[pos + 1] != "BSPLINE") {
            throw std::runtime_error("Unsupported ITD object near token " + std::to_string(pos));
        }

        ItdPatch patch;
        patch.id = static_cast<int>(patches.size());
        patch.ncoeffs = {
            parse_int_token(tokens[pos + 2]),
            parse_int_token(tokens[pos + 3]),
            parse_int_token(tokens[pos + 4])
        };

        const std::array<int, 3> orders{
            parse_int_token(tokens[pos + 5]),
            parse_int_token(tokens[pos + 6]),
            parse_int_token(tokens[pos + 7])
        };
        patch.degrees = {orders[0] - 1, orders[1] - 1, orders[2] - 1};

        pos += 9; // TRIVAR BSPLINE n0 n1 n2 order0 order1 order2 E3

        for (int dim = 0; dim < 3; ++dim) {
            if (pos >= tokens.size() || tokens[pos] != "KV") {
                throw std::runtime_error("Expected KV in ITD patch " + std::to_string(patch.id));
            }
            ++pos;

            const auto knotCount =
                static_cast<std::size_t>(patch.ncoeffs[dim] + patch.degrees[dim] + 1);
            patch.knots[dim].reserve(knotCount);
            for (std::size_t i = 0; i < knotCount; ++i) {
                if (pos >= tokens.size()) {
                    throw std::runtime_error("Unexpected end of ITD knot vector.");
                }
                patch.knots[dim].push_back(parse_double_token(tokens[pos++]));
            }
        }

        const auto coeffCount = static_cast<std::size_t>(
            patch.ncoeffs[0] * patch.ncoeffs[1] * patch.ncoeffs[2]);
        patch.coeffs.reserve(coeffCount);
        for (std::size_t i = 0; i < coeffCount; ++i) {
            if (pos + 2 >= tokens.size()) {
                throw std::runtime_error("Unexpected end of ITD coefficients.");
            }
            patch.coeffs.push_back({
                parse_double_token(tokens[pos]),
                parse_double_token(tokens[pos + 1]),
                parse_double_token(tokens[pos + 2])
            });
            pos += 3;
        }

        patches.push_back(std::move(patch));
    }

    if (patches.empty()) {
        throw std::runtime_error("No TRIVAR BSPLINE patches found in ITD file.");
    }

    return patches;
}

std::size_t tensor_index(const std::array<int64_t, 3>& ncoeffs,
                         int64_t i, int64_t j, int64_t k) {
    return static_cast<std::size_t>(i + ncoeffs[0] * (j + ncoeffs[1] * k));
}

void elevate_linear_direction_to_quadratic(ItdPatch& patch, int dim) {
    if (patch.degrees[dim] != 1 || patch.ncoeffs[dim] != 2) {
        throw std::runtime_error(
            "Only 2-control-point linear ITD directions can be elevated automatically.");
    }

    const auto oldN = patch.ncoeffs;
    auto newN = oldN;
    newN[dim] = 3;

    std::vector<std::array<double, 3>> elevated(
        static_cast<std::size_t>(newN[0] * newN[1] * newN[2]));

    auto old_coeff = [&](int64_t i, int64_t j, int64_t k) {
        return patch.coeffs[tensor_index(oldN, i, j, k)];
    };

    for (int64_t k = 0; k < newN[2]; ++k) {
        for (int64_t j = 0; j < newN[1]; ++j) {
            for (int64_t i = 0; i < newN[0]; ++i) {
                std::array<int64_t, 3> idx{i, j, k};
                const int64_t elevatedIndex = idx[dim];

                if (elevatedIndex == 1) {
                    auto lower = idx;
                    auto upper = idx;
                    lower[dim] = 0;
                    upper[dim] = 1;
                    const auto a = old_coeff(lower[0], lower[1], lower[2]);
                    const auto b = old_coeff(upper[0], upper[1], upper[2]);
                    elevated[tensor_index(newN, i, j, k)] = {
                        0.5 * (a[0] + b[0]),
                        0.5 * (a[1] + b[1]),
                        0.5 * (a[2] + b[2])
                    };
                } else {
                    idx[dim] = (elevatedIndex == 0) ? 0 : 1;
                    elevated[tensor_index(newN, i, j, k)] =
                        old_coeff(idx[0], idx[1], idx[2]);
                }
            }
        }
    }

    const double a = patch.knots[dim].front();
    const double b = patch.knots[dim].back();
    patch.ncoeffs = newN;
    patch.degrees[dim] = 2;
    patch.knots[dim] = {a, a, a, b, b, b};
    patch.coeffs = std::move(elevated);
}

void elevate_to_quadratic_tensor_product(ItdPatch& patch) {
    for (int dim = 0; dim < 3; ++dim) {
        if (patch.degrees[dim] == 1) {
            elevate_linear_direction_to_quadratic(patch, dim);
        }
    }
}

void elevate_to_quadratic_tensor_product(std::vector<ItdPatch>& patches) {
    for (auto& patch : patches) {
        elevate_to_quadratic_tensor_product(patch);
    }
}

std::string join_numbers(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::setprecision(17);
    for (double value : values) {
        out << value << ' ';
    }
    return out.str();
}

pugi::xml_document make_patch_xml(const ItdPatch& patch) {
    pugi::xml_document doc;
    auto root = doc.append_child("xml");
    auto geo = root.append_child("Geometry");
    geo.append_attribute("type").set_value("TensorBSpline3");
    geo.append_attribute("id").set_value(0);

    auto bases = geo.append_child("Basis");
    bases.append_attribute("type").set_value("TensorBSplineBasis3");

    for (int dim = 0; dim < 3; ++dim) {
        auto basis = bases.append_child("Basis");
        basis.append_attribute("type").set_value("BSplineBasis");
        basis.append_attribute("index").set_value(dim);

        auto kv = basis.append_child("KnotVector");
        kv.append_attribute("degree").set_value(patch.degrees[dim]);
        kv.append_child(pugi::node_pcdata).set_value(join_numbers(patch.knots[dim]).c_str());
    }

    std::ostringstream coeffStream;
    coeffStream << std::setprecision(17);
    for (const auto& coeff : patch.coeffs) {
        coeffStream << coeff[0] << ' ' << coeff[1] << ' ' << coeff[2] << ' ';
    }
    geo.append_child("coefs").append_child(pugi::node_pcdata).set_value(coeffStream.str().c_str());

    return doc;
}

std::size_t coeff_index(const ItdPatch& patch, int64_t i, int64_t j, int64_t k) {
    return static_cast<std::size_t>(i + patch.ncoeffs[0] * (j + patch.ncoeffs[1] * k));
}

std::vector<std::array<double, 3>> face_coefficients(const ItdPatch& patch, int side) {
    std::vector<std::array<double, 3>> face;

    const auto n0 = patch.ncoeffs[0];
    const auto n1 = patch.ncoeffs[1];
    const auto n2 = patch.ncoeffs[2];

    switch (side) {
        case 1:
        case 2: {
            const int64_t i = (side == 1) ? 0 : n0 - 1;
            face.reserve(static_cast<std::size_t>(n1 * n2));
            for (int64_t k = 0; k < n2; ++k)
                for (int64_t j = 0; j < n1; ++j)
                    face.push_back(patch.coeffs[coeff_index(patch, i, j, k)]);
            break;
        }
        case 3:
        case 4: {
            const int64_t j = (side == 3) ? 0 : n1 - 1;
            face.reserve(static_cast<std::size_t>(n0 * n2));
            for (int64_t k = 0; k < n2; ++k)
                for (int64_t i = 0; i < n0; ++i)
                    face.push_back(patch.coeffs[coeff_index(patch, i, j, k)]);
            break;
        }
        case 5:
        case 6: {
            const int64_t k = (side == 5) ? 0 : n2 - 1;
            face.reserve(static_cast<std::size_t>(n0 * n1));
            for (int64_t j = 0; j < n1; ++j)
                for (int64_t i = 0; i < n0; ++i)
                    face.push_back(patch.coeffs[coeff_index(patch, i, j, k)]);
            break;
        }
        default:
            throw std::invalid_argument("Side must be 1..6.");
    }

    return face;
}

std::vector<int64_t> face_control_point_indices(const std::array<int64_t, 3>& ncoeffs,
                                                int side) {
    std::vector<int64_t> indices;

    const auto n0 = ncoeffs[0];
    const auto n1 = ncoeffs[1];
    const auto n2 = ncoeffs[2];

    switch (side) {
        case 1:
        case 2: {
            const int64_t i = (side == 1) ? 0 : n0 - 1;
            indices.reserve(static_cast<std::size_t>(n1 * n2));
            for (int64_t k = 0; k < n2; ++k)
                for (int64_t j = 0; j < n1; ++j)
                    indices.push_back(static_cast<int64_t>(
                        tensor_index(ncoeffs, i, j, k)));
            break;
        }
        case 3:
        case 4: {
            const int64_t j = (side == 3) ? 0 : n1 - 1;
            indices.reserve(static_cast<std::size_t>(n0 * n2));
            for (int64_t k = 0; k < n2; ++k)
                for (int64_t i = 0; i < n0; ++i)
                    indices.push_back(static_cast<int64_t>(
                        tensor_index(ncoeffs, i, j, k)));
            break;
        }
        case 5:
        case 6: {
            const int64_t k = (side == 5) ? 0 : n2 - 1;
            indices.reserve(static_cast<std::size_t>(n0 * n1));
            for (int64_t j = 0; j < n1; ++j)
                for (int64_t i = 0; i < n0; ++i)
                    indices.push_back(static_cast<int64_t>(
                        tensor_index(ncoeffs, i, j, k)));
            break;
        }
        default:
            throw std::invalid_argument("Side must be 1..6.");
    }

    return indices;
}

void sort_points_lexicographic(std::vector<std::array<double, 3>>& points) {
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
        if (std::abs(a[0] - b[0]) > kInterfaceTolerance) return a[0] < b[0];
        if (std::abs(a[1] - b[1]) > kInterfaceTolerance) return a[1] < b[1];
        return a[2] < b[2];
    });
}

bool same_face_geometry(std::vector<std::array<double, 3>> a,
                        std::vector<std::array<double, 3>> b) {
    if (a.size() != b.size()) {
        return false;
    }

    sort_points_lexicographic(a);
    sort_points_lexicographic(b);

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            if (std::abs(a[i][d] - b[i][d]) > kInterfaceTolerance) {
                return false;
            }
        }
    }
    return true;
}

std::vector<PatchInterfaceConfig> discover_interfaces(const std::vector<ItdPatch>& patches) {
    std::vector<PatchInterfaceConfig> interfaces;

    for (std::size_t a = 0; a < patches.size(); ++a) {
        for (std::size_t b = a + 1; b < patches.size(); ++b) {
            for (int sideA = 1; sideA <= 6; ++sideA) {
                const auto faceA = face_coefficients(patches[a], sideA);
                for (int sideB = 1; sideB <= 6; ++sideB) {
                    if (same_face_geometry(faceA, face_coefficients(patches[b], sideB))) {
                        interfaces.push_back({
                            static_cast<int>(a), sideA, static_cast<int>(b), sideB
                        });
                    }
                }
            }
        }
    }

    return interfaces;
}

bool is_interface_face(const std::vector<PatchInterfaceConfig>& interfaces,
                       int patch, int side) {
    return std::any_of(interfaces.begin(), interfaces.end(), [&](const auto& cfg) {
        return (cfg.patch_a == patch && cfg.side_a == side) ||
               (cfg.patch_b == patch && cfg.side_b == side);
    });
}

double face_min_z(const ItdPatch& patch, int side) {
    const auto face = face_coefficients(patch, side);
    double minZ = std::numeric_limits<double>::infinity();
    for (const auto& point : face) {
        minZ = std::min(minZ, point[2]);
    }
    return minZ;
}

double face_max_z(const ItdPatch& patch, int side) {
    const auto face = face_coefficients(patch, side);
    double maxZ = -std::numeric_limits<double>::infinity();
    for (const auto& point : face) {
        maxZ = std::max(maxZ, point[2]);
    }
    return maxZ;
}

double face_average_z(const ItdPatch& patch, int side) {
    const auto face = face_coefficients(patch, side);
    double sumZ = 0.0;
    for (const auto& point : face) {
        sumZ += point[2];
    }
    return sumZ / static_cast<double>(face.size());
}

std::array<double, 2> global_z_range(const std::vector<ItdPatch>& patches) {
    double minZ = std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();

    for (const auto& patch : patches) {
        for (const auto& point : patch.coeffs) {
            minZ = std::min(minZ, point[2]);
            maxZ = std::max(maxZ, point[2]);
        }
    }

    return {minZ, maxZ};
}

std::vector<PatchConfig> make_patch_configs(const std::vector<ItdPatch>& patches,
                                            const std::vector<PatchInterfaceConfig>& interfaces,
                                            double bottomTolerance,
                                            const TopDisplacementConfig& topDisplacement) {
    const auto zRange = global_z_range(patches);
    std::vector<PatchConfig> configs;
    configs.reserve(patches.size());

    for (std::size_t p = 0; p < patches.size(); ++p) {
        PatchConfig config;
        config.id = static_cast<int>(p);
        config.ncoeffs = patches[p].ncoeffs;

        for (int side = 1; side <= 6; ++side) {
            if (is_interface_face(interfaces, static_cast<int>(p), side)) {
                continue;
            }

            if (face_max_z(patches[p], side) <= zRange[0] + bottomTolerance) {
                config.boundary_conditions.diri_sides.push_back({side, 0.0, 0.0, 0.0});
            } else {
                config.boundary_conditions.tfbc_sides.push_back(side);
            }
        }

        configs.push_back(std::move(config));
    }

    if (topDisplacement.enabled) {
        int targetPatch = topDisplacement.patch;
        int targetSide = topDisplacement.side;

        if (targetPatch < 0 || targetSide < 0) {
            double bestMinZ = -std::numeric_limits<double>::infinity();
            double bestAverageZ = -std::numeric_limits<double>::infinity();

            for (std::size_t p = 0; p < configs.size(); ++p) {
                for (const int side : configs[p].boundary_conditions.tfbc_sides) {
                    const double minZ = face_min_z(patches[p], side);
                    const double averageZ = face_average_z(patches[p], side);
                    if (minZ > bestMinZ ||
                        (std::abs(minZ - bestMinZ) <= bottomTolerance &&
                         averageZ > bestAverageZ)) {
                        bestMinZ = minZ;
                        bestAverageZ = averageZ;
                        targetPatch = static_cast<int>(p);
                        targetSide = side;
                    }
                }
            }
        }

        if (targetPatch < 0 || targetPatch >= static_cast<int>(configs.size()) ||
            targetSide < 1 || targetSide > 6) {
            throw std::runtime_error("Could not determine top displacement boundary face.");
        }

        auto& tfbc = configs[static_cast<std::size_t>(targetPatch)]
                         .boundary_conditions.tfbc_sides;
        const auto it = std::find(tfbc.begin(), tfbc.end(), targetSide);
        if (it == tfbc.end()) {
            throw std::runtime_error(
                "Configured top displacement face is not an exterior traction-free face.");
        }
        tfbc.erase(it);
        configs[static_cast<std::size_t>(targetPatch)]
            .boundary_conditions.diri_sides.push_back(
                {targetSide, 0.0, 0.0, topDisplacement.value});

        std::cout << "Top displacement Dirichlet: patch " << targetPatch
                  << ", side " << targetSide
                  << ", u=(0,0," << topDisplacement.value << ")\n";
    }

    return configs;
}

void append_json_key(const std::string& jsonPath, const std::string& key,
                     const nlohmann::json& data) {
    nlohmann::json jsonData;

    try {
        std::ifstream in(jsonPath);
        if (in.is_open()) {
            in >> jsonData;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not read JSON file " << jsonPath
                  << ": " << e.what() << "\n";
    }

    jsonData[key] = data;

    std::ofstream out(jsonPath);
    if (!out) {
        throw std::runtime_error("Could not write JSON file: " + jsonPath);
    }
    out << jsonData.dump(1);
}

void update_result_json(const std::string& jsonPath,
                        const nlohmann::json& latest,
                        bool appendSnapshot) {
    nlohmann::json jsonData;

    try {
        std::ifstream in(jsonPath);
        if (in.is_open()) {
            in >> jsonData;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not read JSON file " << jsonPath
                  << ": " << e.what() << "\n";
    }

    for (auto it = latest.begin(); it != latest.end(); ++it) {
        jsonData[it.key()] = it.value();
    }

    const nlohmann::json snapshot = {
        {"epoch", latest.value("net_Epoch", -1)},
        {"total_loss", latest.value("net_TotalLoss", 0.0)},
        {"pde_loss", latest.value("net_PDELoss", 0.0)},
        {"bc_loss", latest.value("net_BCLoss", 0.0)},
        {"interface_loss", latest.value("net_InterfaceLoss", 0.0)},
        {"pde_raw_mse", latest.value("net_PDERawMSE", 0.0)},
        {"bc_raw_mse", latest.value("net_BCRawMSE", 0.0)},
        {"interface_raw_mse", latest.value("net_InterfaceRawMSE", 0.0)},
        {"net_Displacements", latest.at("net_Displacements")},
        {"net_CtrlPts", latest.at("net_CtrlPts")}
    };

    if (appendSnapshot) {
        if (!jsonData.contains("net_Snapshots") || !jsonData["net_Snapshots"].is_array()) {
            jsonData["net_Snapshots"] = nlohmann::json::array();
        }
        jsonData["net_Snapshots"].push_back(snapshot);
    }

    std::ofstream out(jsonPath);
    if (!out) {
        throw std::runtime_error("Could not write JSON file: " + jsonPath);
    }
    out << jsonData.dump(1);
}

template <typename Tuple, typename F, std::size_t... Is>
void tuple_for_each_index_impl(Tuple&&, F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
}

template <std::size_t N, typename F>
void for_each_patch_index(F&& f) {
    tuple_for_each_index_impl(std::tuple<>{}, std::forward<F>(f),
                              std::make_index_sequence<N>{});
}

template <std::size_t N, std::size_t... Is>
auto make_ncoeffs_tuple_impl(const std::vector<PatchConfig>& patches,
                             std::index_sequence<Is...>) {
    return std::make_tuple(patches.at(Is).ncoeffs...);
}

template <std::size_t N>
auto make_ncoeffs_tuple(const std::vector<PatchConfig>& patches) {
    return make_ncoeffs_tuple_impl<N>(patches, std::make_index_sequence<N>{});
}

} // namespace

template <typename Optimizer, typename GeometryMap, typename Variable,
          std::size_t NumPatches>
class bone_linear_elasticity
    : public iganet::IgANet<Optimizer, repeat_tuple_t<GeometryMap, NumPatches>,
                            repeat_tuple_t<Variable, NumPatches>>,
      public iganet::IgANetCustomizable<repeat_tuple_t<GeometryMap, NumPatches>,
                                        repeat_tuple_t<Variable, NumPatches>>
{
private:
    using Inputs = repeat_tuple_t<GeometryMap, NumPatches>;
    using Outputs = repeat_tuple_t<Variable, NumPatches>;
    using Base = iganet::IgANet<Optimizer, Inputs, Outputs>;
    using Customizable = iganet::IgANetCustomizable<Inputs, Outputs>;

    struct PatchCache {
        typename Base::template collPts_t<0> collPts;
        typename Base::template collPts_t<0> interiorCollPts;
        typename Customizable::template output_interior_knot_indices_t<0> var_knot_indices;
        typename Customizable::template output_interior_coeff_indices_t<0> var_coeff_indices;
        typename Customizable::template output_interior_knot_indices_t<0> var_knot_indices_interior;
        typename Customizable::template output_interior_coeff_indices_t<0> var_coeff_indices_interior;
        typename Customizable::template input_interior_knot_indices_t<0> G_knot_indices;
        typename Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices;
        typename Customizable::template input_interior_knot_indices_t<0> G_knot_indices_interior;
        typename Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_interior;
        GeometryDerivativeCache geometry_interior;
        SplineDerivativeBasisCache output_basis_interior;
    };

    template <std::size_t Patch>
    auto& patch_input() { return this->template input<Patch>(); }

    template <std::size_t Patch>
    auto& patch_output() { return this->template output<Patch>(); }

    template <std::size_t Patch>
    PatchCache& patch_cache() { return patchCaches_[Patch]; }

    template <std::size_t Patch>
    const PatchCache& patch_cache() const { return patchCaches_[Patch]; }

    static FaceEvalPoints make_face_eval_points(int side,
                                                const std::array<torch::Tensor, 2>& eta) {
        const auto& a = eta[0];
        const auto& b = eta[1];
        switch (side) {
            case 1: return {{{torch::zeros_like(a), a, b}}, eta};
            case 2: return {{{torch::ones_like(a), a, b}}, eta};
            case 3: return {{{a, torch::zeros_like(a), b}}, eta};
            case 4: return {{{a, torch::ones_like(a), b}}, eta};
            case 5: return {{{a, b, torch::zeros_like(a)}}, eta};
            case 6: return {{{a, b, torch::ones_like(a)}}, eta};
            default: throw std::invalid_argument("Side must be 1..6.");
        }
    }

    template <std::size_t Patch>
    FaceEvalPoints side_collocation_points(
        int side, int64_t stride = kBoundaryCollocationStride) const {
        const auto& boundary = patch_cache<Patch>().collPts.second;
        FaceEvalPoints points;
        switch (side) {
            case 1:
                points = make_face_eval_points(side, {std::get<0>(boundary)[0],
                                                      std::get<0>(boundary)[1]});
                break;
            case 2:
                points = make_face_eval_points(side, {std::get<1>(boundary)[0],
                                                      std::get<1>(boundary)[1]});
                break;
            case 3:
                points = make_face_eval_points(side, {std::get<2>(boundary)[0],
                                                      std::get<2>(boundary)[1]});
                break;
            case 4:
                points = make_face_eval_points(side, {std::get<3>(boundary)[0],
                                                      std::get<3>(boundary)[1]});
                break;
            case 5:
                points = make_face_eval_points(side, {std::get<4>(boundary)[0],
                                                      std::get<4>(boundary)[1]});
                break;
            case 6:
                points = make_face_eval_points(side, {std::get<5>(boundary)[0],
                                                      std::get<5>(boundary)[1]});
                break;
            default: throw std::invalid_argument("Side must be 1..6.");
        }
        return subsample_face_eval_points(points, stride);
    }

    torch::Tensor sort_indices_by_physical_position(std::size_t patchIndex,
                                                    const FaceEvalPoints& points) {
        torch::Tensor coords;
        bool found = false;

        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            if (patchIndex == Patch) {
                auto values = patch_input<Patch>().eval(points.xi);
                coords = torch::stack({*values[0], *values[1], *values[2]}, 1).detach().cpu();
                found = true;
            }
        });

        if (!found) {
            throw std::runtime_error("Patch index out of range.");
        }

        std::vector<int64_t> order(static_cast<std::size_t>(coords.size(0)));
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
            const double lx = coords[lhs][0].item<double>();
            const double ly = coords[lhs][1].item<double>();
            const double lz = coords[lhs][2].item<double>();
            const double rx = coords[rhs][0].item<double>();
            const double ry = coords[rhs][1].item<double>();
            const double rz = coords[rhs][2].item<double>();

            if (std::abs(lx - rx) > kInterfaceTolerance) return lx < rx;
            if (std::abs(ly - ry) > kInterfaceTolerance) return ly < ry;
            return lz < rz;
        });

        return torch::tensor(order, torch::TensorOptions().dtype(torch::kInt64)
                                      .device(points.xi[0].device()));
    }

    template <std::size_t Patch>
    torch::Tensor displacement_tensor(const FaceEvalPoints& points) {
        auto disp = patch_output<Patch>().eval(points.xi);
        return torch::stack({*disp[0], *disp[1], *disp[2]}, 1);
    }

    template <std::size_t Patch, int Side>
    std::array<torch::Tensor, 3> boundary_normal(const FaceEvalPoints& points) {
        auto nv = patch_input<Patch>().boundary().template side<Side>().nv(points.eta);
        auto nx = *nv[0];
        auto ny = *nv[1];
        auto nz = *nv[2];
        auto norm = torch::sqrt(nx * nx + ny * ny + nz * nz).clamp_min(1e-14);
        return {nx / norm, ny / norm, nz / norm};
    }

    template <std::size_t Patch>
    std::array<torch::Tensor, 3> boundary_normal(int side, const FaceEvalPoints& points) {
        switch (side) {
            case 1: return boundary_normal<Patch, 1>(points);
            case 2: return boundary_normal<Patch, 2>(points);
            case 3: return boundary_normal<Patch, 3>(points);
            case 4: return boundary_normal<Patch, 4>(points);
            case 5: return boundary_normal<Patch, 5>(points);
            case 6: return boundary_normal<Patch, 6>(points);
            default: throw std::invalid_argument("Side must be 1..6.");
        }
    }

    template <std::size_t Patch>
    torch::Tensor traction_tensor(int side, const FaceEvalPoints& points) {
        auto varKnot =
            patch_output<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                points.xi);
        auto varCoeff =
            patch_output<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                varKnot);
        auto geoKnot =
            patch_input<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                points.xi);
        auto geoCoeff =
            patch_input<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                geoKnot);

        auto jac = patch_output<Patch>().ijac(
            patch_input<Patch>(), points.xi, varKnot, varCoeff, geoKnot, geoCoeff);

        auto ux_x = *jac[0];
        auto ux_y = *jac[1];
        auto ux_z = *jac[2];
        auto uy_x = *jac[3];
        auto uy_y = *jac[4];
        auto uy_z = *jac[5];
        auto uz_x = *jac[6];
        auto uz_y = *jac[7];
        auto uz_z = *jac[8];

        const auto tr = ux_x + uy_y + uz_z;
        const auto sigma_xx = lambda_ * tr + 2.0 * mu_ * ux_x;
        const auto sigma_yy = lambda_ * tr + 2.0 * mu_ * uy_y;
        const auto sigma_zz = lambda_ * tr + 2.0 * mu_ * uz_z;
        const auto sigma_xy = mu_ * (ux_y + uy_x);
        const auto sigma_xz = mu_ * (ux_z + uz_x);
        const auto sigma_yz = mu_ * (uy_z + uz_y);

        const auto normal = boundary_normal<Patch>(side, points);
        const auto& nx = normal[0];
        const auto& ny = normal[1];
        const auto& nz = normal[2];

        return torch::stack({
            sigma_xx * nx + sigma_xy * ny + sigma_xz * nz,
            sigma_xy * nx + sigma_yy * ny + sigma_yz * nz,
            sigma_xz * nx + sigma_yz * ny + sigma_zz * nz
        }, 1);
    }

    template <std::size_t Patch>
    iganet::utils::BlockTensor<torch::Tensor, 3, 3, 3>
    physical_hessian_with_cached_geometry() {
        auto& cache = patch_cache<Patch>();

        const auto numEval = cache.interiorCollPts.first[0].numel();
        const auto sizes = cache.interiorCollPts.first[0].sizes();
        auto evalPrecomputed = [&](const torch::Tensor& basis) {
            return patch_output<Patch>().eval_from_precomputed(
                basis, cache.var_coeff_indices_interior, numEval, sizes);
        };

        iganet::utils::BlockTensor<torch::Tensor, 3, 3> paramJac;
        for (int derivDim = 0; derivDim < 3; ++derivDim) {
            auto values = evalPrecomputed(cache.output_basis_interior.grad[derivDim]);
            for (int component = 0; component < 3; ++component) {
                paramJac.set(component, derivDim, values(0, component));
            }
        }

        iganet::utils::BlockTensor<torch::Tensor, 3, 3, 3> paramHess;
        for (int derivA = 0; derivA < 3; ++derivA) {
            for (int derivB = 0; derivB < 3; ++derivB) {
                auto values = evalPrecomputed(cache.output_basis_interior.hess[derivA][derivB]);
                for (int component = 0; component < 3; ++component) {
                    paramHess.set(derivA, derivB, component, values(0, component));
                }
            }
        }

        auto physicalJac = paramJac * cache.geometry_interior.jac_inv;

        iganet::utils::BlockTensor<torch::Tensor, 3, 3, 3> physicalHess;

        for (int component = 0; component < 3; ++component) {
            auto hessComponent = paramHess.slice(component);

            for (int geomComponent = 0; geomComponent < 3; ++geomComponent) {
                hessComponent -= physicalJac(component, geomComponent) *
                                 cache.geometry_interior.hess.slice(geomComponent);
            }

            auto transformed = cache.geometry_interior.jac_inv.tr() *
                               hessComponent *
                               cache.geometry_interior.jac_inv;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    physicalHess.set(i, j, component, transformed(i, j));
                }
            }
        }

        return physicalHess;
    }

    template <std::size_t Patch>
    torch::Tensor compute_patch_pde_loss() {
        auto& cache = patch_cache<Patch>();
        auto hess = physical_hessian_with_cached_geometry<Patch>();

        const auto lapUx = hess(0, 0, 0) + hess(1, 1, 0) + hess(2, 2, 0);
        const auto lapUy = hess(0, 0, 1) + hess(1, 1, 1) + hess(2, 2, 1);
        const auto lapUz = hess(0, 0, 2) + hess(1, 1, 2) + hess(2, 2, 2);

        const auto dDivDx = hess(0, 0, 0) + hess(0, 1, 1) + hess(0, 2, 2);
        const auto dDivDy = hess(1, 0, 0) + hess(1, 1, 1) + hess(1, 2, 2);
        const auto dDivDz = hess(2, 0, 0) + hess(2, 1, 1) + hess(2, 2, 2);

        auto divStress = torch::stack({
            mu_ * lapUx + (lambda_ + mu_) * dDivDx,
            mu_ * lapUy + (lambda_ + mu_) * dDivDy,
            mu_ * lapUz + (lambda_ + mu_) * dDivDz
        }, 1);

        auto bodyForce = bodyForceRow_.expand({divStress.size(0), 3});

        return torch::mse_loss(divStress, -bodyForce);
    }

    template <std::size_t Patch>
    torch::Tensor compute_patch_boundary_loss(const torch::TensorOptions& opts) {
        const auto& bc = patches_[Patch].boundary_conditions;
        auto loss = torch::zeros({}, opts);

        for (int side : bc.tfbc_sides) {
            auto points = side_collocation_points<Patch>(side, kBoundaryCollocationStride);
            auto traction = traction_tensor<Patch>(side, points);
            loss += torch::mse_loss(traction, torch::zeros_like(traction));
        }

        for (const auto& force : bc.force_sides) {
            auto points = side_collocation_points<Patch>(force.side, kBoundaryCollocationStride);
            auto traction = traction_tensor<Patch>(force.side, points);
            auto target = torch::zeros_like(traction);
            target.slice(1, 0, 1).fill_(force.x);
            target.slice(1, 1, 2).fill_(force.y);
            target.slice(1, 2, 3).fill_(force.z);
            loss += torch::mse_loss(traction, target);
        }

        return loss;
    }

    template <std::size_t PatchA, std::size_t PatchB>
    torch::Tensor compute_interface_raw_loss_for(const PatchInterfaceConfig& cfg,
                                                 const InterfaceOrderCache& orderCache,
                                                 const torch::TensorOptions& opts) {
        auto ptsA = side_collocation_points<PatchA>(cfg.side_a, kInterfaceCollocationStride);
        auto ptsB = side_collocation_points<PatchB>(cfg.side_b, kInterfaceCollocationStride);

        if (ptsA.xi[0].size(0) != ptsB.xi[0].size(0)) {
            throw std::runtime_error("Interface collocation point counts do not match.");
        }

        auto tracA = traction_tensor<PatchA>(cfg.side_a, ptsA)
                         .index_select(0, orderCache.order_a);
        auto tracB = traction_tensor<PatchB>(cfg.side_b, ptsB)
                         .index_select(0, orderCache.order_b);

        return torch::mse_loss(tracA + tracB, torch::zeros_like(tracA)) +
               torch::zeros({}, opts);
    }

    torch::Tensor compute_interface_raw_loss(const torch::TensorOptions& opts) {
        auto loss = torch::zeros({}, opts);

        for (std::size_t interfaceIndex = 0; interfaceIndex < interfaces_.size(); ++interfaceIndex) {
            const auto& cfg = interfaces_[interfaceIndex];
            const auto& orderCache = interfaceOrderCaches_.at(interfaceIndex);
            bool found = false;
            for_each_patch_index<NumPatches>([&](auto PA) {
                for_each_patch_index<NumPatches>([&](auto PB) {
                    constexpr std::size_t PatchA = decltype(PA)::value;
                    constexpr std::size_t PatchB = decltype(PB)::value;
                    if (cfg.patch_a == static_cast<int>(PatchA) &&
                        cfg.patch_b == static_cast<int>(PatchB)) {
                        loss += compute_interface_raw_loss_for<PatchA, PatchB>(
                            cfg, orderCache, opts);
                        found = true;
                    }
                });
            });
            if (!found) {
                throw std::runtime_error("Interface patch index out of range.");
            }
        }

        return loss;
    }

    std::array<long long, 3> control_point_key(const torch::Tensor& geometryTensor,
                                               const PatchConfig& patch,
                                               int64_t localIndex) const {
        const auto n = patch.ncoeffs[0] * patch.ncoeffs[1] * patch.ncoeffs[2];
        return {
            static_cast<long long>(std::llround(
                geometryTensor[localIndex].template item<double>() / kInterfaceTolerance)),
            static_cast<long long>(std::llround(
                geometryTensor[localIndex + n].template item<double>() / kInterfaceTolerance)),
            static_cast<long long>(std::llround(
                geometryTensor[localIndex + 2 * n].template item<double>() / kInterfaceTolerance))
        };
    }

    void build_strong_coupling_groups() {
        std::vector<torch::Tensor> geometryTensors(NumPatches);
        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            geometryTensors[Patch] = patch_input<Patch>().as_tensor().detach().cpu();
        });

        std::map<std::array<long long, 3>, std::vector<ControlPointRef>> byPosition;

        for (std::size_t patchIndex = 0; patchIndex < patches_.size(); ++patchIndex) {
            const auto& patch = patches_.at(patchIndex);
            const auto n = patch.ncoeffs[0] * patch.ncoeffs[1] * patch.ncoeffs[2];
            for (int64_t localIndex = 0; localIndex < n; ++localIndex) {
                byPosition[control_point_key(
                    geometryTensors.at(patchIndex), patch, localIndex)]
                    .push_back({static_cast<int>(patchIndex), localIndex});
            }
        }

        strongCouplingGroups_.clear();
        strongCouplingGroups_.reserve(byPosition.size());

        for (auto& [_, refs] : byPosition) {
            std::sort(refs.begin(), refs.end(), [](const auto& a, const auto& b) {
                if (a.patch != b.patch) return a.patch < b.patch;
                return a.local_index < b.local_index;
            });
            refs.erase(std::unique(refs.begin(), refs.end(), [](const auto& a, const auto& b) {
                return a.patch == b.patch && a.local_index == b.local_index;
            }), refs.end());

            const auto firstPatch = refs.front().patch;
            const bool crossesPatches = std::any_of(
                refs.begin(), refs.end(), [&](const auto& ref) {
                    return ref.patch != firstPatch;
                });

            if (refs.size() > 1 && crossesPatches) {
                strongCouplingGroups_.push_back(refs);
            }
        }

        std::size_t coupledRefs = 0;
        for (const auto& group : strongCouplingGroups_) {
            coupledRefs += group.size();
        }

        std::cout << "Strong C0 coupling: " << strongCouplingGroups_.size()
                  << " control-point groups, " << coupledRefs
                  << " patch-local references." << std::endl;
    }

    void build_strong_dirichlet_values() {
        strongDirichletValues_.clear();
        strongDirichletValues_.resize(patches_.size());

        auto assign_value = [&](int patchIndex, int component, int64_t localIndex,
                                double value) {
            auto& values = strongDirichletValues_.at(static_cast<std::size_t>(patchIndex))
                               .at(static_cast<std::size_t>(component));
            const auto [it, inserted] = values.emplace(localIndex, value);
            if (!inserted && std::abs(it->second - value) > 1e-12) {
                throw std::runtime_error(
                    "Conflicting strong Dirichlet values on the same control point.");
            }
        };

        std::size_t constrainedDofs = 0;
        for (std::size_t patchIndex = 0; patchIndex < patches_.size(); ++patchIndex) {
            const auto& patch = patches_[patchIndex];
            for (const auto& side : patch.boundary_conditions.diri_sides) {
                const auto indices = face_control_point_indices(patch.ncoeffs, side.side);
                for (const auto localIndex : indices) {
                    assign_value(static_cast<int>(patchIndex), 0, localIndex, side.x);
                    assign_value(static_cast<int>(patchIndex), 1, localIndex, side.y);
                    assign_value(static_cast<int>(patchIndex), 2, localIndex, side.z);
                    constrainedDofs += 3;
                }
            }
        }

        std::cout << "Strong Dirichlet constraints: " << constrainedDofs
                  << " patch-local component DOFs." << std::endl;
    }

    std::optional<double> prescribed_dirichlet_value(const ControlPointRef& ref,
                                                     int component) const {
        if (ref.patch < 0 ||
            static_cast<std::size_t>(ref.patch) >= strongDirichletValues_.size()) {
            return std::nullopt;
        }

        const auto& values =
            strongDirichletValues_[static_cast<std::size_t>(ref.patch)]
                                  [static_cast<std::size_t>(component)];
        const auto it = values.find(ref.local_index);
        if (it == values.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    template <std::size_t PatchA, std::size_t PatchB>
    void initialize_interface_order_cache_for(std::size_t interfaceIndex) {
        const auto& cfg = interfaces_.at(interfaceIndex);
        auto ptsA = side_collocation_points<PatchA>(cfg.side_a, kInterfaceCollocationStride);
        auto ptsB = side_collocation_points<PatchB>(cfg.side_b, kInterfaceCollocationStride);

        if (ptsA.xi[0].size(0) != ptsB.xi[0].size(0)) {
            throw std::runtime_error("Interface collocation point counts do not match.");
        }

        interfaceOrderCaches_.at(interfaceIndex) = {
            sort_indices_by_physical_position(PatchA, ptsA),
            sort_indices_by_physical_position(PatchB, ptsB)
        };
    }

    void initialize_interface_order_caches() {
        interfaceOrderCaches_.resize(interfaces_.size());

        for (std::size_t interfaceIndex = 0; interfaceIndex < interfaces_.size(); ++interfaceIndex) {
            const auto& cfg = interfaces_[interfaceIndex];
            bool found = false;
            for_each_patch_index<NumPatches>([&](auto PA) {
                for_each_patch_index<NumPatches>([&](auto PB) {
                    constexpr std::size_t PatchA = decltype(PA)::value;
                    constexpr std::size_t PatchB = decltype(PB)::value;
                    if (cfg.patch_a == static_cast<int>(PatchA) &&
                        cfg.patch_b == static_cast<int>(PatchB)) {
                        initialize_interface_order_cache_for<PatchA, PatchB>(interfaceIndex);
                        found = true;
                    }
                });
            });
            if (!found) {
                throw std::runtime_error("Interface patch index out of range.");
            }
        }
    }

    void enforce_strong_coupling(std::vector<torch::Tensor>& patchTensors) const {
        for (const auto& group : strongCouplingGroups_) {
            for (int component = 0; component < 3; ++component) {
                std::optional<double> prescribedValue;
                for (const auto& ref : group) {
                    const auto value = prescribed_dirichlet_value(ref, component);
                    if (!value.has_value()) {
                        continue;
                    }
                    if (prescribedValue.has_value() &&
                        std::abs(*prescribedValue - *value) > 1e-12) {
                        throw std::runtime_error(
                            "Conflicting strong Dirichlet values in one coupling group.");
                    }
                    prescribedValue = *value;
                }

                torch::Tensor coupledValue;
                if (prescribedValue.has_value()) {
                    coupledValue = torch::full(
                        {1}, *prescribedValue, patchTensors.front().options());
                } else {
                    std::vector<torch::Tensor> values;
                    values.reserve(group.size());

                    for (const auto& ref : group) {
                        const auto& patch = patches_.at(static_cast<std::size_t>(ref.patch));
                        const auto n = patch.ncoeffs[0] * patch.ncoeffs[1] * patch.ncoeffs[2];
                        values.push_back(patchTensors.at(static_cast<std::size_t>(ref.patch))
                                             .slice(0, component * n + ref.local_index,
                                                    component * n + ref.local_index + 1));
                    }

                    coupledValue = torch::stack(values).mean().reshape({1});
                }

                for (const auto& ref : group) {
                    const auto& patch = patches_.at(static_cast<std::size_t>(ref.patch));
                    const auto n = patch.ncoeffs[0] * patch.ncoeffs[1] * patch.ncoeffs[2];
                    patchTensors.at(static_cast<std::size_t>(ref.patch))
                        .slice(0, component * n + ref.local_index,
                               component * n + ref.local_index + 1)
                        .copy_(coupledValue);
                }
            }
        }
    }

    void enforce_strong_dirichlet(std::vector<torch::Tensor>& patchTensors) const {
        for (std::size_t patchIndex = 0; patchIndex < strongDirichletValues_.size(); ++patchIndex) {
            const auto& patch = patches_[patchIndex];
            const auto n = patch.ncoeffs[0] * patch.ncoeffs[1] * patch.ncoeffs[2];

            for (int component = 0; component < 3; ++component) {
                const auto& values = strongDirichletValues_[patchIndex]
                                                         [static_cast<std::size_t>(component)];
                if (values.empty()) {
                    continue;
                }

                std::vector<int64_t> indices;
                std::vector<double> prescribed;
                indices.reserve(values.size());
                prescribed.reserve(values.size());
                for (const auto& [localIndex, value] : values) {
                    indices.push_back(localIndex);
                    prescribed.push_back(value);
                }

                const torch::Tensor indexTensor = torch::tensor(
                    indices,
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(patchTensors[patchIndex].device()));
                const torch::Tensor valueTensor = torch::tensor(
                    prescribed,
                    patchTensors[patchIndex].options());

                patchTensors[patchIndex]
                    .slice(0, component * n, (component + 1) * n)
                    .index_put_({indexTensor}, valueTensor);
            }
        }
    }

    void assign_outputs_from_tensor(const torch::Tensor& outputs) {
        int64_t offset = 0;
        std::vector<torch::Tensor> patchTensors(NumPatches);

        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            const auto patchSize = patch_output<Patch>().as_tensor().size(0);
            patchTensors[Patch] = outputs.slice(0, offset, offset + patchSize).clone();
            offset += patchSize;
        });

        enforce_strong_coupling(patchTensors);
        enforce_strong_dirichlet(patchTensors);

        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            patch_output<Patch>().from_tensor(patchTensors[Patch]);
        });
    }

    template <std::size_t Patch>
    void initialize_output_derivative_basis_cache() {
        auto& cache = patch_cache<Patch>();
        auto& output = patch_output<Patch>();
        const auto& xi = cache.interiorCollPts.first;
        const auto& knot = cache.var_knot_indices_interior;

        cache.output_basis_interior.grad[0] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dx>(xi, knot).detach();
        cache.output_basis_interior.grad[1] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dy>(xi, knot).detach();
        cache.output_basis_interior.grad[2] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dz>(xi, knot).detach();

        cache.output_basis_interior.hess[0][0] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dx + iganet::deriv::dx>(xi, knot).detach();
        cache.output_basis_interior.hess[0][1] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dx + iganet::deriv::dy>(xi, knot).detach();
        cache.output_basis_interior.hess[0][2] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dx + iganet::deriv::dz>(xi, knot).detach();
        cache.output_basis_interior.hess[1][0] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dy + iganet::deriv::dx>(xi, knot).detach();
        cache.output_basis_interior.hess[1][1] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dy + iganet::deriv::dy>(xi, knot).detach();
        cache.output_basis_interior.hess[1][2] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dy + iganet::deriv::dz>(xi, knot).detach();
        cache.output_basis_interior.hess[2][0] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dz + iganet::deriv::dx>(xi, knot).detach();
        cache.output_basis_interior.hess[2][1] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dz + iganet::deriv::dy>(xi, knot).detach();
        cache.output_basis_interior.hess[2][2] =
            output.template eval_basfunc<iganet::functionspace::interior, iganet::deriv::dz + iganet::deriv::dz>(xi, knot).detach();
    }

    template <std::size_t Patch>
    void initialize_geometry_derivative_cache() {
        auto& cache = patch_cache<Patch>();
        auto hessG = patch_input<Patch>().hess(
            cache.interiorCollPts.first,
            cache.G_knot_indices_interior,
            cache.G_coeff_indices_interior);
        auto jacInvG = patch_input<Patch>().jac(
            cache.interiorCollPts.first,
            cache.G_knot_indices_interior,
            cache.G_coeff_indices_interior).ginv();

        for (int component = 0; component < 3; ++component) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    cache.geometry_interior.hess.set(
                        i, j, component, hessG(i, j, component).detach());
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                cache.geometry_interior.jac_inv.set(i, j, jacInvG(i, j).detach());
            }
        }
    }

    template <std::size_t Patch>
    void initialize_patch_data() {
        auto& cache = patch_cache<Patch>();
        cache.collPts = Base::template collPts<Patch>(iganet::collPts::greville);
        cache.interiorCollPts = Base::template collPts<Patch>(iganet::collPts::greville_interior);
        cache.interiorCollPts.first =
            subsample_tensor_array(cache.interiorCollPts.first, kInteriorCollocationStride);

        cache.var_knot_indices =
            patch_output<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.collPts.first);
        cache.var_coeff_indices =
            patch_output<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.var_knot_indices);
        cache.var_knot_indices_interior =
            patch_output<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.interiorCollPts.first);
        cache.var_coeff_indices_interior =
            patch_output<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.var_knot_indices_interior);
        cache.G_knot_indices =
            patch_input<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.collPts.first);
        cache.G_coeff_indices =
            patch_input<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.G_knot_indices);
        cache.G_knot_indices_interior =
            patch_input<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.interiorCollPts.first);
        cache.G_coeff_indices_interior =
            patch_input<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.G_knot_indices_interior);

        initialize_output_derivative_basis_cache<Patch>();
        initialize_geometry_derivative_cache<Patch>();
    }

    template <std::size_t Patch>
    void load_patch_geometry(const ItdPatch& patch) {
        auto doc = make_patch_xml(patch);
        patch_input<Patch>().from_xml(doc);
    }

    template <std::size_t Patch>
    void append_patch_postproc(nlohmann::json& origin, nlohmann::json& disp,
                               nlohmann::json& deformed, nlohmann::json& patchIds) {
        const auto n = patches_[Patch].ncoeffs[0] * patches_[Patch].ncoeffs[1] *
                       patches_[Patch].ncoeffs[2];
        auto geometryTensor = patch_input<Patch>().as_tensor();
        auto displacementTensor = patch_output<Patch>().as_tensor();

        for (int64_t i = 0; i < n; ++i) {
            const double x = geometryTensor[i].template item<double>();
            const double y = geometryTensor[i + n].template item<double>();
            const double z = geometryTensor[i + 2 * n].template item<double>();
            const double ux = displacementTensor[i].template item<double>();
            const double uy = displacementTensor[i + n].template item<double>();
            const double uz = displacementTensor[i + 2 * n].template item<double>();

            origin.push_back({x, y, z});
            disp.push_back({ux, uy, uz});
            deformed.push_back({x + ux, y + uy, z + uz});
            patchIds.push_back(static_cast<int>(Patch));
        }
    }

    double lambda_ = 0.0;
    double mu_ = 0.0;
    std::array<double, 3> BODY_FORCE_{0.0, 0.0, 1.0};
    std::vector<PatchConfig> patches_;
    std::vector<PatchInterfaceConfig> interfaces_;
    std::vector<InterfaceOrderCache> interfaceOrderCaches_;
    std::vector<std::vector<ControlPointRef>> strongCouplingGroups_;
    std::vector<std::array<std::map<int64_t, double>, 3>> strongDirichletValues_;
    std::array<PatchCache, NumPatches> patchCaches_;
    torch::Tensor bodyForceRow_;
    int maxEpoch_ = 0;
    double minLoss_ = 0.0;
    std::string jsonPath_;
    std::string optimizerName_;
    double learningRate_ = 1.0;
    int64_t lastExportEpoch_ = -1;
    int64_t closureEvalCount_ = 0;
    double lastTotalLoss_ = 0.0;
    double lastPDELoss_ = 0.0;
    double lastBCLoss_ = 0.0;
    double lastInterfaceLoss_ = 0.0;
    double lastPDERawLoss_ = 0.0;
    double lastBCRawLoss_ = 0.0;
    double lastInterfaceRawLoss_ = 0.0;
    std::string deviceName_ = "unknown";

public:
    template <typename... Args>
    bone_linear_elasticity(double lambda, double mu, std::array<double, 3> bodyForce,
                           std::vector<PatchConfig> patches,
                           std::vector<PatchInterfaceConfig> interfaces,
                           int maxEpoch, double minLoss, std::string jsonPath,
                           std::string optimizerName, double learningRate,
                           std::vector<int64_t>&& layers,
                           std::vector<std::vector<std::any>>&& activations,
                           Args&&... args)
        : Base(std::forward<std::vector<int64_t>>(layers),
               std::forward<std::vector<std::vector<std::any>>>(activations),
               std::forward<Args>(args)...),
          lambda_(lambda),
          mu_(mu),
          BODY_FORCE_(bodyForce),
          patches_(std::move(patches)),
          interfaces_(std::move(interfaces)),
          maxEpoch_(maxEpoch),
          minLoss_(minLoss),
          jsonPath_(std::move(jsonPath)),
          optimizerName_(std::move(optimizerName)),
          learningRate_(learningRate) {}

    void load_geometry(const std::vector<ItdPatch>& patches) {
        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            load_patch_geometry<Patch>(patches.at(Patch));
        });
        deviceName_ = patch_input<0>().as_tensor().device().str();
        build_strong_dirichlet_values();
        build_strong_coupling_groups();
    }

    void initialize_problem_data() {
        bodyForceRow_ = torch::tensor(
            {BODY_FORCE_[0], BODY_FORCE_[1], BODY_FORCE_[2]},
            patch_input<0>().as_tensor().options()).view({1, 3});

        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            initialize_patch_data<Patch>();
        });
        initialize_interface_order_caches();
    }

    bool epoch(int64_t epoch) override {
        std::cout << "Epoch: " << epoch << std::endl;
        return epoch == 0;
    }

    torch::Tensor loss(const torch::Tensor& outputs, int64_t epoch) override {
        ++closureEvalCount_;
        assign_outputs_from_tensor(outputs);

        auto rawPDE = torch::zeros({}, outputs.options());
        auto rawBC = torch::zeros({}, outputs.options());

        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            rawPDE += compute_patch_pde_loss<Patch>();
            rawBC += compute_patch_boundary_loss<Patch>(outputs.options());
        });

        auto rawINTER = compute_interface_raw_loss(outputs.options());
        auto lossPDE = kPDEWeight * rawPDE;
        auto lossBC = kBoundaryConditionWeight * rawBC;
        auto lossINTER = kInterfaceTractionWeight * rawINTER;
        auto totalLoss = lossPDE + lossBC + lossINTER;

        const bool shouldExport = ((epoch % 10 == 0) ||
                                   (epoch == maxEpoch_ - 1)) &&
                                  epoch != lastExportEpoch_;
        const bool shouldPrint =
            (closureEvalCount_ == 1) ||
            (closureEvalCount_ % kLossPrintClosureStride == 0) ||
            shouldExport;

        if (shouldPrint || shouldExport) {
            const double rawPDEValue = rawPDE.template item<double>();
            const double rawBCValue = rawBC.template item<double>();
            const double rawINTERValue = rawINTER.template item<double>();
            const double lossPDEValue = lossPDE.template item<double>();
            const double lossBCValue = lossBC.template item<double>();
            const double lossINTERValue = lossINTER.template item<double>();
            const double totalLossValue = totalLoss.template item<double>();

            lastTotalLoss_ = totalLossValue;
            lastPDELoss_ = lossPDEValue;
            lastBCLoss_ = lossBCValue;
            lastInterfaceLoss_ = lossINTERValue;
            lastPDERawLoss_ = rawPDEValue;
            lastBCRawLoss_ = rawBCValue;
            lastInterfaceRawLoss_ = rawINTERValue;

            if (shouldPrint) {
                std::cout << "loss " << std::setw(11) << totalLossValue
                          << " | PDE mse " << std::setw(10) << rawPDEValue
                          << " * " << kPDEWeight << " = " << std::setw(10) << lossPDEValue
                          << " | BC mse " << std::setw(10) << rawBCValue
                          << " * " << kBoundaryConditionWeight << " = " << std::setw(10) << lossBCValue
                          << " | IF mse " << std::setw(10) << rawINTERValue
                          << " * " << kInterfaceTractionWeight << " = " << std::setw(10) << lossINTERValue
                          << " | closure " << closureEvalCount_
                          << std::endl;
            }

            if (shouldExport) {
                write_result(epoch, totalLoss, lossPDE, lossBC, lossINTER, true);
                lastExportEpoch_ = epoch;
            }
        }

        return totalLoss;
    }

    nlohmann::json make_result_json(int64_t epoch, const torch::Tensor& totalLoss,
                                    const torch::Tensor& lossPDE,
                                    const torch::Tensor& lossBC,
                                    const torch::Tensor& lossINTER) {
        nlohmann::json origin = nlohmann::json::array();
        nlohmann::json disp = nlohmann::json::array();
        nlohmann::json deformed = nlohmann::json::array();
        nlohmann::json patchIds = nlohmann::json::array();

        for_each_patch_index<NumPatches>([&](auto P) {
            constexpr std::size_t Patch = decltype(P)::value;
            append_patch_postproc<Patch>(origin, disp, deformed, patchIds);
        });

        nlohmann::json interfacesJson = nlohmann::json::array();
        for (const auto& cfg : interfaces_) {
            interfacesJson.push_back({
                {"patch_a", cfg.patch_a},
                {"side_a", cfg.side_a},
                {"patch_b", cfg.patch_b},
                {"side_b", cfg.side_b}
            });
        }

        return {
            {"net_Epoch", epoch},
            {"net_TotalLoss", totalLoss.template item<double>()},
            {"net_PDELoss", lossPDE.template item<double>()},
            {"net_BCLoss", lossBC.template item<double>()},
            {"net_InterfaceLoss", lossINTER.template item<double>()},
            {"net_PDERawMSE", lastPDERawLoss_},
            {"net_BCRawMSE", lastBCRawLoss_},
            {"net_InterfaceRawMSE", lastInterfaceRawLoss_},
            {"net_LossWeights", {
                {"pde", kPDEWeight},
                {"boundary", kBoundaryConditionWeight},
                {"interface", kInterfaceTractionWeight}
            }},
            {"net_Optimizer", optimizerName_},
            {"net_LearningRate", learningRate_},
            {"net_OriginCtrlPts", origin},
            {"net_Displacements", disp},
            {"net_CtrlPts", deformed},
            {"net_PatchIds", patchIds},
            {"net_Interfaces", interfacesJson},
            {"net_StrongCouplingGroups", strongCouplingGroups_.size()},
            {"net_StrongDirichlet", true},
            {"net_Device", deviceName_},
            {"net_Precision", "float32"},
            {"net_GeometryDerivativeCache", true},
            {"net_OutputDerivativeBasisCache", true},
            {"net_LossPrintClosureStride", kLossPrintClosureStride},
            {"net_CollocationStrides", {
                {"interior", kInteriorCollocationStride},
                {"boundary", kBoundaryCollocationStride},
                {"interface", kInterfaceCollocationStride}
            }},
            {"net_Degree", 2}
        };
    }

    void write_result(int64_t epoch, const torch::Tensor& totalLoss,
                      const torch::Tensor& lossPDE, const torch::Tensor& lossBC,
                      const torch::Tensor& lossINTER, bool appendSnapshot) {
        update_result_json(jsonPath_,
                           make_result_json(epoch, totalLoss, lossPDE, lossBC, lossINTER),
                           appendSnapshot);
    }

    void PostProc() {
        auto total = torch::tensor(lastTotalLoss_);
        auto pde = torch::tensor(lastPDELoss_);
        auto bc = torch::tensor(lastBCLoss_);
        auto inter = torch::tensor(lastInterfaceLoss_);
        write_result(lastExportEpoch_, total, pde, bc, inter, false);
    }
};

template <int Degree, typename Optimizer>
int run_bone_case_with_optimizer(const std::vector<ItdPatch>& itdPatches,
                                 const std::vector<PatchConfig>& patchConfigs,
                                 const std::vector<PatchInterfaceConfig>& interfaces,
                                 double lambda, double mu, std::array<double, 3> bodyForce,
                                 int maxEpoch, double minLoss,
                                 const std::string& jsonPath,
                                 const std::string& optimizerName,
                                 double learningRate) {
    using real_t = float;
    using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 3, Degree, Degree, Degree>>;
    using variable_t = iganet::S<iganet::UniformBSpline<real_t, 3, Degree, Degree, Degree>>;
    using net_t = bone_linear_elasticity<Optimizer, geometry_t, variable_t, kNumBonePatches>;

    const auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                    : torch::Device(torch::kCPU);
    const auto igaOptions = iganet::Options<real_t>{}.device(device);
    std::cout << "Using IGANet tensor device: " << device << "\n"
              << "Using spline precision: float32\n"
              << "Using optimizer: " << optimizerName
              << " (learning_rate=" << learningRate << ")\n"
              << "Collocation strides: interior=" << kInteriorCollocationStride
              << ", boundary=" << kBoundaryCollocationStride
              << ", interface=" << kInterfaceCollocationStride << "\n"
              << "PDE caches: geometry derivatives + output derivative basis\n"
              << "Loss print closure stride: " << kLossPrintClosureStride << "\n";

    net_t net(lambda, mu, bodyForce, patchConfigs, interfaces, maxEpoch, minLoss, jsonPath,
              optimizerName, learningRate,
              {48, 48, 48},
              {{iganet::activation::sigmoid}, {iganet::activation::sigmoid},
               {iganet::activation::sigmoid}, {iganet::activation::none}},
              make_ncoeffs_tuple<kNumBonePatches>(patchConfigs),
              make_ncoeffs_tuple<kNumBonePatches>(patchConfigs),
              iganet::init::greville,
              iganet::IgANetOptions{},
              igaOptions);

    net.load_geometry(itdPatches);
    net.initialize_problem_data();
    net.options().max_epoch(maxEpoch);
    net.options().min_loss(minLoss);
    net.optimizerOptions().lr(learningRate);

    const auto t1 = std::chrono::high_resolution_clock::now();
    net.train();
    const auto t2 = std::chrono::high_resolution_clock::now();

    iganet::Log(iganet::log::info)
        << "Training took "
        << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
        << " seconds\n";

    net.PostProc();
    return 0;
}

template <int Degree>
int run_bone_case(const std::vector<ItdPatch>& itdPatches,
                  const std::vector<PatchConfig>& patchConfigs,
                  const std::vector<PatchInterfaceConfig>& interfaces,
                  double lambda, double mu, std::array<double, 3> bodyForce,
                  int maxEpoch, double minLoss, const std::string& jsonPath,
                  const std::string& optimizerName, double learningRate) {
    if (optimizerName == "lbfgs") {
        return run_bone_case_with_optimizer<Degree, torch::optim::LBFGS>(
            itdPatches, patchConfigs, interfaces, lambda, mu, bodyForce, maxEpoch,
            minLoss, jsonPath, optimizerName, learningRate);
    }

    if (optimizerName == "adamw") {
        return run_bone_case_with_optimizer<Degree, torch::optim::AdamW>(
            itdPatches, patchConfigs, interfaces, lambda, mu, bodyForce, maxEpoch,
            minLoss, jsonPath, optimizerName, learningRate);
    }

    std::cerr << "Unsupported optimizer '" << optimizerName
              << "'. Use 'adamw' or 'lbfgs'.\n";
    return 1;
}

int main() {
    iganet::init();
    iganet::verbose(std::cout);

    std::filesystem::path repoRoot;
    try {
        repoRoot = repo_root_from_build_exe();
    } catch (const std::exception& e) {
        std::cerr << "Could not determine repo root: " << e.what() << "\n";
        return 1;
    }

    const auto configPath = repoRoot / "sim_config.json";
    const auto resultPath = repoRoot / "result_bone_3D.json";

    nlohmann::json config = nlohmann::json::object();
    if (std::ifstream configFile(configPath); configFile.is_open()) {
        try {
            configFile >> config;
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse config JSON: " << e.what() << "\n";
            return 1;
        }
    }

    const auto geometryPath = config.contains("geometry") && config["geometry"].contains("file")
        ? std::filesystem::path(config["geometry"]["file"].get<std::string>())
        : std::filesystem::path(
              "/usr2/obermair/Documents/02_Forschung/Betreuung/MA_Doboczky/bone_simplified.itd");

    double youngModulus = 1.0;
    double poissonRatio = 0.3;
    int maxEpoch = 50;
    double minLoss = 1e-10;
    std::string optimizerName = "adamw";
    double learningRate = 1e-3;
    bool useBodyForce = false;
    std::array<double, 3> bodyForce{0.0, 0.0, 0.0};
    TopDisplacementConfig topDisplacement;

    try {
        if (config.contains("material")) {
            youngModulus = require(config, "material.young_modulus").get<double>();
            poissonRatio = require(config, "material.poisson_ratio").get<double>();
        }
        if (config.contains("simulation")) {
            maxEpoch = require(config, "simulation.max_epoch").get<int>();
            minLoss = require(config, "simulation.min_loss").get<double>();
            if (config["simulation"].contains("optimizer")) {
                optimizerName = config["simulation"]["optimizer"].get<std::string>();
                std::transform(optimizerName.begin(), optimizerName.end(),
                               optimizerName.begin(), [](unsigned char c) {
                                   return static_cast<char>(std::tolower(c));
                               });
            }
            if (config["simulation"].contains("learning_rate")) {
                learningRate = config["simulation"]["learning_rate"].get<double>();
            }
            if (config["simulation"].contains("use_body_force")) {
                useBodyForce = config["simulation"]["use_body_force"].get<bool>();
            }
            if (config["simulation"].contains("use_top_displacement")) {
                topDisplacement.enabled =
                    config["simulation"]["use_top_displacement"].get<bool>();
            }
            if (config["simulation"].contains("top_displacement")) {
                topDisplacement.value =
                    config["simulation"]["top_displacement"].get<double>();
            }
            if (config["simulation"].contains("top_displacement_patch")) {
                topDisplacement.patch =
                    config["simulation"]["top_displacement_patch"].get<int>();
            }
            if (config["simulation"].contains("top_displacement_side")) {
                topDisplacement.side =
                    config["simulation"]["top_displacement_side"].get<int>();
            }
        }
        if (useBodyForce && config.contains("body_force")) {
            const auto& bf = require(config, "body_force");
            if (bf.size() == 3) {
                bodyForce = {
                    bf.at(0).get<double>(),
                    bf.at(1).get<double>(),
                    bf.at(2).get<double>()
                };
            } else {
                std::cerr << "Warning: ignoring non-3D body_force from sim_config.json; "
                          << "using default upward force [0, 0, 50.0].\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    std::ofstream reset(resultPath);
    if (!reset) {
        std::cerr << "Could not reset result file: " << resultPath << "\n";
        return 1;
    }
    reset << "{}\n";
    reset.close();

    std::vector<ItdPatch> itdPatches;
    try {
        itdPatches = load_itd_patches(geometryPath);
        elevate_to_quadratic_tensor_product(itdPatches);
    } catch (const std::exception& e) {
        std::cerr << "Geometry load error: " << e.what() << "\n";
        return 1;
    }

    if (static_cast<int>(itdPatches.size()) != kNumBonePatches) {
        std::cerr << "This bone executable currently instantiates "
                  << kNumBonePatches << " patches, but the ITD file contains "
                  << itdPatches.size() << ".\n";
        return 1;
    }

    const auto degree = itdPatches.front().degrees;
    for (const auto& patch : itdPatches) {
        if (patch.degrees != degree) {
            std::cerr << "All ITD patches must currently share the same spline degree.\n";
            return 1;
        }
    }

    const auto interfaces = discover_interfaces(itdPatches);
    const auto zRange = global_z_range(itdPatches);
    const double bottomTolerance = std::max(1e-8, 1e-6 * std::abs(zRange[1] - zRange[0]));
    const auto patchConfigs = make_patch_configs(
        itdPatches, interfaces, bottomTolerance, topDisplacement);

    std::cout << "Loaded " << itdPatches.size() << " ITD patches from "
              << geometryPath << "\n"
              << "Detected " << interfaces.size() << " patch interfaces. "
              << "Bottom tolerance: " << bottomTolerance << "\n";

    nlohmann::json patchBcJson = nlohmann::json::array();
    nlohmann::json patchMetaJson = nlohmann::json::array();
    for (const auto& patch : patchConfigs) {
        nlohmann::json diriSides = nlohmann::json::array();
        for (const auto& side : patch.boundary_conditions.diri_sides) {
            diriSides.push_back({
                {"side", side.side},
                {"value", {side.x, side.y, side.z}}
            });
        }
        patchBcJson.push_back({
            {"id", patch.id},
            {"diri_sides", patch.boundary_conditions.diri_sides.size()},
            {"diri_side_values", diriSides},
            {"tfbc_sides", patch.boundary_conditions.tfbc_sides.size()}
        });
    }
    for (const auto& patch : itdPatches) {
        patchMetaJson.push_back({
            {"id", patch.id},
            {"ncoeffs", {patch.ncoeffs[0], patch.ncoeffs[1], patch.ncoeffs[2]}},
            {"degrees", {patch.degrees[0], patch.degrees[1], patch.degrees[2]}},
            {"knot_vectors", {patch.knots[0], patch.knots[1], patch.knots[2]}}
        });
    }
    append_json_key(resultPath.string(), "geometry_file", geometryPath.string());
    append_json_key(resultPath.string(), "patch_boundary_summary", patchBcJson);
    append_json_key(resultPath.string(), "net_PatchMeta", patchMetaJson);
    append_json_key(resultPath.string(), "load_case", {
        {"use_body_force", useBodyForce},
        {"body_force", {bodyForce[0], bodyForce[1], bodyForce[2]}},
        {"use_top_displacement", topDisplacement.enabled},
        {"top_displacement", topDisplacement.value}
    });

    const double lambda = (youngModulus * poissonRatio) /
                          ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));
    const double mu = youngModulus / (2.0 * (1.0 + poissonRatio));

    if (degree[0] != degree[1] || degree[0] != degree[2]) {
        std::cerr << "Only equal degrees in all parametric directions are supported currently.\n";
        return 1;
    }

    if (degree[0] != 2) {
        std::cerr << "The current bone executable is instantiated for quadratic "
                  << "ITD splines only, but the file has degree " << degree[0] << ".\n";
        return 1;
    }

    return run_bone_case<2>(itdPatches, patchConfigs, interfaces, lambda, mu,
                            bodyForce, maxEpoch, minLoss, resultPath.string(),
                            optimizerName, learningRate);
}
