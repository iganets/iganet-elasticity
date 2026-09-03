/*
 * Example: 3D multi-patch elasticity from an XML spline geometry.
 *
 * This example focuses on loading an existing spline geometry, constructing a
 * multi-patch description from it, and then training an IgANet whose output is
 * the global displacement coefficient tensor. It is a useful integration case
 * for complex geometries such as the bone example.
 */

#include "headers/lin_elasticity_utils.hpp"
#include "headers/lin_elasticity_multipatch_net.hpp"

#include <iganet.h>

#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

using iganet_elasticity::utils::config::require;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;

namespace {

// -----------------------------------------------------------------------------
// Helper functions for reading example-specific geometry settings
// -----------------------------------------------------------------------------

std::filesystem::path resolveXmlPath(int argc, char** argv,
                                     const std::filesystem::path& repoRoot,
                                     const nlohmann::json& j) {
    if (argc > 1) {
        return std::filesystem::path(argv[1]);
    }

    if (j.contains("geometry")) {
        const auto& gj = j["geometry"];

        if (gj.contains("multipatch_xml_path")) {
            auto path = std::filesystem::path(gj["multipatch_xml_path"].get<std::string>());
            return path.is_relative() ? repoRoot / path : path;
        }

        if (gj.contains("xml_path")) {
            auto path = std::filesystem::path(gj["xml_path"].get<std::string>());
            return path.is_relative() ? repoRoot / path : path;
        }
    }

    return repoRoot / "filedata" / "bone_simplified.xml";
}

int resolveMultiPatchId(const nlohmann::json& j) {
    if (j.contains("geometry") && j["geometry"].contains("multipatch_id")) {
        return j["geometry"]["multipatch_id"].get<int>();
    }
    return 0;
}

torch::Device resolveComputeDevice(const nlohmann::json& j) {
    const std::string mode =
        j.contains("simulation") && j["simulation"].contains("device")
            ? j["simulation"]["device"].get<std::string>()
            : "auto";
    if (mode == "cpu" || mode == "CPU") {
        return torch::Device(torch::kCPU);
    }
    if (mode == "cuda" || mode == "CUDA" || mode == "gpu" || mode == "GPU") {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error(
                "simulation.device requested CUDA, but CUDA is unavailable");
        }
        return torch::Device(torch::kCUDA);
    }
    if (mode == "auto" || mode == "AUTO") {
        return torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                           : torch::Device(torch::kCPU);
    }
    throw std::runtime_error(
        "Unsupported simulation.device '" + mode +
        "'. Expected 'auto', 'cpu', or 'cuda'.");
}

template <typename MultiPatch>
int64_t countPatchGrevillePoints(const MultiPatch& multipatch,
                                 bool interior) {
    int64_t total = 0;
    for (const auto& xi : multipatch.patch_greville(interior)) {
        total += xi[0].numel();
    }
    return total;
}

template <typename BoundaryPointSets>
int64_t countBoundaryPoints(const BoundaryPointSets& pointSets) {
    return std::accumulate(pointSets.begin(), pointSets.end(), int64_t{0},
        [](int64_t sum, const auto& entry) {
            return sum + entry.second[0].numel();
        });
}

template <typename MultiPatch>
void moveMultipatchToDevice(MultiPatch& geometry, torch::Device device) {
    using real_t = typename MultiPatch::value_type;
    using patch_t = typename MultiPatch::patch_type;

    for (std::size_t p = 0; p < geometry.npatches(); ++p) {
        const auto& patch = geometry.patch(p);
        const std::array<iganet::short_t, 3> degrees{
            patch.degree(0), patch.degree(1), patch.degree(2)};
        const std::array<std::vector<real_t>, 3> knots{
            iganet::detail::tensor_to_vector<real_t>(patch.knots(0)),
            iganet::detail::tensor_to_vector<real_t>(patch.knots(1)),
            iganet::detail::tensor_to_vector<real_t>(patch.knots(2))};

        patch_t movedPatch(degrees, knots, iganet::init::zeros,
                           iganet::Options<real_t>{}.device(device));
        movedPatch.from_tensor(patch.as_tensor().to(device));
        geometry.patches()[p] = std::make_shared<patch_t>(std::move(movedPatch));
    }
    geometry.build_dof_map();
}

using MultipatchConfig = MultipatchElasticityConfig<double>;

MultipatchConfig loadMultipatchConfig(const nlohmann::json& j) {
    MultipatchConfig cfg;

    if (j.contains("material")) {
        cfg.youngModulus = require(j, "material.young_modulus").get<double>();
        cfg.poissonRatio = require(j, "material.poisson_ratio").get<double>();
    }

    if (j.contains("simulation")) {
        cfg.maxEpoch = require(j, "simulation.max_epoch").get<int>();
        cfg.minLoss = require(j, "simulation.min_loss").get<double>();
    }

    if (j.contains("multipatch")) {
        const auto& mj = j["multipatch"];
        if (mj.contains("learning_rate")) {
            cfg.learningRate = mj["learning_rate"].get<double>();
        }
        if (mj.contains("lbfgs_history_size")) {
            cfg.lbfgsHistorySize = mj["lbfgs_history_size"].get<int>();
        }
    }

    if (j.contains("network") && j["network"].contains("hidden_layers")) {
        cfg.hiddenLayers = j["network"]["hidden_layers"].get<std::vector<int64_t>>();
    }

    if (j.contains("body_force")) {
        const auto& bf = require(j, "body_force");
        cfg.bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};
    }

    if (j.contains("multipatch") && j["multipatch"].contains("body_force")) {
        const auto& bf = j["multipatch"]["body_force"];
        cfg.bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};
    }

    cfg.patchConfigs = iganet_elasticity::utils::config::load_patch_configs_3d(j);

    const nlohmann::json* patchEntries = nullptr;
    if (j.contains("multipatch_3D") && j["multipatch_3D"].contains("patches")) {
        patchEntries = &j["multipatch_3D"]["patches"];
    } else if (j.contains("patches_3d")) {
        patchEntries = &j["patches_3d"];
    }
    if (patchEntries != nullptr) {
        for (std::size_t i = 0; i < cfg.patchConfigs.size(); ++i) {
            if (!patchEntries->at(i).contains("body_force")) {
                cfg.patchConfigs[i].body_force = cfg.bodyForce;
            }
        }
    }

    return cfg;
}

template <typename MultiPatch>
std::size_t resolvePatchIndex(const MultiPatch& geometry, int patchId) {
    for (std::size_t p = 0; p < geometry.npatches(); ++p) {
        if (geometry.patch_xml_id(p) == patchId) {
            return p;
        }
    }
    throw std::runtime_error(
        "Could not resolve XML patch_id " + std::to_string(patchId));
}

template <typename MultiPatch>
bool isOuterBoundary(const MultiPatch& geometry, std::size_t patchIndex, int side) {
    return std::any_of(geometry.boundaries().begin(), geometry.boundaries().end(),
        [&](const auto& boundary) {
            return boundary.patch == patchIndex && boundary.side == side;
        });
}

template <typename MultiPatch>
void resolveAndValidatePatchConfigs(const MultiPatch& geometry, MultipatchConfig& cfg) {
    for (auto& patchCfg : cfg.patchConfigs) {
        const int xmlPatchId = patchCfg.patch_id;
        const auto patchIndex = resolvePatchIndex(geometry, xmlPatchId);
        const auto validateSide = [&](int side) {
            if (!isOuterBoundary(geometry, patchIndex, side)) {
                throw std::runtime_error(
                    "Patch " + std::to_string(xmlPatchId) + ", side " +
                    std::to_string(side) + " is not an outer boundary");
            }
        };
        for (const auto& entry : patchCfg.diri_sides) {
            validateSide(entry.side);
        }
        for (const auto& entry : patchCfg.force_sides) {
            validateSide(entry.side);
        }
        for (const auto side : patchCfg.tfbc_sides) {
            validateSide(side);
        }
        patchCfg.patch_id = static_cast<int>(patchIndex);
    }
}

std::size_t countDirichletBoundarySets(const MultipatchConfig& cfg) {
    return std::accumulate(cfg.patchConfigs.begin(), cfg.patchConfigs.end(), std::size_t{0},
        [](std::size_t count, const auto& patchCfg) {
            return count + patchCfg.diri_sides.size();
        });
}

std::size_t countForceBoundarySets(const MultipatchConfig& cfg) {
    return std::accumulate(cfg.patchConfigs.begin(), cfg.patchConfigs.end(), std::size_t{0},
        [](std::size_t count, const auto& patchCfg) {
            return count + patchCfg.force_sides.size();
        });
}

nlohmann::json tensor3BlocksToJson(const torch::Tensor& tensor) {
    auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    const int64_t n = cpu.numel() / 3;
    nlohmann::json result = nlohmann::json::array();
    for (int64_t i = 0; i < n; ++i) {
        result.push_back({
            cpu.index({i}).item<double>(),
            cpu.index({i + n}).item<double>(),
            cpu.index({i + 2 * n}).item<double>()});
    }
    return result;
}

template <typename Patch>
nlohmann::json vectorTensorToJson(const torch::Tensor& tensor) {
    auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    nlohmann::json result = nlohmann::json::array();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        result.push_back(cpu.index({i}).item<double>());
    }
    return result;
}

template <typename MultiPatch>
nlohmann::json patchesToJson(const MultiPatch& geometry,
                             const MultiPatch& displacement,
                             const torch::Tensor& displacementTensor) {
    nlohmann::json patches = nlohmann::json::array();

    for (std::size_t patchIndex = 0; patchIndex < geometry.npatches(); ++patchIndex) {
        const auto& patch = geometry.patch(patchIndex);
        auto localGeometry = patch.as_tensor();
        auto localDisplacement = displacement.local_tensor(patchIndex, displacementTensor);

        nlohmann::json degrees = nlohmann::json::array();
        nlohmann::json knotVectors = nlohmann::json::array();
        for (iganet::short_t d = 0; d < MultiPatch::parDim(); ++d) {
            degrees.push_back(patch.degree(d));
            knotVectors.push_back(vectorTensorToJson<typename MultiPatch::patch_type>(patch.knots(d)));
        }

        nlohmann::json entry;
        entry["index"] = patchIndex;
        entry["xml_id"] = geometry.patch_xml_id(patchIndex);
        entry["degrees"] = degrees;
        entry["knot_vectors"] = knotVectors;
        entry["control_points"] = tensor3BlocksToJson(localGeometry);
        entry["displacements"] = tensor3BlocksToJson(localDisplacement);
        entry["deformed_control_points"] =
            tensor3BlocksToJson(localGeometry + localDisplacement);
        patches.push_back(entry);
    }

    return patches;
}

void finalizeIganet() {
    if (torch::cuda::is_available()) {
        iganet::finalize();
    } else {
        std::cout << "[INFO] Succeeded\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    using real_t = double;
    using geometry_patch_t = iganet::DynamicBSplinePatch<real_t, 3, 3>;
    using multipatch_t = iganet::MultiPatch<geometry_patch_t>;

    iganet::init();
    iganet::verbose(std::cout);

    std::filesystem::path repoRoot;
    try {
        repoRoot = repo_root_from_build_exe();
    } catch (const std::exception& e) {
        std::cerr << "Could not determine repo root: " << e.what() << "\n";
        return 1;
    }

    const auto configPath = repoRoot / "src" / "examples3D" / "multiPatch" /
                            "sim_config_3D_multi_patch_bone.json";
    const auto resultPath =
        repoRoot / "results" / "result_iganet_lin_elasticity_3D_multipatch_bone.json";

    nlohmann::json j;
    try {
        std::ifstream cfgFile(configPath);
        if (!cfgFile) {
            throw std::runtime_error("Could not open config file: " + configPath.string());
        }
        cfgFile >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse config JSON: " << e.what() << "\n";
        return 1;
    }

    const auto xmlPath = resolveXmlPath(argc, argv, repoRoot, j);
    const int multipatchId = resolveMultiPatchId(j);
    auto cfg = loadMultipatchConfig(j);

    const torch::Device computeDevice = resolveComputeDevice(j);
    const auto options = iganet::Options<real_t>{}.device(computeDevice);

    try {
        auto start = std::chrono::high_resolution_clock::now();

        multipatch_t geometry;
        geometry.set_matching_tolerance(1e-6, 1e-6);
        geometry.from_xml(xmlPath.string(), multipatchId);
        moveMultipatchToDevice(geometry, computeDevice);
        resolveAndValidatePatchConfigs(geometry, cfg);
        if (cfg.patchConfigs.empty()) {
            throw std::runtime_error("No patches_3d boundary conditions configured");
        }

        auto displacement = geometry.make_isoparametric_solution_space<3>(options);
        iganet::StrongDirichletConstraints<real_t> constraints(displacement);
        for (const auto& patchCfg : cfg.patchConfigs) {
            const auto patchIndex = static_cast<std::size_t>(patchCfg.patch_id);
            for (const auto& entry : patchCfg.diri_sides) {
                constraints
                    .fix_boundary(displacement, patchIndex, entry.side, 0, entry.x)
                    .fix_boundary(displacement, patchIndex, entry.side, 1, entry.y)
                    .fix_boundary(displacement, patchIndex, entry.side, 2, entry.z);
            }
        }
        if (constraints.nfixed() == 0) {
            throw std::runtime_error("Configured diri_sides fixed no displacement DOFs");
        }

        const int64_t allGreville = countPatchGrevillePoints(geometry, false);
        const int64_t interiorGreville = countPatchGrevillePoints(geometry, true);
        const auto boundaryPointSets = geometry.boundary_greville();
        const int64_t boundaryGreville = countBoundaryPoints(boundaryPointSets);

        using optimizer_t = torch::optim::LBFGS;
        using net_t =
            iganet_elasticity::multipatch::linear_elasticity<optimizer_t, multipatch_t>;

        iganet::IgANetOptions netDefaults;
        netDefaults.max_epoch(cfg.maxEpoch);
        netDefaults.min_loss(cfg.minLoss);
        netDefaults.min_loss_change(0.0);
        netDefaults.min_loss_rel_change(0.0);

        std::vector<std::vector<std::any>> activations(
            cfg.hiddenLayers.size(), {iganet::activation::sigmoid});
        activations.push_back({iganet::activation::none});

        net_t net(geometry, displacement, constraints, cfg, cfg.hiddenLayers,
                  activations, netDefaults, options);
        auto lbfgsOptions = torch::optim::LBFGSOptions(cfg.learningRate);
        lbfgsOptions.history_size(cfg.lbfgsHistorySize);
        lbfgsOptions.tolerance_grad(1e-12);
        lbfgsOptions.tolerance_change(1e-12);
        net.optimizerOptionsReset(lbfgsOptions);

        net.train();
        net.eval();

        const auto& geometryOut = net.geometry();
        const auto& displacementOut = net.displacement();
        const auto geometryTensor = geometryOut.as_tensor();
        const auto displacementTensor = displacementOut.as_tensor().detach();
        const auto& lossHistory = net.history();

        if (!lossHistory.empty() && !std::isfinite(lossHistory.back())) {
            throw std::runtime_error("IgANet training produced a non-finite loss");
        }

        auto stop = std::chrono::high_resolution_clock::now();
        const double elapsed =
            std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

        nlohmann::json summary;
        summary["xml_path"] = xmlPath.string();
        summary["multipatch_id"] = multipatchId;
        summary["example"] = "multipatch_xml_iganet_3d";
        summary["device"] = computeDevice.str();
        summary["npatches"] = geometryOut.npatches();
        summary["ninterfaces"] = geometryOut.ninterfaces();
        summary["nboundaries"] = geometryOut.nboundaries();
        summary["geometry_scalar_dofs"] = geometryOut.ndofs();
        summary["displacement_scalar_dofs"] = displacementOut.ndofs();
        summary["geometry_tensor_size"] = geometryTensor.numel();
        summary["displacement_tensor_size"] = displacementTensor.numel();
        summary["patch_greville_points"] = allGreville;
        summary["patch_interior_greville_points"] = interiorGreville;
        summary["boundary_point_sets"] = boundaryPointSets.size();
        summary["boundary_greville_points"] = boundaryGreville;
        summary["dirichlet_boundary_sets"] = countDirichletBoundarySets(cfg);
        summary["force_boundary_sets"] = countForceBoundarySets(cfg);
        summary["max_epoch"] = cfg.maxEpoch;
        summary["learning_rate"] = cfg.learningRate;
        summary["strong_dirichlet_fixed_dofs"] = constraints.nfixed();
        summary["strong_dirichlet_free_dofs"] = constraints.nfree();
        summary["loss_initial"] = lossHistory.empty() ? 0.0 : lossHistory.front();
        summary["loss_final"] = lossHistory.empty() ? 0.0 : lossHistory.back();
        summary["loss_history"] = lossHistory;
        summary["geometry_control_points"] = tensor3BlocksToJson(geometryTensor);
        summary["displacements"] = tensor3BlocksToJson(displacementTensor);
        summary["deformed_control_points"] =
            tensor3BlocksToJson(geometryTensor + displacementTensor.detach());
        summary["patches"] = patchesToJson(geometryOut, displacementOut, displacementTensor);
        summary["elapsed_seconds"] = elapsed;
        summary["note"] =
            "IgANet maps the fixed XML geometry coefficients to global displacement "
            "coefficients; Dirichlet values are imposed strongly.";

        {
            std::ofstream out(resultPath);
            if (!out) {
                throw std::runtime_error("Could not open result file: " + resultPath.string());
            }
            nlohmann::json result;
            result["multipatch_elasticity"] = summary;
            out << result.dump(1);
        }

        std::cout << "\n=== XML MULTIPATCH IGANET TRAINING ===\n"
                  << "XML: " << xmlPath << "\n"
                  << "config: " << configPath << "\n"
                  << "device: " << computeDevice.str() << "\n"
                  << "patches: " << geometryOut.npatches() << "\n"
                  << "interfaces: " << geometryOut.ninterfaces() << "\n"
                  << "outer boundaries: " << geometryOut.nboundaries() << "\n"
                  << "geometry scalar DOFs: " << geometryOut.ndofs() << "\n"
                  << "displacement scalar DOFs: " << displacementOut.ndofs() << "\n"
                  << "fixed DOFs: " << constraints.nfixed() << "\n"
                  << "patch Greville points: " << allGreville << "\n"
                  << "interior Greville points: " << interiorGreville << "\n"
                  << "boundary Greville points: " << boundaryGreville << "\n"
                  << "loss initial: " << summary["loss_initial"] << "\n"
                  << "loss final: " << summary["loss_final"] << "\n"
                  << "result: " << resultPath << "\n"
                  << "==================================\n";
    } catch (const std::exception& e) {
        std::cerr << "XML MultiPatch IgANet example failed: " << e.what() << "\n";
        finalizeIganet();
        return 1;
    }

    finalizeIganet();
    return 0;
}
