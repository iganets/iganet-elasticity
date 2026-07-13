#include "headers/lin_elasticity_utils.hpp"
#include "headers/lin_elasticity_multipatch_net.hpp"

#include <iganet.h>

#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <any>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using iganet_elasticity::utils::config::require;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using patch_config_3d_t = iganet_elasticity::utils::config::patch_config_3d;

namespace {

using real_t = double;
using patch_t = iganet::DynamicBSplinePatch<real_t, 3, 3>;
using multipatch_t = iganet::MultiPatch<patch_t>;
using interface_t = iganet::MultiPatchInterface<3>;
using config_t = MultipatchElasticityConfig<real_t>;
using boundary_value_t = typename config_t::boundary_value_t;

enum class ComputeDeviceMode { Auto, CPU, CUDA };

std::vector<boundary_value_t> readBoundaryValues(const nlohmann::json& values) {
    std::vector<boundary_value_t> result;
    for (const auto& value : values) {
        result.emplace_back(value.at(0).get<int>(),
                            value.at(1).get<double>(),
                            value.at(2).get<double>(),
                            value.at(3).get<double>());
    }
    return result;
}

std::filesystem::path resolveXmlPath(const std::filesystem::path& repoRoot,
                                     const nlohmann::json& j) {
    if (!j.contains("geometry")) {
        throw std::runtime_error("geometry section missing for xml mode");
    }

    const auto& gj = j["geometry"];
    if (gj.contains("multipatch_xml_path")) {
        auto path = std::filesystem::path(gj["multipatch_xml_path"].get<std::string>());
        return path.is_relative() ? repoRoot / path : path;
    }
    if (gj.contains("xml_path")) {
        auto path = std::filesystem::path(gj["xml_path"].get<std::string>());
        return path.is_relative() ? repoRoot / path : path;
    }

    throw std::runtime_error("geometry.multipatch_xml_path or geometry.xml_path missing");
}

int resolveMultiPatchId(const nlohmann::json& j) {
    if (j.contains("geometry") && j["geometry"].contains("multipatch_id")) {
        return j["geometry"]["multipatch_id"].get<int>();
    }
    return 0;
}

ComputeDeviceMode parseComputeDeviceMode(const nlohmann::json& j) {
    if (!j.contains("simulation") || !j["simulation"].contains("device")) {
        return ComputeDeviceMode::Auto;
    }

    const auto mode = j["simulation"]["device"].get<std::string>();
    if (mode == "auto" || mode == "AUTO") {
        return ComputeDeviceMode::Auto;
    }
    if (mode == "cpu" || mode == "CPU") {
        return ComputeDeviceMode::CPU;
    }
    if (mode == "cuda" || mode == "CUDA" || mode == "gpu" || mode == "GPU") {
        return ComputeDeviceMode::CUDA;
    }

    throw std::runtime_error(
        "Unsupported simulation.device '" + mode +
        "'. Expected 'auto', 'cpu', or 'cuda'.");
}

torch::Device resolveComputeDevice(const nlohmann::json& j) {
    switch (parseComputeDeviceMode(j)) {
        case ComputeDeviceMode::CPU:
            return torch::Device(torch::kCPU);
        case ComputeDeviceMode::CUDA:
            if (!torch::cuda::is_available()) {
                throw std::runtime_error(
                    "simulation.device requested CUDA, but torch::cuda::is_available() is false");
            }
            return torch::Device(torch::kCUDA);
        case ComputeDeviceMode::Auto:
            return torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                               : torch::Device(torch::kCPU);
    }

    throw std::runtime_error("Unhandled compute device mode");
}

config_t loadConfig(const nlohmann::json& j) {
    config_t cfg;

    if (j.contains("material")) {
        cfg.youngModulus = require(j, "material.young_modulus").get<double>();
        cfg.poissonRatio = require(j, "material.poisson_ratio").get<double>();
    }

    if (j.contains("simulation")) {
        cfg.maxEpoch = require(j, "simulation.max_epoch").get<int>();
    }

    const auto geometrySplineCfg =
        iganet_elasticity::utils::config::load_geometry_spline_config(j);
    const auto solutionSplineCfg =
        iganet_elasticity::utils::config::load_solution_spline_config(j);
    cfg.degree = solutionSplineCfg.degree;
    cfg.ncoeffs = solutionSplineCfg.nr_ctrl_pts;

    if (geometrySplineCfg.degree != solutionSplineCfg.degree ||
        geometrySplineCfg.nr_ctrl_pts != solutionSplineCfg.nr_ctrl_pts) {
        throw std::runtime_error(
            "3D multipatch example currently assumes isoparametric spaces: "
            "geometry_spline and solution_spline must match");
    }

    if (j.contains("body_force")) {
        const auto& bf = require(j, "body_force");
        cfg.bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};
    }

    if (j.contains("boundary_conditions")) {
        cfg.diriSides = readBoundaryValues(require(j, "boundary_conditions.diri_sides"));
        cfg.forceSides = readBoundaryValues(require(j, "boundary_conditions.force_sides"));
        cfg.tfbcSides = require(j, "boundary_conditions.tfbc_sides").get<std::vector<int>>();
    }

    if (j.contains("multipatch")) {
        const auto& mj = j["multipatch"];
        if (mj.contains("learning_rate")) {
            cfg.learningRate = mj["learning_rate"].get<double>();
        }
        if (mj.contains("body_force")) {
            const auto& bf = mj["body_force"];
            cfg.bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};
        }
        if (mj.contains("degree")) {
            cfg.degree = mj["degree"].get<int>();
        }
        if (mj.contains("nr_ctrl_pts")) {
            cfg.ncoeffs = mj["nr_ctrl_pts"].get<int>();
        }
        if (mj.contains("lbfgs_history_size")) {
            cfg.lbfgsHistorySize = mj["lbfgs_history_size"].get<int>();
        }
        if (mj.contains("boundary_conditions")) {
            const auto& bc = mj["boundary_conditions"];
            if (bc.contains("diri_sides")) {
                cfg.diriSides = readBoundaryValues(bc["diri_sides"]);
            }
            if (bc.contains("force_sides")) {
                cfg.forceSides = readBoundaryValues(bc["force_sides"]);
            }
            if (bc.contains("tfbc_sides")) {
                cfg.tfbcSides = bc["tfbc_sides"].get<std::vector<int>>();
            }
        }
    }

    cfg.patchConfigs = iganet_elasticity::utils::config::load_patch_configs_3d(j);

    if (cfg.degree < 1) {
        throw std::runtime_error("Multipatch degree must be >= 1");
    }
    if (cfg.ncoeffs <= cfg.degree) {
        throw std::runtime_error("Multipatch nr_ctrl_pts must be larger than degree");
    }
    if (cfg.maxEpoch <= 0) {
        throw std::runtime_error("simulation.max_epoch must be positive");
    }

    return cfg;
}

std::string sideLabel(int side) {
    return "side_" + std::to_string(side);
}

std::array<std::vector<real_t>, 3> openUniformKnots(int ncoeffs, int degree) {
    const auto kv = makeUniformKnotVector(ncoeffs, degree);
    return {kv, kv, kv};
}

patch_t makeCubePatch(real_t x0, real_t x1, real_t y0, real_t y1,
                      real_t z0, real_t z1, int ncoeffs, int degree,
                      const iganet::Options<real_t>& options) {
    const std::array<iganet::short_t, 3> degrees{
        static_cast<iganet::short_t>(degree),
        static_cast<iganet::short_t>(degree),
        static_cast<iganet::short_t>(degree)};
    patch_t patch(degrees, openUniformKnots(ncoeffs, degree),
                  iganet::init::zeros, options);

    auto xi = patch.greville(false);
    auto x = x0 + (x1 - x0) * xi[0];
    auto y = y0 + (y1 - y0) * xi[1];
    auto z = z0 + (z1 - z0) * xi[2];
    patch.from_tensor(torch::cat({x, y, z}));
    return patch;
}

interface_t interface(std::size_t p1, iganet::short_t s1,
                      std::size_t p2, iganet::short_t s2) {
    interface_t result;
    result.patch1 = p1;
    result.side1 = s1;
    result.patch2 = p2;
    result.side2 = s2;
    result.direction_map = {0, 1, 2};
    result.direction_orientation = {true, true, true};
    return result;
}

multipatch_t makeTwoCubeGeometry(const config_t& cfg,
                                 const iganet::Options<real_t>& options) {
    multipatch_t geometry;
    geometry.set_matching_tolerance(1e-6, 1e-6);

    geometry.addPatch(makeCubePatch(0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                    cfg.ncoeffs, cfg.degree, options), 0);
    geometry.addPatch(makeCubePatch(1.0, 2.0, 0.0, 1.0, 0.0, 1.0,
                                    cfg.ncoeffs, cfg.degree, options), 1);
    geometry.addInterface(interface(0, iganet::side::east, 1, iganet::side::west));

    geometry.addBoundary({0, iganet::side::west, sideLabel(iganet::side::west)});
    geometry.addBoundary({1, iganet::side::east, sideLabel(iganet::side::east)});
    for (std::size_t p = 0; p < 2; ++p) {
        geometry.addBoundary({p, iganet::side::south, sideLabel(iganet::side::south)});
        geometry.addBoundary({p, iganet::side::north, sideLabel(iganet::side::north)});
        geometry.addBoundary({p, iganet::side::front, sideLabel(iganet::side::front)});
        geometry.addBoundary({p, iganet::side::back, sideLabel(iganet::side::back)});
    }

    geometry.build_dof_map();
    return geometry;
}

multipatch_t makeXmlGeometry(const std::filesystem::path& xmlPath,
                             int multipatchId,
                             const iganet::Options<real_t>&) {
    multipatch_t geometry;
    geometry.set_matching_tolerance(1e-6, 1e-6);
    geometry.from_xml(xmlPath.string(), multipatchId);
    geometry.build_dof_map();
    return geometry;
}

void moveMultipatchToDevice(multipatch_t& geometry, torch::Device device) {
    for (std::size_t p = 0; p < geometry.npatches(); ++p) {
        const auto& patch = geometry.patch(p);
        const std::array<iganet::short_t, 3> degrees{
            patch.degree(0), patch.degree(1), patch.degree(2)};
        const std::array<std::vector<real_t>, 3> knots{
            iganet::detail::tensor_to_vector<real_t>(patch.knots(0)),
            iganet::detail::tensor_to_vector<real_t>(patch.knots(1)),
            iganet::detail::tensor_to_vector<real_t>(patch.knots(2))};

        patch_t movedPatch(
            degrees,
            knots,
            iganet::init::zeros,
            iganet::Options<real_t>{}.device(device));
        movedPatch.from_tensor(patch.as_tensor().to(device));
        geometry.patches()[p] = std::make_shared<patch_t>(std::move(movedPatch));
    }
    geometry.build_dof_map();
}

std::size_t resolvePatchIndex(const multipatch_t& geometry, int patchId) {
    for (std::size_t p = 0; p < geometry.npatches(); ++p) {
        const int xmlId = geometry.patch_xml_id(p);
        if (xmlId == patchId) {
            return p;
        }
    }

    if (patchId >= 0 && static_cast<std::size_t>(patchId) < geometry.npatches()) {
        return static_cast<std::size_t>(patchId);
    }

    throw std::runtime_error(
        "Could not resolve patch_id " + std::to_string(patchId) +
        " to a patch in the loaded 3D multipatch geometry");
}

void resolvePatchConfigs(const multipatch_t& geometry, config_t& cfg) {
    for (auto& patchCfg : cfg.patchConfigs) {
        patchCfg.patch_id = static_cast<int>(resolvePatchIndex(geometry, patchCfg.patch_id));
    }
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

nlohmann::json tensorToJson(const torch::Tensor& tensor) {
    auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    nlohmann::json result = nlohmann::json::array();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        result.push_back(cpu.index({i}).item<double>());
    }
    return result;
}

nlohmann::json patchesToJson(const multipatch_t& geometry,
                             const multipatch_t& displacement,
                             const torch::Tensor& displacementTensor) {
    nlohmann::json patches = nlohmann::json::array();
    for (std::size_t p = 0; p < geometry.npatches(); ++p) {
        const auto& patch = geometry.patch(p);
        const auto localGeometry = patch.as_tensor();
        const auto localDisplacement = displacement.local_tensor(p, displacementTensor);

        nlohmann::json degrees = nlohmann::json::array();
        nlohmann::json knotVectors = nlohmann::json::array();
        for (iganet::short_t d = 0; d < multipatch_t::parDim(); ++d) {
            degrees.push_back(patch.degree(d));
            knotVectors.push_back(tensorToJson(patch.knots(d)));
        }

        nlohmann::json entry;
        entry["index"] = p;
        entry["xml_id"] = geometry.patch_xml_id(p);
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

void writeJsonFile(const std::filesystem::path& path, const nlohmann::json& data) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open JSON output file: " + path.string());
    }
    out << data.dump(1);
}

} // namespace

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

    const auto configPath = repoRoot / "src" / "examples3D" / "multiPatch" /
                            "sim_config_3D_multi_patch.json";
    const auto resultPath =
        repoRoot / "results" / "result_iganet_lin_elasticity_3D_multipatch_parametrized.json";
    nlohmann::json j;
    {
        std::ifstream cfgFile(configPath);
        if (!cfgFile) {
            std::cerr << "Could not open config file: " << configPath << "\n";
            iganet::finalize();
            return 1;
        }
        cfgFile >> j;
    }

    const auto computeDevice = resolveComputeDevice(j);
    const auto options = iganet::Options<real_t>{}.device(computeDevice);
    const auto tensorOptions =
        torch::TensorOptions().dtype(torch::kFloat64).device(computeDevice);

    try {
        auto cfg = loadConfig(j);
        const auto mode = parseGeometryMode(j);
        std::optional<std::filesystem::path> xmlPath;
        multipatch_t geometry;
        if (mode == GeometryMode::Xml) {
            xmlPath = resolveXmlPath(repoRoot, j);
            geometry = makeXmlGeometry(*xmlPath, resolveMultiPatchId(j), options);
        } else {
            geometry = makeTwoCubeGeometry(cfg, options);
        }
        moveMultipatchToDevice(geometry, computeDevice);
        resolvePatchConfigs(geometry, cfg);
        auto displacement = geometry.make_isoparametric_solution_space<3>(options);

        iganet::StrongDirichletConstraints<real_t> constraints(displacement);
        if (!cfg.patchConfigs.empty()) {
            for (const auto& patchCfg : cfg.patchConfigs) {
                const auto patchIndex = static_cast<std::size_t>(patchCfg.patch_id);
                for (const auto& entry : patchCfg.diri_sides) {
                    constraints
                        .fix_boundary(displacement, patchIndex, entry.side, 0, entry.x)
                        .fix_boundary(displacement, patchIndex, entry.side, 1, entry.y)
                        .fix_boundary(displacement, patchIndex, entry.side, 2, entry.z);
                }
            }
        } else {
            for (const auto& entry : cfg.diriSides) {
                const int side = std::get<0>(entry);
                constraints.fix_boundary_label(displacement, sideLabel(side), 0, std::get<1>(entry))
                    .fix_boundary_label(displacement, sideLabel(side), 1, std::get<2>(entry))
                    .fix_boundary_label(displacement, sideLabel(side), 2, std::get<3>(entry));
            }
        }

        using optimizer_t = torch::optim::LBFGS;
        using net_t = iganet_elasticity::multipatch::linear_elasticity<optimizer_t, multipatch_t>;

        iganet::IgANetOptions netDefaults;
        netDefaults.max_epoch(cfg.maxEpoch);
        netDefaults.min_loss(1e-12);
        netDefaults.min_loss_change(0.0);
        netDefaults.min_loss_rel_change(0.0);

        net_t net(
            geometry,
            displacement,
            constraints,
            cfg,
            {25, 25},
            {{iganet::activation::sigmoid},
             {iganet::activation::sigmoid},
             {iganet::activation::none}},
            netDefaults,
            options);

        auto lbfgsOptions = torch::optim::LBFGSOptions(cfg.learningRate);
        lbfgsOptions.history_size(cfg.lbfgsHistorySize);
        lbfgsOptions.tolerance_grad(1e-12);
        lbfgsOptions.tolerance_change(1e-12);
        net.optimizerOptionsReset(lbfgsOptions);

        net.train();
        net.eval();

        const auto& geometryOut = net.geometry();
        const auto& displacementOut = net.displacement();
        const auto displacementTensor = displacementOut.as_tensor().detach();
        const auto geometryTensor = geometryOut.as_tensor();
        const auto& history = net.history();

        nlohmann::json summary;
        summary["example"] = "multipatch_parametric_3d";
        summary["geometry_mode"] = (mode == GeometryMode::Xml ? "xml" : "parametric");
        summary["device"] = computeDevice.str();
        if (xmlPath.has_value()) {
            summary["xml_path"] = xmlPath->string();
        }
        summary["npatches"] = geometryOut.npatches();
        summary["ninterfaces"] = geometryOut.ninterfaces();
        summary["nboundaries"] = geometryOut.nboundaries();
        summary["geometry_scalar_dofs"] = geometryOut.ndofs();
        summary["displacement_scalar_dofs"] = displacementOut.ndofs();
        summary["strong_dirichlet_fixed_dofs"] = constraints.nfixed();
        summary["strong_dirichlet_free_dofs"] = constraints.nfree();
        summary["loss_initial"] = history.empty() ? 0.0 : history.front();
        summary["loss_final"] = history.empty() ? 0.0 : history.back();
        summary["loss_history"] = history;
        summary["geometry_control_points"] = tensor3BlocksToJson(geometryTensor);
        summary["displacements"] = tensor3BlocksToJson(displacementTensor);
        summary["deformed_control_points"] =
            tensor3BlocksToJson(geometryTensor + displacementTensor);
        summary["patches"] = patchesToJson(geometryOut, displacementOut, displacementTensor);
        summary["note"] =
            "3D MultiPatch collocation example using core MultiPatch and "
            "StrongDirichletConstraints.";

        std::ofstream out(resultPath);
        nlohmann::json result;
        result["multipatch_elasticity"] = summary;
        out << result.dump(1);

        std::cout << "\n=== PARAMETRIC MULTIPATCH 3D ===\n"
                  << "config: " << configPath << "\n"
                  << "device: " << computeDevice.str() << "\n"
                  << "patches: " << geometryOut.npatches() << "\n"
                  << "interfaces: " << geometryOut.ninterfaces() << "\n"
                  << "scalar dofs: " << geometryOut.ndofs() << "\n"
                  << "fixed dofs: " << constraints.nfixed() << "\n"
                  << "free dofs: " << constraints.nfree() << "\n"
                  << "loss initial: " << summary["loss_initial"] << "\n"
                  << "loss final: " << summary["loss_final"] << "\n"
                  << "result: " << resultPath << "\n"
                  << "======================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Parametric MultiPatch example failed: " << e.what() << "\n";
        iganet::finalize();
        return 1;
    }

    iganet::finalize();
    return 0;
}
