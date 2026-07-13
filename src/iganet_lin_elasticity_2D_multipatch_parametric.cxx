#include "lin_elasticity_utils.hpp"

#include <iganet.h>

#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <any>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using iganet_elasticity::utils::config::require;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using optimizer_config_t = iganet_elasticity::utils::config::optimizer_config;
using optimizer_type_t = iganet_elasticity::utils::config::optimizer_type;
using patch_config_2d_t = iganet_elasticity::utils::config::patch_config_2d;

namespace {

using real_t = double;
using patch_t = iganet::DynamicBSplinePatch<real_t, 2, 2>;
using multipatch_t = iganet::MultiPatch<patch_t>;
using interface_t = iganet::MultiPatchInterface<2>;

struct ParametricConfig {
    double youngModulus{210.0};
    double poissonRatio{0.25};
    int maxEpoch{50};
    int geometryDegree{3};
    int geometryNcoeffs{5};
    int solutionDegree{3};
    int solutionNcoeffs{5};
    optimizer_config_t optimizer;
    std::vector<patch_config_2d_t> patchConfigs;
};

enum class ComputeDeviceMode { Auto, CPU, CUDA };

ParametricConfig loadConfig(const nlohmann::json& j) {
    ParametricConfig cfg;

    if (j.contains("material")) {
        cfg.youngModulus = require(j, "material.young_modulus").get<double>();
        cfg.poissonRatio = require(j, "material.poisson_ratio").get<double>();
    }

    if (j.contains("simulation")) {
        cfg.maxEpoch = require(j, "simulation.max_epoch").get<int>();
    }

    cfg.optimizer = iganet_elasticity::utils::config::load_optimizer_config(j);
    cfg.patchConfigs = iganet_elasticity::utils::config::load_patch_configs_2d(j);

    const auto geometrySplineCfg =
        iganet_elasticity::utils::config::load_geometry_spline_config(j);
    const auto solutionSplineCfg =
        iganet_elasticity::utils::config::load_solution_spline_config(j);
    cfg.geometryDegree = geometrySplineCfg.degree;
    cfg.geometryNcoeffs = geometrySplineCfg.nr_ctrl_pts;
    cfg.solutionDegree = solutionSplineCfg.degree;
    cfg.solutionNcoeffs = solutionSplineCfg.nr_ctrl_pts;

    if (cfg.geometryDegree != cfg.solutionDegree ||
        cfg.geometryNcoeffs != cfg.solutionNcoeffs) {
        throw std::runtime_error(
            "2D multipatch example currently assumes isoparametric spaces: "
            "geometry_spline and solution_spline must match");
    }

    if (cfg.solutionDegree < 1) {
        throw std::runtime_error("2D multipatch degree must be >= 1");
    }
    if (cfg.solutionNcoeffs <= cfg.solutionDegree) {
        throw std::runtime_error("2D multipatch nr_ctrl_pts must be larger than degree");
    }
    if (cfg.maxEpoch <= 0) {
        throw std::runtime_error("simulation.max_epoch must be positive");
    }

    return cfg;
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

std::string sideLabel(int side) {
    return "side_" + std::to_string(side);
}

std::vector<double> makeOpenUniformKnotVector(int ncoeffs, int degree) {
    if (ncoeffs <= degree) {
        throw std::runtime_error("ncoeffs must be larger than degree");
    }

    const int nknots = ncoeffs + degree + 1;
    std::vector<double> kv(nknots);

    for (int i = 0; i <= degree; ++i) {
        kv[i] = 0.0;
        kv[nknots - 1 - i] = 1.0;
    }

    const int nInterior = ncoeffs - degree - 1;
    for (int j = 1; j <= nInterior; ++j) {
        kv[degree + j] = static_cast<double>(j) / static_cast<double>(nInterior + 1);
    }

    return kv;
}

std::vector<double> openUniformKnots(int ncoeffs, int degree) {
    return makeOpenUniformKnotVector(ncoeffs, degree);
}

patch_t makeSquarePatch(real_t x0, real_t x1, real_t y0, real_t y1,
                        int ncoeffs, int degree,
                        const iganet::Options<real_t>& options) {
    const std::array<iganet::short_t, 2> degrees{
        static_cast<iganet::short_t>(degree),
        static_cast<iganet::short_t>(degree)
    };

    const auto kv = openUniformKnots(ncoeffs, degree);
    patch_t patch(degrees, {kv, kv}, iganet::init::zeros, options);

    const auto torchOptions = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(patch.knots(0).device());

    auto xi = patch.greville(false);
    auto x = x0 + (x1 - x0) * xi[0];
    auto y = y0 + (y1 - y0) * xi[1];

    patch.from_tensor(torch::cat({x, y}));

    return patch;
}

interface_t interface(std::size_t p1, iganet::short_t s1,
                      std::size_t p2, iganet::short_t s2) {
    interface_t result;
    result.patch1 = p1;
    result.side1 = s1;
    result.patch2 = p2;
    result.side2 = s2;
    result.direction_map = {0, 1};
    result.direction_orientation = {true, true};
    return result;
}

multipatch_t makeTwoSquareGeometry(const ParametricConfig& cfg,
                                   const iganet::Options<real_t>& options) {
    multipatch_t geometry;
    geometry.set_matching_tolerance(1e-6, 1e-6);

    geometry.addPatch(makeSquarePatch(0.0, 1.0, 0.0, 1.0,
                                      cfg.geometryNcoeffs, cfg.geometryDegree, options), 0);
    geometry.addPatch(makeSquarePatch(1.0, 2.0, 0.0, 1.0,
                                      cfg.geometryNcoeffs, cfg.geometryDegree, options), 1);
    geometry.addInterface(interface(0, iganet::side::east, 1, iganet::side::west));

    geometry.addBoundary({0, iganet::side::west, sideLabel(iganet::side::west)});
    geometry.addBoundary({1, iganet::side::east, sideLabel(iganet::side::east)});
    for (std::size_t p = 0; p < 2; ++p) {
        geometry.addBoundary({p, iganet::side::south, sideLabel(iganet::side::south)});
        geometry.addBoundary({p, iganet::side::north, sideLabel(iganet::side::north)});
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
        const std::array<iganet::short_t, 2> degrees{
            patch.degree(0), patch.degree(1)};
        const std::array<std::vector<real_t>, 2> knots{
            iganet::detail::tensor_to_vector<real_t>(patch.knots(0)),
            iganet::detail::tensor_to_vector<real_t>(patch.knots(1))};

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
        " to a patch in the loaded 2D multipatch geometry");
}

void resolvePatchConfigs(const multipatch_t& geometry, ParametricConfig& cfg) {
    for (auto& patchCfg : cfg.patchConfigs) {
        patchCfg.patch_id = static_cast<int>(resolvePatchIndex(geometry, patchCfg.patch_id));
    }
}

template <typename MultiPatch>
typename MultiPatch::patch_type localPatchWithTensor(const MultiPatch& space,
                                                     std::size_t patchIndex,
                                                     const torch::Tensor& tensor) {
    auto patch = space.patch(patchIndex);
    patch.from_tensor(space.local_tensor(patchIndex, tensor));
    return patch;
}

iganet::utils::TensorArray<2> toDevice(iganet::utils::TensorArray<2> xi,
                                       torch::Device device) {
    for (auto& x : xi) {
        x = x.to(device);
    }
    return xi;
}

torch::Tensor grevilleLine(const patch_t& patch,
                           iganet::short_t direction,
                           bool interior,
                           torch::Device device) {
    const auto knotsCpu = patch.knots(direction).to(torch::kCPU).contiguous();
    std::vector<double> knotVector(knotsCpu.numel());
    for (int64_t i = 0; i < knotsCpu.numel(); ++i) {
        knotVector[static_cast<std::size_t>(i)] = knotsCpu.index({i}).template item<double>();
    }

    auto line = computeGrevilleAbscissae(
        knotVector,
        static_cast<int>(patch.degree(direction)),
        static_cast<int>(patch.ncoeffs(direction)));

    if (interior) {
        if (line.size() <= 2) {
            return torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
        }
        line.erase(line.begin());
        line.pop_back();
    }

    return torch::tensor(line, torch::TensorOptions().dtype(torch::kFloat64).device(device));
}

template <typename EvalXi0, typename EvalXi1>
torch::Tensor stackParametricJacobian(const EvalXi0& dx,
                                      const EvalXi1& dy) {
    return torch::stack({
        torch::stack({*dx[0], *dy[0]}, 1),
        torch::stack({*dx[1], *dy[1]}, 1)}, 1);
}

template <typename EvalXX, typename EvalXY, typename EvalYY>
std::array<torch::Tensor, 2> stackParametricHessians(const EvalXX& xx,
                                                     const EvalXY& xy,
                                                     const EvalYY& yy) {
    std::array<torch::Tensor, 2> result;
    for (iganet::short_t c = 0; c < 2; ++c) {
        result[c] = torch::stack({
            torch::stack({*xx[c], *xy[c]}, 1),
            torch::stack({*xy[c], *yy[c]}, 1)}, 1);
    }
    return result;
}

template <typename Patch>
std::array<torch::Tensor, 2> parametricHessians(
    const Patch& patch,
    const iganet::utils::TensorArray<2>& xi) {
    const auto xx = patch.template eval<iganet::deriv::dx ^ 2>(xi);
    const auto xy = patch.template eval<iganet::deriv::dx + iganet::deriv::dy>(xi);
    const auto yy = patch.template eval<iganet::deriv::dy ^ 2>(xi);
    return stackParametricHessians(xx, xy, yy);
}

template <typename Optimizer>
class multipatch_linear_elasticity_2d
    : public iganet::IgANet<Optimizer, std::tuple<multipatch_t>, std::tuple<multipatch_t>> {
public:
    using base_t = iganet::IgANet<Optimizer, std::tuple<multipatch_t>, std::tuple<multipatch_t>>;
    using base_t::inputs;
    using base_t::outputs;
    using PreparedPointSet = typename patch_t::PreparedEvaluation;
    using CollocationData = typename iganet::CollPtsHelper<multipatch_t>::type;

    struct PatchResidualCache {
        std::size_t patchIndex{0};
        PreparedPointSet eval;
        torch::Tensor body;
        torch::Tensor J;
        torch::Tensor invJ;
        std::array<torch::Tensor, 2> hessG;
    };

    struct BoundaryTractionCache {
        std::size_t patchIndex{0};
        iganet::short_t side{0};
        PreparedPointSet eval;
        torch::Tensor target;
        torch::Tensor J;
        torch::Tensor invJ;
    };

    struct InterfaceCache {
        std::size_t patch1{0};
        iganet::short_t side1{0};
        PreparedPointSet eval1;
        torch::Tensor body1;
        torch::Tensor J1;
        torch::Tensor invJ1;
        std::array<torch::Tensor, 2> hessG1;
        std::size_t patch2{0};
        iganet::short_t side2{0};
        PreparedPointSet eval2;
        torch::Tensor body2;
        torch::Tensor J2;
        torch::Tensor invJ2;
        std::array<torch::Tensor, 2> hessG2;
    };

    struct LossParts {
        torch::Tensor total;
        torch::Tensor collocation;
        torch::Tensor traction;
        torch::Tensor interfaceTraction;
    };

    multipatch_linear_elasticity_2d(multipatch_t geometry,
                                    multipatch_t displacement,
                                    iganet::StrongDirichletConstraints<real_t> constraints,
                                    ParametricConfig cfg,
                                    std::vector<int64_t> layers,
                                    std::vector<std::vector<std::any>> activations,
                                    iganet::IgANetOptions defaults,
                                    iganet::Options<real_t> options)
        : base_t(defaults, options)
        , constraints_(std::move(constraints))
        , cfg_(std::move(cfg))
        , options_(torch::TensorOptions().dtype(torch::kFloat64).device(options.device())) {
        this->inputs_ = std::make_tuple(std::move(geometry));
        this->outputs_ = std::make_tuple(std::move(displacement));
        this->net_ = iganet::IgANetGenerator<real_t>(
            iganet::utils::concat(
                std::vector<int64_t>{this->inputs(0).size(0)},
                layers,
                std::vector<int64_t>{this->outputs(0).size(0)}),
            activations,
            options);
        this->net_->to(options.device(), options.dtype(), true);
        this->opt_ = std::make_unique<Optimizer>(this->net_->parameters());
        collocationData_ =
            iganet::CollPtsHelper<multipatch_t>::collPts(iganet::collPts::greville_interior,
                                                         this->geometry());

        lambda_ = (cfg_.youngModulus * cfg_.poissonRatio) /
                  ((1.0 + cfg_.poissonRatio) * (1.0 - 2.0 * cfg_.poissonRatio));
        mu_ = cfg_.youngModulus / (2.0 * (1.0 + cfg_.poissonRatio));

        const auto prepStart = std::chrono::steady_clock::now();
        prepare_caches();
        const auto prepEnd = std::chrono::steady_clock::now();
        preparationSeconds_ =
            std::chrono::duration<double>(prepEnd - prepStart).count();

        std::cout << "Preparation"
                  << " | seconds " << std::setw(12) << preparationSeconds_
                  << " | interior_sets " << patchResidualCaches_.size()
                  << " | traction_sets " << tractionCaches_.size()
                  << " | interface_sets " << interfaceCaches_.size()
                  << "\n";
    }

    bool epoch(int64_t epochIndex) override {
        std::cout << "\nEpoch " << epochIndex << "\n";
        return true;
    }

    torch::Tensor inputs(int64_t epoch) const override {
        return base_t::inputs(epoch).to(options_.device());
    }

    torch::Tensor outputs(int64_t epoch) const override {
        return base_t::outputs(epoch).to(options_.device());
    }

    torch::Tensor loss(const torch::Tensor& outputs, int64_t) override {
        const auto displacementTensor = constraints_.apply(outputs);
        const auto parts = loss_parts(displacementTensor);
        history_.push_back(parts.total.detach().template item<double>());
        std::cout << "  loss"
                  << " | total " << std::setw(14) << parts.total.detach().template item<double>()
                  << " | coll " << std::setw(14) << parts.collocation.detach().template item<double>()
                  << " | traction " << std::setw(14) << parts.traction.detach().template item<double>()
                  << " | interface_t " << std::setw(14) << parts.interfaceTraction.detach().template item<double>()
                  << "\n";
        return parts.total;
    }

    void eval() {
        const auto outputs = this->net_->forward(this->inputs(0));
        base_t::outputs(constraints_.apply(outputs));
    }

    const auto& history() const noexcept {
        return history_;
    }

    double preparation_seconds() const noexcept {
        return preparationSeconds_;
    }

    const auto& geometry() const {
        return this->template input<0>();
    }

    const auto& displacement() const {
        return this->template output<0>();
    }

private:
    const patch_config_2d_t* patch_config(std::size_t patchIndex) const {
        for (const auto& entry : cfg_.patchConfigs) {
            if (static_cast<std::size_t>(entry.patch_id) == patchIndex) {
                return &entry;
            }
        }
        return nullptr;
    }

    std::array<double, 2> body_force(std::size_t patchIndex) const {
        if (const auto* entry = patch_config(patchIndex)) {
            return entry->body_force;
        }
        return {0.0, 0.0};
    }

    bool hasDirichletSide(std::size_t patchIndex, int side) const {
        if (const auto* entry = patch_config(patchIndex)) {
            return std::any_of(
                entry->diri_sides.begin(), entry->diri_sides.end(),
                [&](const auto& bc) { return bc.side == side; });
        }
        return false;
    }

    bool hasForceSide(std::size_t patchIndex, int side) const {
        if (const auto* entry = patch_config(patchIndex)) {
            return std::any_of(
                entry->force_sides.begin(), entry->force_sides.end(),
                [&](const auto& bc) { return bc.side == side; });
        }
        return false;
    }

    bool hasTfbcSide(std::size_t patchIndex, int side) const {
        if (const auto* entry = patch_config(patchIndex)) {
            return std::find(entry->tfbc_sides.begin(), entry->tfbc_sides.end(), side) !=
                   entry->tfbc_sides.end();
        }
        return false;
    }

    bool side_is_occupied(std::size_t patchIndex,
                          int side,
                          bool includeTfbcAsOccupied) const {
        if (hasDirichletSide(patchIndex, side) || hasForceSide(patchIndex, side)) {
            return true;
        }
        return includeTfbcAsOccupied && hasTfbcSide(patchIndex, side);
    }

    iganet::utils::TensorArray<2> trim_side_collocation_points(
        std::size_t patchIndex,
        int side,
        iganet::utils::TensorArray<2> xi,
        bool includeTfbcAsOccupied) const {
        if (xi[0].numel() == 0) {
            return xi;
        }

        int64_t begin = 0;
        int64_t end = xi[0].size(0);
        switch (side) {
            case iganet::side::west:
            case iganet::side::east:
                if (side_is_occupied(patchIndex, iganet::side::south, includeTfbcAsOccupied)) {
                    begin += 1;
                }
                if (side_is_occupied(patchIndex, iganet::side::north, includeTfbcAsOccupied)) {
                    end -= 1;
                }
                break;
            case iganet::side::south:
            case iganet::side::north:
                if (side_is_occupied(patchIndex, iganet::side::west, includeTfbcAsOccupied)) {
                    begin += 1;
                }
                if (side_is_occupied(patchIndex, iganet::side::east, includeTfbcAsOccupied)) {
                    end -= 1;
                }
                break;
            default:
                throw std::runtime_error("Unsupported 2D boundary side.");
        }

        if (end < begin) {
            end = begin;
        }

        for (auto& x : xi) {
            x = x.slice(0, begin, end);
        }
        return xi;
    }

    iganet::utils::TensorArray<2> boundary_collocation_points(
        const typename multipatch_t::boundary_type& boundary,
        bool includeTfbcAsOccupied) const {
        for (const auto& [candidate, xi] : collocationData_.boundary()) {
            if (candidate.patch == boundary.patch && candidate.side == boundary.side) {
                return trim_side_collocation_points(
                    boundary.patch,
                    boundary.side,
                    toDevice(xi, options_.device()),
                    includeTfbcAsOccupied);
            }
        }
        throw std::runtime_error("Could not find requested boundary collocation set");
    }

    std::pair<iganet::utils::TensorArray<2>, iganet::utils::TensorArray<2>>
    interface_collocation_points(const interface_t& iface) const {
        for (const auto& [candidate, xis] : collocationData_.interfaces) {
            if (candidate.patch1 == iface.patch1 && candidate.side1 == iface.side1 &&
                candidate.patch2 == iface.patch2 && candidate.side2 == iface.side2) {
                auto xi1 = trim_side_collocation_points(
                    iface.patch1, iface.side1, toDevice(xis.first, options_.device()), true);
                auto xi2 = trim_side_collocation_points(
                    iface.patch2, iface.side2, toDevice(xis.second, options_.device()), true);
                return {std::move(xi1), std::move(xi2)};
            }
        }
        throw std::runtime_error("Could not find requested interface collocation set");
    }

    torch::Tensor make_body_tensor(std::size_t patchIndex) const {
        const auto patchBodyForce = body_force(patchIndex);
        return torch::tensor(
            {patchBodyForce[0], patchBodyForce[1]}, options_).view({1, 2});
    }

    PreparedPointSet prepare_point_set(std::size_t patchIndex,
                                       const iganet::utils::TensorArray<2>& xi) const {
        const auto G = geometry().patch(patchIndex);
        return G.template prepare_evaluation<
            iganet::deriv::dx,
            iganet::deriv::dy,
            iganet::deriv::dx ^ 2,
            iganet::deriv::dx + iganet::deriv::dy,
            iganet::deriv::dy ^ 2>(xi);
    }

    std::tuple<torch::Tensor, torch::Tensor, std::array<torch::Tensor, 2>>
    prepare_geometry_terms(std::size_t patchIndex, const PreparedPointSet& cache) const {
        if (cache.numeval == 0) {
            return {
                torch::empty({0, 2, 2}, options_),
                torch::empty({0, 2, 2}, options_),
                {torch::empty({0, 2, 2}, options_), torch::empty({0, 2, 2}, options_)}};
        }

        const auto G = geometry().patch(patchIndex);
        const auto gdx = G.template eval_from_prepared<iganet::deriv::dx>(cache);
        const auto gdy = G.template eval_from_prepared<iganet::deriv::dy>(cache);
        const auto J = stackParametricJacobian(gdx, gdy);
        const auto invJ = torch::linalg_inv(J);

        const auto gxx = G.template eval_from_prepared<iganet::deriv::dx ^ 2>(cache);
        const auto gxy =
            G.template eval_from_prepared<iganet::deriv::dx + iganet::deriv::dy>(cache);
        const auto gyy = G.template eval_from_prepared<iganet::deriv::dy ^ 2>(cache);
        const auto hessG = stackParametricHessians(gxx, gxy, gyy);
        return {J, invJ, hessG};
    }

    void prepare_caches() {
        patchResidualCaches_.clear();
        tractionCaches_.clear();
        interfaceCaches_.clear();

        patchResidualCaches_.reserve(geometry().npatches());
        for (std::size_t patchIndex = 0; patchIndex < geometry().npatches(); ++patchIndex) {
            auto xi = toDevice(collocationData_.interior()[patchIndex], options_.device());
            auto eval = prepare_point_set(patchIndex, xi);
            auto [J, invJ, hessG] = prepare_geometry_terms(patchIndex, eval);
            PatchResidualCache cache;
            cache.patchIndex = patchIndex;
            cache.eval = std::move(eval);
            cache.body = make_body_tensor(patchIndex);
            cache.J = std::move(J);
            cache.invJ = std::move(invJ);
            cache.hessG = std::move(hessG);
            patchResidualCaches_.push_back(std::move(cache));
        }

        for (const auto& patchCfg : cfg_.patchConfigs) {
            const auto patchIndex = static_cast<std::size_t>(patchCfg.patch_id);
            for (const auto side : patchCfg.tfbc_sides) {
                auto xi = boundary_collocation_points(
                    {patchIndex, static_cast<iganet::short_t>(side), ""}, false);
                auto eval = prepare_point_set(patchIndex, xi);
                auto [J, invJ, hessG] = prepare_geometry_terms(patchIndex, eval);
                BoundaryTractionCache cache;
                cache.patchIndex = patchIndex;
                cache.side = static_cast<iganet::short_t>(side);
                cache.eval = std::move(eval);
                cache.target = torch::zeros({1, 2}, options_);
                cache.J = std::move(J);
                cache.invJ = std::move(invJ);
                tractionCaches_.push_back(std::move(cache));
            }

            for (const auto& entry : patchCfg.force_sides) {
                auto xi = boundary_collocation_points(
                    {patchIndex, static_cast<iganet::short_t>(entry.side), ""}, false);
                auto eval = prepare_point_set(patchIndex, xi);
                auto [J, invJ, hessG] = prepare_geometry_terms(patchIndex, eval);
                BoundaryTractionCache cache;
                cache.patchIndex = patchIndex;
                cache.side = static_cast<iganet::short_t>(entry.side);
                cache.eval = std::move(eval);
                cache.target = torch::tensor({entry.x, entry.y}, options_).view({1, 2});
                cache.J = std::move(J);
                cache.invJ = std::move(invJ);
                tractionCaches_.push_back(std::move(cache));
            }
        }

        interfaceCaches_.reserve(geometry().ninterfaces());
        for (const auto& iface : geometry().interfaces()) {
            auto [xi1, xi2] = interface_collocation_points(iface);
            auto eval1 = prepare_point_set(iface.patch1, xi1);
            auto eval2 = prepare_point_set(iface.patch2, xi2);
            auto [J1, invJ1, hessG1] = prepare_geometry_terms(iface.patch1, eval1);
            auto [J2, invJ2, hessG2] = prepare_geometry_terms(iface.patch2, eval2);
            InterfaceCache cache;
            cache.patch1 = iface.patch1;
            cache.side1 = iface.side1;
            cache.eval1 = std::move(eval1);
            cache.body1 = make_body_tensor(iface.patch1);
            cache.J1 = std::move(J1);
            cache.invJ1 = std::move(invJ1);
            cache.hessG1 = std::move(hessG1);
            cache.patch2 = iface.patch2;
            cache.side2 = iface.side2;
            cache.eval2 = std::move(eval2);
            cache.body2 = make_body_tensor(iface.patch2);
            cache.J2 = std::move(J2);
            cache.invJ2 = std::move(invJ2);
            cache.hessG2 = std::move(hessG2);
            interfaceCaches_.push_back(std::move(cache));
        }
    }

    torch::Tensor evaluate_parametric_gradient(
        const patch_t& U,
        const PreparedPointSet& cache) const {
        const auto udx = U.template eval_from_prepared<iganet::deriv::dx>(cache);
        const auto udy = U.template eval_from_prepared<iganet::deriv::dy>(cache);
        return stackParametricJacobian(udx, udy);
    }

    std::array<torch::Tensor, 2> evaluate_parametric_hessians(
        const patch_t& U,
        const PreparedPointSet& cache) const {
        const auto uxx = U.template eval_from_prepared<iganet::deriv::dx ^ 2>(cache);
        const auto uxy =
            U.template eval_from_prepared<iganet::deriv::dx + iganet::deriv::dy>(cache);
        const auto uyy = U.template eval_from_prepared<iganet::deriv::dy ^ 2>(cache);
        auto hess = stackParametricHessians(uxx, uxy, uyy);
        return {std::move(hess[0]), std::move(hess[1])};
    }

    torch::Tensor strong_form_residual(
        std::size_t patchIndex,
        const torch::Tensor& displacementTensor,
        const PreparedPointSet& cache,
        const torch::Tensor& invJ,
        const std::array<torch::Tensor, 2>& hessG,
        const torch::Tensor& body) const {
        if (cache.numeval == 0) {
            return torch::empty({0, 2}, options_);
        }

        auto U = localPatchWithTensor(displacement(), patchIndex, displacementTensor);
        const auto gradUxi = evaluate_parametric_gradient(U, cache);
        const auto gradU = torch::matmul(gradUxi, invJ);
        const auto hessUxi = evaluate_parametric_hessians(U, cache);
        std::array<torch::Tensor, 2> hessU;
        for (iganet::short_t c = 0; c < 2; ++c) {
            auto corrected = hessUxi[c].clone();
            for (iganet::short_t k = 0; k < 2; ++k) {
                corrected = corrected -
                            gradU.index({torch::indexing::Slice(), c, k}).view({-1, 1, 1}) *
                                hessG[k];
            }
            hessU[c] = torch::matmul(invJ.transpose(1, 2), torch::matmul(corrected, invJ));
        }

        const auto ux_xx = hessU[0].index({torch::indexing::Slice(), 0, 0});
        const auto ux_yy = hessU[0].index({torch::indexing::Slice(), 1, 1});
        const auto uy_xx = hessU[1].index({torch::indexing::Slice(), 0, 0});
        const auto uy_yy = hessU[1].index({torch::indexing::Slice(), 1, 1});
        const auto uy_xy = hessU[1].index({torch::indexing::Slice(), 0, 1});
        const auto ux_xy = hessU[0].index({torch::indexing::Slice(), 0, 1});

        const auto divStress = torch::stack({
            (lambda_ + 2.0 * mu_) * ux_xx + mu_ * ux_yy + (lambda_ + mu_) * uy_xy,
            mu_ * uy_xx + (lambda_ + 2.0 * mu_) * uy_yy + (lambda_ + mu_) * ux_xy}, 1);

        return divStress - body.repeat({divStress.size(0), 1});
    }

    LossParts loss_parts(const torch::Tensor& displacementTensor) const {
        auto collocationLoss = torch::zeros({}, options_);
        auto tractionLoss = torch::zeros({}, options_);
        auto interfaceTractionLoss = torch::zeros({}, options_);

        for (const auto& cache : patchResidualCaches_) {
            if (cache.eval.numeval == 0) {
                continue;
            }
            const auto residual = strong_form_residual(
                cache.patchIndex, displacementTensor, cache.eval, cache.invJ, cache.hessG,
                cache.body);
            collocationLoss = collocationLoss +
                              torch::mse_loss(residual, torch::zeros_like(residual));
        }

        for (const auto& cache : tractionCaches_) {
            if (cache.eval.numeval == 0) {
                continue;
            }
            const auto traction = traction_on_boundary(
                cache.patchIndex, cache.side, displacementTensor, cache.eval, cache.invJ);
            tractionLoss = tractionLoss + torch::mse_loss(
                traction, cache.target.repeat({traction.size(0), 1}));
        }

        for (const auto& iface : interfaceCaches_) {
            if (iface.eval1.numeval > 0 && iface.eval2.numeval > 0) {
                const auto t1 = traction_on_boundary(
                    iface.patch1, iface.side1, displacementTensor, iface.eval1, iface.invJ1);
                const auto t2 = traction_on_boundary(
                    iface.patch2, iface.side2, displacementTensor, iface.eval2, iface.invJ2);
                interfaceTractionLoss =
                    interfaceTractionLoss + torch::mse_loss(t1 + t2, torch::zeros_like(t1));
            }

            // const auto r1 = strong_form_residual(
            //     iface.patch1, displacementTensor, iface.eval1, iface.invJ1, iface.hessG1,
            //     iface.body1);
            // const auto r2 = strong_form_residual(
            //     iface.patch2, displacementTensor, iface.eval2, iface.invJ2, iface.hessG2,
            //     iface.body2);
            // if (iface.eval1.numeval > 0) {
            //     collocationLoss =
            //         collocationLoss + torch::mse_loss(r1, torch::zeros_like(r1));
            // }
            // if (iface.eval2.numeval > 0) {
            //     collocationLoss =
            //         collocationLoss + torch::mse_loss(r2, torch::zeros_like(r2));
            // }
        }

        return {
            collocationLoss + tractionLoss + interfaceTractionLoss,
            collocationLoss,
            tractionLoss,
            interfaceTractionLoss};
    }

    torch::Tensor traction_on_boundary(std::size_t patchIndex,
                                       iganet::short_t side,
                                       const torch::Tensor& displacementTensor,
                                       const PreparedPointSet& cache,
                                       const torch::Tensor& invJ) const {
        if (cache.numeval == 0) {
            return torch::empty({0, 2}, options_);
        }

        auto U = localPatchWithTensor(displacement(), patchIndex, displacementTensor);
        const auto gradUxi = evaluate_parametric_gradient(U, cache);
        const auto grad = torch::matmul(gradUxi, invJ);
        const auto ux_x = grad.index({torch::indexing::Slice(), 0, 0});
        const auto ux_y = grad.index({torch::indexing::Slice(), 0, 1});
        const auto uy_x = grad.index({torch::indexing::Slice(), 1, 0});
        const auto uy_y = grad.index({torch::indexing::Slice(), 1, 1});

        torch::Tensor tx;
        torch::Tensor ty;
        switch (side) {
            case iganet::side::west:
                tx = -lambda_ * (ux_x + uy_y) - 2.0 * mu_ * ux_x;
                ty = -mu_ * (uy_x + ux_y);
                break;
            case iganet::side::east:
                tx = lambda_ * (ux_x + uy_y) + 2.0 * mu_ * ux_x;
                ty = mu_ * (uy_x + ux_y);
                break;
            case iganet::side::south:
                tx = -mu_ * (uy_x + ux_y);
                ty = -lambda_ * (ux_x + uy_y) - 2.0 * mu_ * uy_y;
                break;
            case iganet::side::north:
                tx = mu_ * (uy_x + ux_y);
                ty = lambda_ * (ux_x + uy_y) + 2.0 * mu_ * uy_y;
                break;
            default:
                throw std::runtime_error("Unsupported 2D boundary side.");
        }

        return torch::stack({tx, ty}, 1);
    }

    iganet::StrongDirichletConstraints<real_t> constraints_;
    ParametricConfig cfg_;
    torch::TensorOptions options_;
    CollocationData collocationData_;
    std::vector<double> history_;
    std::vector<PatchResidualCache> patchResidualCaches_;
    std::vector<BoundaryTractionCache> tractionCaches_;
    std::vector<InterfaceCache> interfaceCaches_;
    double preparationSeconds_{0.0};
    double lambda_{0.0};
    double mu_{0.0};
};

nlohmann::json tensor2BlocksToJson(const torch::Tensor& tensor) {
    auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    const int64_t n = cpu.numel() / 2;
    nlohmann::json result = nlohmann::json::array();
    for (int64_t i = 0; i < n; ++i) {
        result.push_back({
            cpu.index({i}).item<double>(),
            cpu.index({i + n}).item<double>()});
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
        entry["control_points"] = tensor2BlocksToJson(localGeometry);
        entry["displacements"] = tensor2BlocksToJson(localDisplacement);
        entry["deformed_control_points"] =
            tensor2BlocksToJson(localGeometry + localDisplacement);
        patches.push_back(entry);
    }
    return patches;
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

    const auto configPath = repoRoot / "sim_config_2D.json";
    const auto resultPath = repoRoot / "results" / "result_multipatch_2D_parametric.json";
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

    try {
        auto cfg = loadConfig(j);
        const auto computeDevice = resolveComputeDevice(j);
        const auto options = iganet::Options<real_t>{}.device(computeDevice);
        const auto mode = parseGeometryMode(j);
        multipatch_t geometry;
        std::optional<std::filesystem::path> xmlPath;
        if (mode == GeometryMode::Xml) {
            xmlPath = resolveXmlPath(repoRoot, j);
            geometry = makeXmlGeometry(*xmlPath, resolveMultiPatchId(j), options);
        } else {
            geometry = makeTwoSquareGeometry(cfg, options);
        }
        moveMultipatchToDevice(geometry, computeDevice);
        resolvePatchConfigs(geometry, cfg);
        auto displacement = geometry.make_isoparametric_solution_space<2>(options);

        iganet::StrongDirichletConstraints<real_t> constraints(displacement);
        for (const auto& patchCfg : cfg.patchConfigs) {
            const auto patchIndex = static_cast<std::size_t>(patchCfg.patch_id);
            for (const auto& entry : patchCfg.diri_sides) {
                constraints
                    .fix_boundary(displacement, patchIndex, entry.side, 0, entry.x)
                    .fix_boundary(displacement, patchIndex, entry.side, 1, entry.y);
            }
        }

        iganet::IgANetOptions netDefaults;
        netDefaults.max_epoch(cfg.maxEpoch);
        netDefaults.min_loss(1e-7);
        netDefaults.min_loss_change(0.0);
        netDefaults.min_loss_rel_change(0.0);

        auto run = [&]<typename optimizer_t>() {
            using net_t = multipatch_linear_elasticity_2d<optimizer_t>;

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

            net.train();
            net.eval();

            auto geometryOut = net.geometry();
            auto displacementOut = net.displacement();
            auto displacementTensor = displacementOut.as_tensor().detach();
            auto geometryTensor = geometryOut.as_tensor();
            auto history = net.history();
            auto preparationSeconds = net.preparation_seconds();

            return std::tuple{
                std::move(geometryOut),
                std::move(displacementOut),
                std::move(displacementTensor),
                std::move(geometryTensor),
                std::move(history),
                preparationSeconds};
        };

        auto [geometryOut, displacementOut, displacementTensor, geometryTensor, history,
              preparationSeconds] =
            (cfg.optimizer.type == optimizer_type_t::adam)
                ? run.template operator()<torch::optim::Adam>()
                : run.template operator()<torch::optim::LBFGS>();

        nlohmann::json summary;
        summary["example"] = "multipatch_parametric_2d";
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
        summary["preparation_seconds"] = preparationSeconds;
        summary["loss_initial"] = history.empty() ? 0.0 : history.front();
        summary["loss_final"] = history.empty() ? 0.0 : history.back();
        summary["loss_history"] = history;
        summary["geometry_control_points"] = tensor2BlocksToJson(geometryTensor);
        summary["displacements"] = tensor2BlocksToJson(displacementTensor);
        summary["deformed_control_points"] =
            tensor2BlocksToJson(geometryTensor + displacementTensor);
        summary["patches"] = patchesToJson(geometryOut, displacementOut, displacementTensor);

        std::ofstream out(resultPath);
        nlohmann::json result;
        result["multipatch_elasticity_2d"] = summary;
        out << result.dump(1);

        std::cout << "\n=== PARAMETRIC MULTIPATCH 2D ===\n"
                  << "config: " << configPath << "\n"
                  << "device: " << computeDevice.str() << "\n"
                  << "patches: " << geometryOut.npatches() << "\n"
                  << "interfaces: " << geometryOut.ninterfaces() << "\n"
                  << "scalar dofs: " << geometryOut.ndofs() << "\n"
                  << "fixed dofs: " << constraints.nfixed() << "\n"
                  << "free dofs: " << constraints.nfree() << "\n"
                  << "preparation [s]: " << preparationSeconds << "\n"
                  << "loss initial: " << summary["loss_initial"] << "\n"
                  << "loss final: " << summary["loss_final"] << "\n"
                  << "result: " << resultPath << "\n"
                  << "==========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "2D parametric MultiPatch example failed: " << e.what() << "\n";
        iganet::finalize();
        return 1;
    }

    iganet::finalize();
    return 0;
}
