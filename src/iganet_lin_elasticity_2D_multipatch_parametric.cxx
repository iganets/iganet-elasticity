#include "lin_elasticity_utils.hpp"

#include <iganet.h>

#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <any>
#include <array>
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
    int degree{3};
    int ncoeffs{5};
    optimizer_config_t optimizer;
    std::vector<patch_config_2d_t> patchConfigs;
};

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

    if (j.contains("spline")) {
        cfg.degree = require(j, "spline.degree").get<int>();
        cfg.ncoeffs = require(j, "spline.nr_ctrl_pts").get<int>();
    }

    if (cfg.degree < 1) {
        throw std::runtime_error("2D multipatch degree must be >= 1");
    }
    if (cfg.ncoeffs <= cfg.degree) {
        throw std::runtime_error("2D multipatch nr_ctrl_pts must be larger than degree");
    }
    if (cfg.maxEpoch <= 0) {
        throw std::runtime_error("simulation.max_epoch must be positive");
    }

    return cfg;
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

    geometry.addPatch(makeSquarePatch(0.0, 1.0, 0.0, 1.0, cfg.ncoeffs, cfg.degree, options), 0);
    geometry.addPatch(makeSquarePatch(1.0, 2.0, 0.0, 1.0, cfg.ncoeffs, cfg.degree, options), 1);
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

template <typename HasSideFn>
iganet::utils::TensorArray<2> buildSideCollocationPoints(
    const patch_t& patch,
    int side,
    bool trimCorners,
    torch::Device device,
    HasSideFn&& hasOccupiedSide,
    bool reverseDirection = false) {
    auto line = grevilleLine(
        patch,
        (side == iganet::side::west || side == iganet::side::east) ? 1 : 0,
        false,
        device);

    int64_t begin = 0;
    int64_t end = line.size(0);
    if (trimCorners) {
        switch (side) {
            case iganet::side::west:
            case iganet::side::east:
                if (hasOccupiedSide(iganet::side::south)) begin += 1;
                if (hasOccupiedSide(iganet::side::north)) end -= 1;
                break;
            case iganet::side::south:
            case iganet::side::north:
                if (hasOccupiedSide(iganet::side::west)) begin += 1;
                if (hasOccupiedSide(iganet::side::east)) end -= 1;
                break;
            default:
                throw std::runtime_error("Unsupported 2D boundary side.");
        }
    }

    line = line.slice(0, begin, end);
    if (reverseDirection) {
        line = torch::flip(line, {0});
    }

    switch (side) {
        case iganet::side::west:
            return {torch::zeros({line.size(0)}, line.options()), line};
        case iganet::side::east:
            return {torch::ones({line.size(0)}, line.options()), line};
        case iganet::side::south:
            return {line, torch::zeros({line.size(0)}, line.options())};
        case iganet::side::north:
            return {line, torch::ones({line.size(0)}, line.options())};
        default:
            throw std::runtime_error("Unsupported 2D boundary side.");
    }
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

        lambda_ = (cfg_.youngModulus * cfg_.poissonRatio) /
                  ((1.0 + cfg_.poissonRatio) * (1.0 - 2.0 * cfg_.poissonRatio));
        mu_ = cfg_.youngModulus / (2.0 * (1.0 + cfg_.poissonRatio));
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

    iganet::utils::TensorArray<2> build_side_collocation_points(
        std::size_t patchIndex,
        int side,
        bool trimCorners,
        bool reverseDirection = false,
        bool includeTfbcAsOccupied = false) const {
        return buildSideCollocationPoints(
            geometry().patch(patchIndex),
            side,
            trimCorners,
            options_.device(),
            [&](int testSide) {
                if (hasDirichletSide(patchIndex, testSide) ||
                    hasForceSide(patchIndex, testSide)) {
                    return true;
                }
                if (includeTfbcAsOccupied) {
                    return hasTfbcSide(patchIndex, testSide);
                }
                return false;
            },
            reverseDirection);
    }

    torch::Tensor compute_patch_pde_loss(
        std::size_t patchIndex,
        const torch::Tensor& displacementTensor) const {
        auto G = geometry().patch(patchIndex);
        const auto xi = toDevice(G.greville(true), options_.device());
        const auto patchBodyForce = body_force(patchIndex);
        const auto body = torch::tensor(
            {patchBodyForce[0], patchBodyForce[1]}, options_).view({1, 2});
        const auto residual =
            strong_form_residual(patchIndex, displacementTensor, xi, body);
        return torch::mse_loss(residual, torch::zeros_like(residual));
    }

    torch::Tensor strong_form_residual(
        std::size_t patchIndex,
        const torch::Tensor& displacementTensor,
        const iganet::utils::TensorArray<2>& xi,
        const torch::Tensor& body) const {
        auto G = geometry().patch(patchIndex);
        auto U = localPatchWithTensor(displacement(), patchIndex, displacementTensor);
        const auto gdx = G.template eval<iganet::deriv::dx>(xi);
        const auto gdy = G.template eval<iganet::deriv::dy>(xi);
        const auto udx = U.template eval<iganet::deriv::dx>(xi);
        const auto udy = U.template eval<iganet::deriv::dy>(xi);
        const auto J = stackParametricJacobian(gdx, gdy);
        const auto invJ = torch::linalg_inv(J);
        const auto gradUxi = stackParametricJacobian(udx, udy);
        const auto gradU = torch::matmul(gradUxi, invJ);

        const auto hessG = parametricHessians(G, xi);
        const auto hessUxi = parametricHessians(U, xi);
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

        for (std::size_t patchIndex = 0; patchIndex < geometry().npatches(); ++patchIndex) {
            collocationLoss =
                collocationLoss + compute_patch_pde_loss(patchIndex, displacementTensor);
        }

        for (const auto& patchCfg : cfg_.patchConfigs) {
            const auto patchIndex = static_cast<std::size_t>(patchCfg.patch_id);
            for (const auto side : patchCfg.tfbc_sides) {
                const auto xi = build_side_collocation_points(patchIndex, side, true);
                const auto traction =
                    traction_on_boundary(patchIndex, static_cast<iganet::short_t>(side),
                                         displacementTensor, xi);
                tractionLoss = tractionLoss +
                               torch::mse_loss(traction, torch::zeros_like(traction));
            }
        }

        for (const auto& patchCfg : cfg_.patchConfigs) {
            const auto patchIndex = static_cast<std::size_t>(patchCfg.patch_id);
            for (const auto& entry : patchCfg.force_sides) {
                const int side = entry.side;
                const auto target =
                    torch::tensor({entry.x, entry.y}, options_).view({1, 2});
                const auto xi = build_side_collocation_points(patchIndex, side, true);
                const auto traction =
                    traction_on_boundary(patchIndex, static_cast<iganet::short_t>(side),
                                         displacementTensor, xi);
                tractionLoss = tractionLoss +
                               torch::mse_loss(traction, target.repeat({traction.size(0), 1}));
            }
        }

        for (const auto& iface : geometry().interfaces()) {
            const auto xi1 = build_side_collocation_points(
                iface.patch1, iface.side1, true, false, true);
            const auto xi2 = build_side_collocation_points(
                iface.patch2, iface.side2, true, false, true);
            const auto t1 = traction_on_boundary(
                iface.patch1, iface.side1, displacementTensor, xi1);
            const auto t2 = traction_on_boundary(
                iface.patch2, iface.side2, displacementTensor, xi2);
            interfaceTractionLoss =
                interfaceTractionLoss + torch::mse_loss(t1 + t2, torch::zeros_like(t1));

            const auto body1 = body_force(iface.patch1);
            const auto body2 = body_force(iface.patch2);
            const auto r1 = strong_form_residual(
                iface.patch1,
                displacementTensor,
                xi1,
                torch::tensor({body1[0], body1[1]}, options_).view({1, 2}));
            const auto r2 = strong_form_residual(
                iface.patch2,
                displacementTensor,
                xi2,
                torch::tensor({body2[0], body2[1]}, options_).view({1, 2}));
            collocationLoss = collocationLoss +
                              torch::mse_loss(r1, torch::zeros_like(r1)) +
                              torch::mse_loss(r2, torch::zeros_like(r2));
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
                                       const iganet::utils::TensorArray<2>& xi) const {
        auto G = geometry().patch(patchIndex);
        auto U = localPatchWithTensor(displacement(), patchIndex, displacementTensor);

        const auto gdx = G.template eval<iganet::deriv::dx>(xi);
        const auto gdy = G.template eval<iganet::deriv::dy>(xi);
        const auto udx = U.template eval<iganet::deriv::dx>(xi);
        const auto udy = U.template eval<iganet::deriv::dy>(xi);
        const auto J = stackParametricJacobian(gdx, gdy);
        const auto invJ = torch::linalg_inv(J);
        const auto gradUxi = stackParametricJacobian(udx, udy);
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
    std::vector<double> history_;
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
    ::setenv("IGANET_DEVICE", "CPU", 1);
    ::setenv("IGANET_DEVICE_INDEX", "0", 1);

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

    const auto computeDevice = torch::Device(torch::kCPU);
    const auto options = iganet::Options<real_t>{}.device(computeDevice);

    try {
        auto cfg = loadConfig(j);
        const auto mode = parseGeometryMode(j);
        multipatch_t geometry;
        std::optional<std::filesystem::path> xmlPath;
        if (mode == GeometryMode::Xml) {
            xmlPath = resolveXmlPath(repoRoot, j);
            geometry = makeXmlGeometry(*xmlPath, resolveMultiPatchId(j), options);
        } else {
            geometry = makeTwoSquareGeometry(cfg, options);
        }
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

            return std::tuple{
                std::move(geometryOut),
                std::move(displacementOut),
                std::move(displacementTensor),
                std::move(geometryTensor),
                std::move(history)};
        };

        auto [geometryOut, displacementOut, displacementTensor, geometryTensor, history] =
            (cfg.optimizer.type == optimizer_type_t::adam)
                ? run.template operator()<torch::optim::Adam>()
                : run.template operator()<torch::optim::LBFGS>();

        nlohmann::json summary;
        summary["example"] = "two_parametric_squares";
        summary["geometry_mode"] = (mode == GeometryMode::Xml ? "xml" : "parametric");
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
        summary["geometry_control_points"] = tensor2BlocksToJson(geometryTensor);
        summary["displacements"] = tensor2BlocksToJson(displacementTensor);
        summary["deformed_control_points"] =
            tensor2BlocksToJson(geometryTensor + displacementTensor);
        summary["patches"] = patchesToJson(geometryOut, displacementOut, displacementTensor);

        std::ofstream out(resultPath);
        nlohmann::json result;
        result["multipatch_elasticity_2d"] = summary;
        out << result.dump(1);

        std::cout << "\n=== PARAMETRIC TWO-SQUARE MULTIPATCH 2D ===\n"
                  << "config: " << configPath << "\n"
                  << "patches: " << geometryOut.npatches() << "\n"
                  << "interfaces: " << geometryOut.ninterfaces() << "\n"
                  << "scalar dofs: " << geometryOut.ndofs() << "\n"
                  << "fixed dofs: " << constraints.nfixed() << "\n"
                  << "free dofs: " << constraints.nfree() << "\n"
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
