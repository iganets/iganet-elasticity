#pragma once

#include <iganet.h>
#include <utils/config.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

template <typename Real>
struct MultipatchElasticityConfig {
    using real_t = Real;
    using boundary_value_t = std::tuple<int, real_t, real_t, real_t>;
    using patch_config_t = iganet_elasticity::utils::config::patch_config_3d;

    real_t youngModulus{210.0};
    real_t poissonRatio{0.25};
    real_t learningRate{1.0};
    int maxEpoch{30};
    int lbfgsHistorySize{50};
    int degree{2};
    int ncoeffs{3};
    std::array<real_t, 3> bodyForce{0.0, 0.0, 0.0};
    std::vector<boundary_value_t> diriSides;
    std::vector<boundary_value_t> forceSides;
    std::vector<int> tfbcSides;
    std::vector<patch_config_t> patchConfigs;
};

namespace iganet_elasticity::multipatch {

template <typename EvalXi0, typename EvalXi1, typename EvalXi2>
inline torch::Tensor stack_parametric_jacobian(const EvalXi0& dx,
                                               const EvalXi1& dy,
                                               const EvalXi2& dz) {
    return torch::stack({
        torch::stack({*dx[0], *dy[0], *dz[0]}, 1),
        torch::stack({*dx[1], *dy[1], *dz[1]}, 1),
        torch::stack({*dx[2], *dy[2], *dz[2]}, 1)}, 1);
}

template <typename EvalXX, typename EvalXY, typename EvalXZ,
          typename EvalYY, typename EvalYZ, typename EvalZZ>
inline std::array<torch::Tensor, 3> stack_parametric_hessians(
    const EvalXX& xx,
    const EvalXY& xy,
    const EvalXZ& xz,
    const EvalYY& yy,
    const EvalYZ& yz,
    const EvalZZ& zz) {
    std::array<torch::Tensor, 3> result;
    for (iganet::short_t c = 0; c < 3; ++c) {
        result[c] = torch::stack({
            torch::stack({*xx[c], *xy[c], *xz[c]}, 1),
            torch::stack({*xy[c], *yy[c], *yz[c]}, 1),
            torch::stack({*xz[c], *yz[c], *zz[c]}, 1)}, 1);
    }
    return result;
}

template <typename Patch>
inline std::array<torch::Tensor, 3> parametric_hessians(
    const Patch& patch,
    const iganet::utils::TensorArray<3>& xi) {
    const auto xx = patch.template eval<iganet::deriv::dx ^ 2>(xi);
    const auto xy = patch.template eval<iganet::deriv::dx + iganet::deriv::dy>(xi);
    const auto xz = patch.template eval<iganet::deriv::dx + iganet::deriv::dz>(xi);
    const auto yy = patch.template eval<iganet::deriv::dy ^ 2>(xi);
    const auto yz = patch.template eval<iganet::deriv::dy + iganet::deriv::dz>(xi);
    const auto zz = patch.template eval<iganet::deriv::dz ^ 2>(xi);
    return stack_parametric_hessians(xx, xy, xz, yy, yz, zz);
}

template <typename MultiPatch>
inline typename MultiPatch::patch_type local_patch_with_tensor(
    const MultiPatch& space,
    std::size_t patchIndex,
    const torch::Tensor& tensor) {
    auto patch = space.patch(patchIndex);
    patch.from_tensor(space.local_tensor(patchIndex, tensor));
    return patch;
}

inline iganet::utils::TensorArray<3> to_device(
    iganet::utils::TensorArray<3> xi,
    torch::Device device) {
    for (auto& x : xi) {
        x = x.to(device);
    }
    return xi;
}

template <typename Optimizer, typename MultiPatch>
class linear_elasticity
    : public iganet::IgANet<Optimizer, std::tuple<MultiPatch>, std::tuple<MultiPatch>> {
public:
    using real_t = typename MultiPatch::value_type;
    using patch_t = typename MultiPatch::patch_type;
    using base_t = iganet::IgANet<Optimizer, std::tuple<MultiPatch>, std::tuple<MultiPatch>>;
    using config_t = MultipatchElasticityConfig<real_t>;
    using PreparedPointSet = typename patch_t::PreparedEvaluation;

    struct PatchResidualCache {
        std::size_t patchIndex{0};
        PreparedPointSet eval;
        torch::Tensor body;
        torch::Tensor J;
        torch::Tensor invJ;
        std::array<torch::Tensor, 3> hessG;
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
        torch::Tensor J1;
        torch::Tensor invJ1;
        std::size_t patch2{0};
        iganet::short_t side2{0};
        PreparedPointSet eval2;
        torch::Tensor J2;
        torch::Tensor invJ2;
    };

    struct LossParts {
        torch::Tensor total;
        torch::Tensor collocation;
        torch::Tensor traction;
        torch::Tensor interfaceTraction;
    };

    linear_elasticity(MultiPatch geometry,
                      MultiPatch displacement,
                      iganet::StrongDirichletConstraints<real_t> constraints,
                      config_t cfg,
                      std::vector<int64_t> layers,
                      std::vector<std::vector<std::any>> activations,
                      iganet::IgANetOptions defaults,
                      iganet::Options<real_t> options)
        : base_t(defaults, options)
        , constraints_(std::move(constraints))
        , cfg_(std::move(cfg))
        , tensorOptions_(torch::TensorOptions().dtype(torch::kFloat64).device(options.device())) {
        this->inputs_ = std::make_tuple(std::move(geometry));
        this->outputs_ = std::make_tuple(std::move(displacement));
        this->net_ = iganet::IgANetGenerator<real_t>(
            iganet::utils::concat(
                std::vector<int64_t>{this->inputs(0).size(0)},
                layers,
                std::vector<int64_t>{this->outputs(0).size(0)}),
            activations,
            options);
        this->opt_ = std::make_unique<Optimizer>(this->net_->parameters());

        lambda_ = (cfg_.youngModulus * cfg_.poissonRatio) /
                  ((1.0 + cfg_.poissonRatio) * (1.0 - 2.0 * cfg_.poissonRatio));
        mu_ = cfg_.youngModulus / (2.0 * (1.0 + cfg_.poissonRatio));

        prepare_caches();
    }

    bool epoch(int64_t) override {
        return true;
    }

    torch::Tensor loss(const torch::Tensor& outputs, int64_t) override {
        const auto displacementTensor = constraints_.apply(outputs);
        const auto parts = loss_parts(displacementTensor);
        lastLossParts_ = {
            parts.total.detach().template item<double>(),
            parts.collocation.detach().template item<double>(),
            parts.traction.detach().template item<double>(),
            parts.interfaceTraction.detach().template item<double>()};
        history_.push_back(lastLossParts_[0]);
        std::cout << "loss"
                  << " | total " << std::setw(14) << lastLossParts_[0]
                  << " | coll " << std::setw(14) << lastLossParts_[1]
                  << " | traction " << std::setw(14) << lastLossParts_[2]
                  << " | interface " << std::setw(14) << lastLossParts_[3] << "\n";
        return parts.total;
    }

    void eval() {
        const auto outputs = this->net_->forward(this->inputs(0));
        this->outputs(constraints_.apply(outputs));
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

    LossParts loss(const torch::Tensor& displacementTensor) const {
        return loss_parts(displacementTensor);
    }

private:
    bool has_patch_configs() const {
        return !cfg_.patchConfigs.empty();
    }

    const typename config_t::patch_config_t* patch_config(std::size_t patchIndex) const {
        for (const auto& entry : cfg_.patchConfigs) {
            if (static_cast<std::size_t>(entry.patch_id) == patchIndex) {
                return &entry;
            }
        }
        return nullptr;
    }

    std::array<real_t, 3> body_force(std::size_t patchIndex) const {
        if (const auto* entry = patch_config(patchIndex)) {
            return {entry->body_force[0], entry->body_force[1], entry->body_force[2]};
        }
        return cfg_.bodyForce;
    }

    torch::Tensor make_body_tensor(std::size_t patchIndex) const {
        const auto patchBodyForce = body_force(patchIndex);
        return torch::tensor(
            {patchBodyForce[0], patchBodyForce[1], patchBodyForce[2]}, tensorOptions_)
            .view({1, 3});
    }

    PreparedPointSet prepare_point_set(std::size_t patchIndex,
                                       const iganet::utils::TensorArray<3>& xi) const {
        const auto G = geometry().patch(patchIndex);
        return G.template prepare_evaluation<
            iganet::deriv::dx,
            iganet::deriv::dy,
            iganet::deriv::dz,
            iganet::deriv::dx ^ 2,
            iganet::deriv::dx + iganet::deriv::dy,
            iganet::deriv::dx + iganet::deriv::dz,
            iganet::deriv::dy ^ 2,
            iganet::deriv::dy + iganet::deriv::dz,
            iganet::deriv::dz ^ 2>(xi);
    }

    std::tuple<torch::Tensor, torch::Tensor, std::array<torch::Tensor, 3>>
    prepare_geometry_terms(std::size_t patchIndex, const PreparedPointSet& cache) const {
        if (cache.numeval == 0) {
            return {
                torch::empty({0, 3, 3}, tensorOptions_),
                torch::empty({0, 3, 3}, tensorOptions_),
                {
                    torch::empty({0, 3, 3}, tensorOptions_),
                    torch::empty({0, 3, 3}, tensorOptions_),
                    torch::empty({0, 3, 3}, tensorOptions_)}};
        }

        const auto G = geometry().patch(patchIndex);
        const auto gdx = G.template eval_from_prepared<iganet::deriv::dx>(cache);
        const auto gdy = G.template eval_from_prepared<iganet::deriv::dy>(cache);
        const auto gdz = G.template eval_from_prepared<iganet::deriv::dz>(cache);
        const auto J = stack_parametric_jacobian(gdx, gdy, gdz);
        const auto invJ = torch::linalg_inv(J);

        const auto gxx = G.template eval_from_prepared<iganet::deriv::dx ^ 2>(cache);
        const auto gxy =
            G.template eval_from_prepared<iganet::deriv::dx + iganet::deriv::dy>(cache);
        const auto gxz =
            G.template eval_from_prepared<iganet::deriv::dx + iganet::deriv::dz>(cache);
        const auto gyy = G.template eval_from_prepared<iganet::deriv::dy ^ 2>(cache);
        const auto gyz =
            G.template eval_from_prepared<iganet::deriv::dy + iganet::deriv::dz>(cache);
        const auto gzz = G.template eval_from_prepared<iganet::deriv::dz ^ 2>(cache);
        const auto hessG = stack_parametric_hessians(gxx, gxy, gxz, gyy, gyz, gzz);
        return {J, invJ, hessG};
    }

    void prepare_caches() {
        patchResidualCaches_.clear();
        tractionCaches_.clear();
        interfaceCaches_.clear();

        patchResidualCaches_.reserve(geometry().npatches());
        for (std::size_t patchIndex = 0; patchIndex < geometry().npatches(); ++patchIndex) {
            auto xi = to_device(geometry().patch(patchIndex).greville(true), tensorOptions_.device());
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

        if (has_patch_configs()) {
            for (const auto& patchCfg : cfg_.patchConfigs) {
                const auto patchIndex = static_cast<std::size_t>(patchCfg.patch_id);
                for (const auto side : patchCfg.tfbc_sides) {
                    auto xi = to_device(
                        geometry().side_greville(patchIndex, static_cast<iganet::short_t>(side)),
                        tensorOptions_.device());
                    auto eval = prepare_point_set(patchIndex, xi);
                    auto [J, invJ, hessG] = prepare_geometry_terms(patchIndex, eval);
                    BoundaryTractionCache cache;
                    cache.patchIndex = patchIndex;
                    cache.side = static_cast<iganet::short_t>(side);
                    cache.eval = std::move(eval);
                    cache.target = torch::zeros({1, 3}, tensorOptions_);
                    cache.J = std::move(J);
                    cache.invJ = std::move(invJ);
                    tractionCaches_.push_back(std::move(cache));
                }

                for (const auto& entry : patchCfg.force_sides) {
                    auto xi = to_device(
                        geometry().side_greville(patchIndex, static_cast<iganet::short_t>(entry.side)),
                        tensorOptions_.device());
                    auto eval = prepare_point_set(patchIndex, xi);
                    auto [J, invJ, hessG] = prepare_geometry_terms(patchIndex, eval);
                    BoundaryTractionCache cache;
                    cache.patchIndex = patchIndex;
                    cache.side = static_cast<iganet::short_t>(entry.side);
                    cache.eval = std::move(eval);
                    cache.target = torch::tensor({entry.x, entry.y, entry.z}, tensorOptions_)
                                       .view({1, 3});
                    cache.J = std::move(J);
                    cache.invJ = std::move(invJ);
                    tractionCaches_.push_back(std::move(cache));
                }
            }
        } else {
            for (const auto& side : cfg_.tfbcSides) {
                for (const auto& [boundary, xiRaw] : geometry().boundary_greville(side_label(side))) {
                    auto xi = to_device(xiRaw, tensorOptions_.device());
                    auto eval = prepare_point_set(boundary.patch, xi);
                    auto [J, invJ, hessG] = prepare_geometry_terms(boundary.patch, eval);
                    BoundaryTractionCache cache;
                    cache.patchIndex = boundary.patch;
                    cache.side = boundary.side;
                    cache.eval = std::move(eval);
                    cache.target = torch::zeros({1, 3}, tensorOptions_);
                    cache.J = std::move(J);
                    cache.invJ = std::move(invJ);
                    tractionCaches_.push_back(std::move(cache));
                }
            }

            for (const auto& entry : cfg_.forceSides) {
                const int side = std::get<0>(entry);
                for (const auto& [boundary, xiRaw] : geometry().boundary_greville(side_label(side))) {
                    auto xi = to_device(xiRaw, tensorOptions_.device());
                    auto eval = prepare_point_set(boundary.patch, xi);
                    auto [J, invJ, hessG] = prepare_geometry_terms(boundary.patch, eval);
                    BoundaryTractionCache cache;
                    cache.patchIndex = boundary.patch;
                    cache.side = boundary.side;
                    cache.eval = std::move(eval);
                    cache.target = torch::tensor(
                                       {std::get<1>(entry), std::get<2>(entry), std::get<3>(entry)},
                                       tensorOptions_)
                                       .view({1, 3});
                    cache.J = std::move(J);
                    cache.invJ = std::move(invJ);
                    tractionCaches_.push_back(std::move(cache));
                }
            }
        }

        interfaceCaches_.reserve(geometry().ninterfaces());
        for (const auto& interface : geometry().interfaces()) {
            auto [xi1, xi2] = geometry().interface_greville(interface);
            xi1 = to_device(xi1, tensorOptions_.device());
            xi2 = to_device(xi2, tensorOptions_.device());
            auto eval1 = prepare_point_set(interface.patch1, xi1);
            auto eval2 = prepare_point_set(interface.patch2, xi2);
            auto [J1, invJ1, hessG1] = prepare_geometry_terms(interface.patch1, eval1);
            auto [J2, invJ2, hessG2] = prepare_geometry_terms(interface.patch2, eval2);
            InterfaceCache cache;
            cache.patch1 = interface.patch1;
            cache.side1 = interface.side1;
            cache.eval1 = std::move(eval1);
            cache.J1 = std::move(J1);
            cache.invJ1 = std::move(invJ1);
            cache.patch2 = interface.patch2;
            cache.side2 = interface.side2;
            cache.eval2 = std::move(eval2);
            cache.J2 = std::move(J2);
            cache.invJ2 = std::move(invJ2);
            interfaceCaches_.push_back(std::move(cache));
        }
    }

    torch::Tensor evaluate_parametric_gradient(const patch_t& U,
                                               const PreparedPointSet& cache) const {
        const auto udx = U.template eval_from_prepared<iganet::deriv::dx>(cache);
        const auto udy = U.template eval_from_prepared<iganet::deriv::dy>(cache);
        const auto udz = U.template eval_from_prepared<iganet::deriv::dz>(cache);
        return stack_parametric_jacobian(udx, udy, udz);
    }

    std::array<torch::Tensor, 3> evaluate_parametric_hessians(
        const patch_t& U,
        const PreparedPointSet& cache) const {
        const auto uxx = U.template eval_from_prepared<iganet::deriv::dx ^ 2>(cache);
        const auto uxy =
            U.template eval_from_prepared<iganet::deriv::dx + iganet::deriv::dy>(cache);
        const auto uxz =
            U.template eval_from_prepared<iganet::deriv::dx + iganet::deriv::dz>(cache);
        const auto uyy = U.template eval_from_prepared<iganet::deriv::dy ^ 2>(cache);
        const auto uyz =
            U.template eval_from_prepared<iganet::deriv::dy + iganet::deriv::dz>(cache);
        const auto uzz = U.template eval_from_prepared<iganet::deriv::dz ^ 2>(cache);
        return stack_parametric_hessians(uxx, uxy, uxz, uyy, uyz, uzz);
    }

    torch::Tensor strong_form_residual(
        std::size_t patchIndex,
        const torch::Tensor& displacementTensor,
        const PreparedPointSet& cache,
        const torch::Tensor& invJ,
        const std::array<torch::Tensor, 3>& hessG,
        const torch::Tensor& body) const {
        if (cache.numeval == 0) {
            return torch::empty({0, 3}, tensorOptions_);
        }

        const auto U = local_patch_with_tensor(displacement(), patchIndex, displacementTensor);
        const auto gradUxi = evaluate_parametric_gradient(U, cache);
        const auto gradU = torch::matmul(gradUxi, invJ);
        const auto hessUxi = evaluate_parametric_hessians(U, cache);
        std::array<torch::Tensor, 3> hessU;
        for (iganet::short_t c = 0; c < 3; ++c) {
            auto corrected = hessUxi[c].clone();
            for (iganet::short_t k = 0; k < 3; ++k) {
                corrected = corrected -
                            gradU.index({torch::indexing::Slice(), c, k}).view({-1, 1, 1}) *
                                hessG[k];
            }
            hessU[c] = torch::matmul(invJ.transpose(1, 2), torch::matmul(corrected, invJ));
        }

        const auto ux_xx = hessU[0].index({torch::indexing::Slice(), 0, 0});
        const auto ux_yy = hessU[0].index({torch::indexing::Slice(), 1, 1});
        const auto ux_zz = hessU[0].index({torch::indexing::Slice(), 2, 2});
        const auto uy_xy = hessU[1].index({torch::indexing::Slice(), 0, 1});
        const auto uz_xz = hessU[2].index({torch::indexing::Slice(), 0, 2});

        const auto uy_xx = hessU[1].index({torch::indexing::Slice(), 0, 0});
        const auto uy_yy = hessU[1].index({torch::indexing::Slice(), 1, 1});
        const auto uy_zz = hessU[1].index({torch::indexing::Slice(), 2, 2});
        const auto ux_yx = hessU[0].index({torch::indexing::Slice(), 1, 0});
        const auto uz_yz = hessU[2].index({torch::indexing::Slice(), 1, 2});

        const auto uz_xx = hessU[2].index({torch::indexing::Slice(), 0, 0});
        const auto uz_yy = hessU[2].index({torch::indexing::Slice(), 1, 1});
        const auto uz_zz = hessU[2].index({torch::indexing::Slice(), 2, 2});
        const auto ux_zx = hessU[0].index({torch::indexing::Slice(), 2, 0});
        const auto uy_zy = hessU[1].index({torch::indexing::Slice(), 2, 1});

        const auto divStress = torch::stack({
            (lambda_ + 2.0 * mu_) * ux_xx + mu_ * ux_yy + mu_ * ux_zz +
                (lambda_ + mu_) * (uy_xy + uz_xz),
            mu_ * uy_xx + (lambda_ + 2.0 * mu_) * uy_yy + mu_ * uy_zz +
                (lambda_ + mu_) * (ux_yx + uz_yz),
            mu_ * uz_xx + mu_ * uz_yy + (lambda_ + 2.0 * mu_) * uz_zz +
                (lambda_ + mu_) * (ux_zx + uy_zy)}, 1);

        return divStress + body.repeat({divStress.size(0), 1});
    }

    LossParts loss_parts(const torch::Tensor& displacementTensor) const {
        auto collocationLoss = torch::zeros({}, tensorOptions_);
        auto tractionLoss = torch::zeros({}, tensorOptions_);
        auto interfaceLoss = torch::zeros({}, tensorOptions_);

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
                cache.patchIndex, cache.side, displacementTensor, cache.eval, cache.J, cache.invJ);
            tractionLoss = tractionLoss +
                           torch::mse_loss(traction, cache.target.repeat({traction.size(0), 1}));
        }

        for (const auto& cache : interfaceCaches_) {
            if (cache.eval1.numeval == 0 || cache.eval2.numeval == 0) {
                continue;
            }
            const auto t1 = traction_on_boundary(
                cache.patch1, cache.side1, displacementTensor, cache.eval1, cache.J1, cache.invJ1);
            const auto t2 = traction_on_boundary(
                cache.patch2, cache.side2, displacementTensor, cache.eval2, cache.J2, cache.invJ2);
            interfaceLoss = interfaceLoss + torch::mse_loss(t1 + t2, torch::zeros_like(t1));
        }

        return {
            collocationLoss + tractionLoss + interfaceLoss,
            collocationLoss,
            tractionLoss,
            interfaceLoss};
    }

    torch::Tensor traction_on_boundary(
        std::size_t patchIndex,
        iganet::short_t side,
        const torch::Tensor& displacementTensor,
        const PreparedPointSet& cache,
        const torch::Tensor& J,
        const torch::Tensor& invJ) const {
        if (cache.numeval == 0) {
            return torch::empty({0, 3}, tensorOptions_);
        }

        const auto U = local_patch_with_tensor(displacement(), patchIndex, displacementTensor);
        const auto gradUxi = evaluate_parametric_gradient(U, cache);
        const auto gradU = torch::matmul(gradUxi, invJ);
        const auto strain = 0.5 * (gradU + gradU.transpose(1, 2));
        const auto trace = strain.index({torch::indexing::Slice(), 0, 0}) +
                           strain.index({torch::indexing::Slice(), 1, 1}) +
                           strain.index({torch::indexing::Slice(), 2, 2});

        auto stress = 2.0 * mu_ * strain;
        for (iganet::short_t c = 0; c < 3; ++c) {
            stress.index_put_(
                {torch::indexing::Slice(), c, c},
                stress.index({torch::indexing::Slice(), c, c}) + lambda_ * trace);
        }

        const auto fixed = static_cast<iganet::short_t>((side - 1) / 2);
        const auto t0 = static_cast<iganet::short_t>(fixed == 0 ? 1 : 0);
        const auto t1 = static_cast<iganet::short_t>(fixed == 2 ? 1 : 2);
        const auto a = J.index({torch::indexing::Slice(), torch::indexing::Slice(), t0});
        const auto b = J.index({torch::indexing::Slice(), torch::indexing::Slice(), t1});
        auto normal = torch::cross(a, b, 1);
        if (!MultiPatch::interface_type::side_parameter(side)) {
            normal = -normal;
        }
        normal = normal / normal.norm(2, 1).clamp_min(1e-12).view({-1, 1});

        return torch::matmul(stress, normal.unsqueeze(2)).squeeze(2);
    }

    static std::string side_label(int side) {
        return "side_" + std::to_string(side);
    }

    iganet::StrongDirichletConstraints<real_t> constraints_;
    config_t cfg_;
    torch::TensorOptions tensorOptions_;
    std::vector<double> history_;
    std::vector<PatchResidualCache> patchResidualCaches_;
    std::vector<BoundaryTractionCache> tractionCaches_;
    std::vector<InterfaceCache> interfaceCaches_;
    std::array<double, 4> lastLossParts_{};
    double lambda_{0.0};
    double mu_{0.0};
};

} // namespace iganet_elasticity::multipatch
