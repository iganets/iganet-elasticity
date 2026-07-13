#pragma once

#include <iganet.h>

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
    using base_t = iganet::IgANet<Optimizer, std::tuple<MultiPatch>, std::tuple<MultiPatch>>;
    using config_t = MultipatchElasticityConfig<real_t>;

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
    LossParts loss_parts(const torch::Tensor& displacementTensor) const {
        auto collocationLoss = torch::zeros({}, tensorOptions_);
        auto tractionLoss = torch::zeros({}, tensorOptions_);
        auto interfaceLoss = torch::zeros({}, tensorOptions_);
        const auto body = torch::tensor(
            {cfg_.bodyForce[0], cfg_.bodyForce[1], cfg_.bodyForce[2]}, tensorOptions_).view({1, 3});

        for (std::size_t patchIndex = 0; patchIndex < geometry().npatches(); ++patchIndex) {
            const auto xi = to_device(geometry().patch(patchIndex).greville(true),
                                      tensorOptions_.device());
            const auto divStress = div_stress_on_patch(patchIndex, displacementTensor, xi);
            collocationLoss = collocationLoss +
                              torch::mse_loss(divStress, -body.repeat({divStress.size(0), 1}));
        }

        for (const auto& side : cfg_.tfbcSides) {
            for (const auto& [boundary, xiRaw] : geometry().boundary_greville(side_label(side))) {
                const auto xi = to_device(xiRaw, tensorOptions_.device());
                const auto traction =
                    traction_on_boundary(boundary.patch, boundary.side, displacementTensor, xi);
                tractionLoss = tractionLoss +
                               torch::mse_loss(traction, torch::zeros_like(traction));
            }
        }

        for (const auto& entry : cfg_.forceSides) {
            const int side = std::get<0>(entry);
            const auto target = torch::tensor(
                {std::get<1>(entry), std::get<2>(entry), std::get<3>(entry)}, tensorOptions_)
                                    .view({1, 3});
            for (const auto& [boundary, xiRaw] : geometry().boundary_greville(side_label(side))) {
                const auto xi = to_device(xiRaw, tensorOptions_.device());
                const auto traction =
                    traction_on_boundary(boundary.patch, boundary.side, displacementTensor, xi);
                tractionLoss = tractionLoss +
                               torch::mse_loss(traction, target.repeat({traction.size(0), 1}));
            }
        }

        for (const auto& interface : geometry().interfaces()) {
            const auto [xi1Raw, xi2Raw] = geometry().interface_greville(interface);
            const auto xi1 = to_device(xi1Raw, tensorOptions_.device());
            const auto xi2 = to_device(xi2Raw, tensorOptions_.device());
            const auto t1 = traction_on_boundary(
                interface.patch1, interface.side1, displacementTensor, xi1);
            const auto t2 = traction_on_boundary(
                interface.patch2, interface.side2, displacementTensor, xi2);
            interfaceLoss = interfaceLoss + torch::mse_loss(t1 + t2, torch::zeros_like(t1));
        }

        return {
            collocationLoss + tractionLoss + interfaceLoss,
            collocationLoss,
            tractionLoss,
            interfaceLoss};
    }

    torch::Tensor div_stress_on_patch(
        std::size_t patchIndex,
        const torch::Tensor& displacementTensor,
        const iganet::utils::TensorArray<3>& xi) const {
        const auto G = geometry().patch(patchIndex);
        const auto U = local_patch_with_tensor(displacement(), patchIndex, displacementTensor);

        const auto gdx = G.template eval<iganet::deriv::dx>(xi);
        const auto gdy = G.template eval<iganet::deriv::dy>(xi);
        const auto gdz = G.template eval<iganet::deriv::dz>(xi);
        const auto udx = U.template eval<iganet::deriv::dx>(xi);
        const auto udy = U.template eval<iganet::deriv::dy>(xi);
        const auto udz = U.template eval<iganet::deriv::dz>(xi);

        const auto J = stack_parametric_jacobian(gdx, gdy, gdz);
        const auto invJ = torch::linalg_inv(J);
        const auto gradUxi = stack_parametric_jacobian(udx, udy, udz);
        const auto gradU = torch::matmul(gradUxi, invJ);

        const auto hessG = parametric_hessians(G, xi);
        const auto hessUxi = parametric_hessians(U, xi);
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

        return torch::stack({
            (lambda_ + 2.0 * mu_) * ux_xx + mu_ * ux_yy + mu_ * ux_zz +
                (lambda_ + mu_) * (uy_xy + uz_xz),
            mu_ * uy_xx + (lambda_ + 2.0 * mu_) * uy_yy + mu_ * uy_zz +
                (lambda_ + mu_) * (ux_yx + uz_yz),
            mu_ * uz_xx + mu_ * uz_yy + (lambda_ + 2.0 * mu_) * uz_zz +
                (lambda_ + mu_) * (ux_zx + uy_zy)}, 1);
    }

    torch::Tensor traction_on_boundary(
        std::size_t patchIndex,
        iganet::short_t side,
        const torch::Tensor& displacementTensor,
        const iganet::utils::TensorArray<3>& xi) const {
        const auto G = geometry().patch(patchIndex);
        const auto U = local_patch_with_tensor(displacement(), patchIndex, displacementTensor);

        const auto gdx = G.template eval<iganet::deriv::dx>(xi);
        const auto gdy = G.template eval<iganet::deriv::dy>(xi);
        const auto gdz = G.template eval<iganet::deriv::dz>(xi);
        const auto udx = U.template eval<iganet::deriv::dx>(xi);
        const auto udy = U.template eval<iganet::deriv::dy>(xi);
        const auto udz = U.template eval<iganet::deriv::dz>(xi);

        const auto J = stack_parametric_jacobian(gdx, gdy, gdz);
        const auto invJ = torch::linalg_inv(J);
        const auto gradUxi = stack_parametric_jacobian(udx, udy, udz);
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
    std::array<double, 4> lastLossParts_{};
    double lambda_{0.0};
    double mu_{0.0};
};

} // namespace iganet_elasticity::multipatch
