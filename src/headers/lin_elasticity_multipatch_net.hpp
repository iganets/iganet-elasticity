#pragma once

#include <iganet.h>
#include <utils/config.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

/// @brief Scatters a per-patch solution into the global coefficient tensor.
///
/// The collocation reference stores its result patch by patch, each in that
/// patch's own control point order. The trainer works on one global vector in
/// which a coefficient shared by several patches appears once. The DOF map
/// holds exactly that correspondence, so this is a lookup and not a coordinate
/// search. A coefficient written twice from two patches gets the same value,
/// because the reference merges those control points as well.
///
/// The global layout is component-major: component g occupies the slots
/// [g * ndofs, (g+1) * ndofs), indexed by global DOF id.
template <typename MultiPatch>
torch::Tensor scatter_patch_values(
    const MultiPatch& space,
    const std::vector<std::vector<std::array<double, 3>>>& perPatch,
    const torch::TensorOptions& options) {
    const auto& map = space.dof_map();
    if (map.empty()) {
        throw std::runtime_error("MultiPatch DOF map has not been built");
    }
    if (perPatch.size() != space.npatches()) {
        throw std::runtime_error(
            "Reference has " + std::to_string(perPatch.size()) + " patches, the model has "
            + std::to_string(space.npatches()));
    }

    const int64_t ndofs = map.ndofs;
    auto host = torch::zeros({3 * ndofs}, torch::TensorOptions().dtype(torch::kFloat64));
    auto view = host.accessor<double, 1>();

    for (std::size_t p = 0; p < space.npatches(); ++p) {
        const auto& ids = map.local_to_global[p];
        if (perPatch[p].size() != ids.size()) {
            throw std::runtime_error(
                "Patch " + std::to_string(p) + ": reference has "
                + std::to_string(perPatch[p].size()) + " control points, the model has "
                + std::to_string(ids.size()));
        }
        for (std::size_t l = 0; l < ids.size(); ++l) {
            for (int g = 0; g < 3; ++g) {
                view[g * ndofs + ids[l]] = perPatch[p][l][g];
            }
        }
    }
    return host.to(options);
}

template <typename Real>
struct MultipatchElasticityConfig {
    using real_t = Real;
    using boundary_value_t = std::tuple<int, real_t, real_t, real_t>;
    using patch_config_t = iganet_elasticity::utils::config::patch_config_3d;

    real_t youngModulus{210.0};
    real_t poissonRatio{0.25};
    /// Optional per-patch material, indexed by patch. An entry that is absent
    /// or empty falls back to the two global values above, so a model without
    /// per-patch data behaves exactly as before.
    std::vector<real_t> patchYoungModulus;
    std::vector<real_t> patchPoissonRatio;
    real_t learningRate{1.0};
    int maxEpoch{100};
    real_t minLoss{1e-6};
    int lbfgsHistorySize{50};
    real_t collocationWeight{1.0};
    bool rowScaling{true};
    /// Only used by the patch-wise evaluation (linearOperator = false). The
    /// linear operator computes the row norms exactly and needs no probes.
    int rowScalingProbes{64};
    /// Evaluates the residual as one sparse matrix-vector product r = A u + b,
    /// assembled once before training, instead of patch by patch in every loss
    /// evaluation. Same loss to round-off, see assemble_linear_operator().
    /// false restores the patch-wise evaluation.
    bool linearOperator{true};
    /// Adds ||u - u_reference||^2 to the loss, weighted by supervisedWeight.
    /// The target comes from the collocation reference, so it only makes sense
    /// once that reference describes the same config - see the examples, which
    /// refresh it before training when this is on.
    bool supervisedLearning{false};
    real_t supervisedWeight{1.0};
    std::vector<int64_t> hiddenLayers{25, 25};
    int degree{2};
    int ncoeffs{3};
    std::array<real_t, 3> bodyForce{0.0, 0.0, 0.0};
    std::vector<boundary_value_t> diriSides;
    std::vector<boundary_value_t> forceSides;
    std::vector<int> tfbcSides;
    std::vector<patch_config_t> patchConfigs;
};

namespace iganet_elasticity::multipatch {

/// @brief y = A x for a constant sparse matrix A, differentiable in x.
///
/// The backward pass is the product with the transpose, which is assembled
/// once as its own CSR matrix. Autograd then records a single node per loss
/// term instead of the thousands of small tensor operations the patch-wise
/// evaluation produces.
struct SparseMatVec : public torch::autograd::Function<SparseMatVec> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 const torch::Tensor& x,
                                 const torch::Tensor& A,
                                 const torch::Tensor& AT) {
        ctx->saved_data["AT"] = AT;
        return A.matmul(x.unsqueeze(1)).squeeze(1);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list gradOutput) {
        const auto AT = ctx->saved_data["AT"].toTensor();
        return {AT.matmul(gradOutput[0].unsqueeze(1)).squeeze(1),
                torch::Tensor(), torch::Tensor()};
    }
};

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
        torch::Tensor rowScale;
    };

    struct BoundaryTractionCache {
        std::size_t patchIndex{0};
        iganet::short_t side{0};
        PreparedPointSet eval;
        torch::Tensor target;
        torch::Tensor J;
        torch::Tensor invJ;
        bool isForce{false};
        torch::Tensor rowScale;
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
        torch::Tensor rowScale;
    };

    struct LossParts {
        torch::Tensor total;
        torch::Tensor collocation;
        torch::Tensor traction;
        torch::Tensor tfbc;
        torch::Tensor interfaceTraction;
        torch::Tensor supervised;
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

        // Lame constants per patch. A chain may mix materials, and every
        // residual is evaluated on one patch, so the constants are looked up by
        // patch index rather than held as single values.
        const std::size_t npatches = this->template input<0>().npatches();
        lambda_.resize(npatches);
        mu_.resize(npatches);
        for (std::size_t p = 0; p < npatches; ++p) {
            const double E = p < cfg_.patchYoungModulus.size()
                                 ? static_cast<double>(cfg_.patchYoungModulus[p])
                                 : static_cast<double>(cfg_.youngModulus);
            const double nu = p < cfg_.patchPoissonRatio.size()
                                  ? static_cast<double>(cfg_.patchPoissonRatio[p])
                                  : static_cast<double>(cfg_.poissonRatio);
            lambda_[p] = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
            mu_[p] = E / (2.0 * (1.0 + nu));
        }

        prepare_caches();
        if (cfg_.linearOperator) {
            assemble_linear_operator();
        } else {
            compute_row_scaling();
        }
    }

    bool epoch(int64_t) override {
        return true;
    }

    torch::Tensor loss(const torch::Tensor& outputs, int64_t epoch) override {
        const auto displacementTensor = constraints_.apply(outputs);
        const auto parts = loss_parts(displacementTensor);
        lastLossParts_ = {
            parts.total.detach().template item<double>(),
            parts.collocation.detach().template item<double>(),
            parts.traction.detach().template item<double>(),
            parts.tfbc.detach().template item<double>(),
            parts.interfaceTraction.detach().template item<double>(),
            parts.supervised.detach().template item<double>()};
        history_.push_back(lastLossParts_[0]);
        // The optimizer evaluates the loss several times per epoch, so the
        // history is longer than the epoch count. Recording the epoch of every
        // entry lets the plot scripts put the curve on a true epoch axis even
        // when min_loss stops training before max_epoch.
        epochHistory_.push_back(epoch);
        std::cout << "epoch " << std::setw(6) << epoch
                  << " | total " << std::setw(14) << lastLossParts_[0]
                  << " | coll " << std::setw(14) << lastLossParts_[1]
                  << " | traction " << std::setw(14) << lastLossParts_[2]
                  << " | tfbc " << std::setw(14) << lastLossParts_[3]
                  << " | interface " << std::setw(14) << lastLossParts_[4];
        if (cfg_.supervisedLearning) {
            std::cout << " | supervised " << std::setw(14) << lastLossParts_[5];
        }
        std::cout << "\n";
        return parts.total;
    }

    void eval() {
        const auto outputs = this->net_->forward(this->inputs(0));
        this->outputs(constraints_.apply(outputs));
    }

    /// @brief Hands over the reference solution to fit against. Expected in the
    /// same layout as the displacement coefficient tensor; the examples build it
    /// with scatter_patch_values().
    void set_supervised_target(torch::Tensor target) {
        if (target.numel() != this->outputs(0).size(0)) {
            throw std::runtime_error(
                "Supervised target has " + std::to_string(target.numel())
                + " entries, the displacement tensor has "
                + std::to_string(this->outputs(0).size(0)));
        }
        supervisedTarget_ = std::move(target);
    }

    const auto& history() const noexcept {
        return history_;
    }

    /// @brief Epoch index of every entry in history(), same length.
    const auto& epoch_history() const noexcept {
        return epochHistory_;
    }

    /// @brief Number of epochs actually run, which is below max_epoch whenever
    /// min_loss stopped the training early.
    int64_t epochs_run() const noexcept {
        return epochHistory_.empty() ? 0 : epochHistory_.back() + 1;
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

    /// @brief True if this side of this patch is glued to a neighbouring
    /// patch rather than being an exterior face.
    bool is_interface_side(std::size_t patchIndex, iganet::short_t side) const {
        for (const auto& interface : geometry().interfaces()) {
            if ((interface.patch1 == patchIndex && interface.side1 == side) ||
                (interface.patch2 == patchIndex && interface.side2 == side)) {
                return true;
            }
        }
        return false;
    }

    
    int bc_priority(std::size_t patchIndex, iganet::short_t side) const {
        
        if (is_interface_side(patchIndex, side)) {
            return 0;
        }
        if (has_patch_configs()) {
            for (const auto& patchCfg : cfg_.patchConfigs) {
                if (static_cast<std::size_t>(patchCfg.patch_id) != patchIndex) {
                    continue;
                }
                for (const auto& entry : patchCfg.diri_sides) {
                    if (entry.side == side) return 3;
                }
                for (const auto& entry : patchCfg.force_sides) {
                    if (entry.side == side) return 2;
                }
                for (const auto other : patchCfg.tfbc_sides) {
                    if (other == side) return 1;
                }
            }
            return 0;
        }
        for (const auto& entry : cfg_.diriSides) {
            if (std::get<0>(entry) == side) return 3;
        }
        for (const auto& entry : cfg_.forceSides) {
            if (std::get<0>(entry) == side) return 2;
        }
        for (const auto other : cfg_.tfbcSides) {
            if (other == side) return 1;
        }
        return 0;
    }

    /// @brief Two faces of a patch share an edge unless they are the same face
    /// or lie opposite each other.
    static bool sides_intersect(iganet::short_t a, iganet::short_t b) {
        return a != b && MultiPatch::interface_type::side_direction(a) !=
                             MultiPatch::interface_type::side_direction(b);
    }

    /// @brief Marks those points of a face that also lie on side `other`.
    torch::Tensor points_on_side(std::size_t patchIndex,
                                 const auto& xi,
                                 iganet::short_t other) const {
        const auto dir = MultiPatch::interface_type::side_direction(other);
        const bool upper = MultiPatch::interface_type::side_parameter(other);
        auto knots = geometry().patch(patchIndex).knots(dir).to(torch::kCPU).contiguous();
        const double value = upper
            ? knots.index({knots.numel() - 1}).template item<double>()
            : knots.index({0}).template item<double>();
        return torch::isclose(xi[dir], torch::full_like(xi[dir], value));
    }

    /// @brief Applies a keep mask to a point set.
    static auto select_points(const auto& xi, const torch::Tensor& keep) {
        auto result = xi;
        const auto idx = torch::nonzero(keep).reshape({-1});
        for (std::size_t d = 0; d < result.size(); ++d) {
            result[d] = xi[d].index_select(0, idx);
        }
        return result;
    }

    /// @brief Physical positions of a parametric point set on one patch.
    torch::Tensor patch_positions(std::size_t patchIndex, const auto& xi) const {
        const auto values = geometry().patch(patchIndex).template eval<iganet::deriv::func>(xi);
        return torch::stack({*values[0], *values[1], *values[2]}, 1);
    }

    /// @brief Priorities returned by bc_priority: a prescribed displacement
    /// outranks a prescribed traction, which outranks a traction-free face.
    /// Zero means the face is an interface and carries no boundary condition.
    static constexpr int kDirichletPriority = 3;

    /// @brief One face that carries a boundary condition, with the physical
    /// positions of its Greville points.
    struct SideClaim {
        std::size_t patch{0};
        iganet::short_t side{0};
        int priority{0};
        torch::Tensor positions;
    };

    /// @brief Every face of the whole model that carries a condition. Needed
    /// because a control point can sit on the outer surface of two different
    /// patches at once; comparing sides within a single patch cannot see that.
    std::vector<SideClaim> collect_side_claims() const {
        std::vector<SideClaim> claims;
        for (std::size_t p = 0; p < geometry().npatches(); ++p) {
            for (iganet::short_t s = 1; s <= 6; ++s) {
                const int priority = bc_priority(p, s);
                if (priority == 0) {
                    continue;
                }
                auto xi = geometry().side_greville(p, s);
                claims.push_back({p, s, priority, patch_positions(p, xi)});
            }
        }
        return claims;
    }

    /// @brief Stiffness of a patch: the P-wave modulus lambda + 2 mu.
    ///
    /// Used to break ties between boundary conditions that meet on an edge.
    /// E alone is not enough - a material with a smaller E but a Poisson ratio
    /// near 0.5 can still be the stiffer one in the normal direction.
    double patch_stiffness(std::size_t patchIndex) const {
        return lambda_[patchIndex] + 2.0 * mu_[patchIndex];
    }

    /// @brief True if `other` outranks (patch, side) at a shared point:
    /// higher priority first, then the STIFFER patch, then the lower patch,
    /// then the lower side.
    ///
    /// Where two patches state a boundary condition at the same point, each
    /// does so with its own material, and only one of them fits. Taking the
    /// stiffer one leaves the smaller residual unenforced, because a softer
    /// material carries less stress at the same strain. Measured on a chain
    /// with a 210/21 material jump, that choice moves the solution by 0.49 %
    /// under refinement, against 2.24 % for the lower-patch rule.
    ///
    /// With one material throughout, the stiffnesses are equal and this falls
    /// straight through to the previous lower-patch ordering.
    bool claim_wins(const SideClaim& other, std::size_t patchIndex,
                    iganet::short_t side, int priority) const {
        if (other.priority != priority) {
            return other.priority > priority;
        }
        if (other.patch != patchIndex) {
            const double theirs = patch_stiffness(other.patch);
            const double ours = patch_stiffness(patchIndex);
            if (theirs != ours) {
                return theirs > ours;
            }
            return other.patch < patchIndex;
        }
        return other.side < side;
    }

    /// @brief Keeps only the points of a boundary face that this side owns.
    /// An edge or corner point is shared by two or three faces whose
    /// conditions contradict each other, so exactly one side may claim it.
    /// Without this, those points put an unsatisfiable term into the loss and
    /// the optimizer trades the interior of the faces away to reduce it.
    auto owned_side_points(std::size_t patchIndex, iganet::short_t side, const auto& xi,
                           const std::vector<SideClaim>& claims) const {
        const auto positions = patch_positions(patchIndex, xi);
        const int priority = bc_priority(patchIndex, side);
        auto keep = torch::ones({positions.size(0)},
                                torch::TensorOptions().dtype(torch::kBool)
                                    .device(positions.device()));
        for (const auto& other : claims) {
            if (other.patch == patchIndex && other.side == side) {
                continue;
            }
            if (!claim_wins(other, patchIndex, side, priority)) {
                continue;
            }
            const auto distance = (positions.unsqueeze(1) -
                                   other.positions.unsqueeze(0)).norm(2, 2);
            const auto shared = std::get<0>(distance.min(1)) < matchingTolerance_;
            keep = torch::logical_and(keep, torch::logical_not(shared));
        }
        return select_points(xi, keep);
    }


    /// @brief Keeps the interface points whose coupling still applies.
    ///
    /// Only a prescribed displacement suppresses the coupling: there the
    /// coefficient is fixed strongly, so a statement about the traction at that
    /// point is meaningless. A prescribed or traction-free face does NOT
    /// suppress it. The interface term is not a competing boundary condition but
    /// the force balance that holds two patches together, and it applies whether
    /// or not the point also sits on a conditioned face.
    ///
    /// Letting it lose to every boundary condition - as this did before - left
    /// the nodes on edges where an interface meets a conditioned face out of
    /// equilibrium. Measured on the bone reference, 128 such nodes carried a
    /// median residual of 6.9 instead of zero, and the solution did not converge
    /// under refinement: going from 1268 to 7908 nodes changed it by 33 %, and
    /// the intermediate mesh lay further from the finest than the coarsest did.
    /// With the coupling kept, refinement changes it by 2.8 %.
    torch::Tensor interface_keep_mask(std::size_t patchIndex, const auto& xi,
                                      const std::vector<SideClaim>& claims) const {
        const auto positions = patch_positions(patchIndex, xi);
        auto keep = torch::ones({positions.size(0)},
                                torch::TensorOptions().dtype(torch::kBool)
                                    .device(positions.device()));
        for (const auto& other : claims) {
            if (other.priority != kDirichletPriority) {
                continue;
            }
            const auto distance = (positions.unsqueeze(1) -
                                   other.positions.unsqueeze(0)).norm(2, 2);
            const auto shared = std::get<0>(distance.min(1)) < matchingTolerance_;
            keep = torch::logical_and(keep, torch::logical_not(shared));
        }
        return keep;
    }

    void prepare_caches() {
        const auto claims = collect_side_claims();
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
                    const auto sideNr = static_cast<iganet::short_t>(side);
                    auto xiOwned = owned_side_points(
                        patchIndex, sideNr, geometry().side_greville(patchIndex, sideNr), claims);
                    if (xiOwned[0].numel() == 0) {
                        continue;
                    }
                    auto xi = to_device(xiOwned, tensorOptions_.device());
                    auto eval = prepare_point_set(patchIndex, xi);
                    auto [J, invJ, hessG] = prepare_geometry_terms(patchIndex, eval);
                    BoundaryTractionCache cache;
                    cache.patchIndex = patchIndex;
                    cache.side = static_cast<iganet::short_t>(side);
                    cache.eval = std::move(eval);
                    cache.target = torch::zeros({1, 3}, tensorOptions_);
                    cache.J = std::move(J);
                    cache.invJ = std::move(invJ);
                    cache.isForce = false;
                    tractionCaches_.push_back(std::move(cache));
                }

                for (const auto& entry : patchCfg.force_sides) {
                    const auto sideNr = static_cast<iganet::short_t>(entry.side);
                    auto xiOwned = owned_side_points(
                        patchIndex, sideNr, geometry().side_greville(patchIndex, sideNr), claims);
                    if (xiOwned[0].numel() == 0) {
                        continue;
                    }
                    auto xi = to_device(xiOwned, tensorOptions_.device());
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
                    cache.isForce = true;
                    tractionCaches_.push_back(std::move(cache));
                }
            }
        } else {
            for (const auto& side : cfg_.tfbcSides) {
                for (const auto& [boundary, xiRaw] : geometry().boundary_greville(side_label(side))) {
                    auto xiOwned = owned_side_points(boundary.patch, boundary.side, xiRaw, claims);
                    if (xiOwned[0].numel() == 0) {
                        continue;
                    }
                    auto xi = to_device(xiOwned, tensorOptions_.device());
                    auto eval = prepare_point_set(boundary.patch, xi);
                    auto [J, invJ, hessG] = prepare_geometry_terms(boundary.patch, eval);
                    BoundaryTractionCache cache;
                    cache.patchIndex = boundary.patch;
                    cache.side = boundary.side;
                    cache.eval = std::move(eval);
                    cache.target = torch::zeros({1, 3}, tensorOptions_);
                    cache.J = std::move(J);
                    cache.invJ = std::move(invJ);
                    cache.isForce = false;
                    tractionCaches_.push_back(std::move(cache));
                }
            }

            for (const auto& entry : cfg_.forceSides) {
                const int side = std::get<0>(entry);
                for (const auto& [boundary, xiRaw] : geometry().boundary_greville(side_label(side))) {
                    auto xiOwned = owned_side_points(boundary.patch, boundary.side, xiRaw, claims);
                    if (xiOwned[0].numel() == 0) {
                        continue;
                    }
                    auto xi = to_device(xiOwned, tensorOptions_.device());
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
                    cache.isForce = true;
                    tractionCaches_.push_back(std::move(cache));
                }
            }
        }

        interfaceCaches_.reserve(geometry().ninterfaces());
        for (const auto& interface : geometry().interfaces()) {
            auto [xi1, xi2] = geometry().interface_greville(interface);
            
            const auto keep = torch::logical_and(
                interface_keep_mask(interface.patch1, xi1, claims),
                interface_keep_mask(interface.patch2, xi2, claims));
            xi1 = select_points(xi1, keep);
            xi2 = select_points(xi2, keep);
            if (xi1[0].numel() == 0) {
                continue;
            }
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

        const double mu = mu_[patchIndex];
        const double lambda = lambda_[patchIndex];

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
            (lambda + 2.0 * mu) * ux_xx + mu * ux_yy + mu * ux_zz +
                (lambda + mu) * (uy_xy + uz_xz),
            mu * uy_xx + (lambda + 2.0 * mu) * uy_yy + mu * uy_zz +
                (lambda + mu) * (ux_yx + uz_yz),
            mu * uz_xx + mu * uz_yy + (lambda + 2.0 * mu) * uz_zz +
                (lambda + mu) * (ux_zx + uy_zy)}, 1);

        return divStress + body.repeat({divStress.size(0), 1});
    }

    /// @brief Scales every residual row to unit sensitivity.
    ///
    /// The four loss terms carry different physical units. The equilibrium
    /// residual scales with second derivatives of the basis (order mu/h^2),
    /// the traction and interface residuals with first derivatives (order
    /// mu/h). With element sizes spanning a factor of 90 across this geometry
    /// the rows of the underlying operator differ by a factor of about 1000,
    /// and summing them unweighted is what makes the squared loss so badly
    /// conditioned that LBFGS cannot reach the minimum in any practical number
    /// of epochs.
    ///
    /// Each row is therefore divided by the norm of its own coefficient row.
    /// That norm depends on the geometry alone, so it is computed once here and
    /// reused in every epoch. It is estimated with random +-1 probes: the
    /// residual is linear in the coefficients, so r(v) - r(0) is the operator
    /// applied to v, and the mean of its square over Rademacher probes equals
    /// the squared row norm in expectation. Twenty probes already recover the
    /// full conditioning gain; the default of 64 leaves margin.
    ///
    /// The scales are normalised to geometric mean one so the reported loss
    /// keeps its former magnitude and min_loss stays comparable. The system is
    /// consistent - the exact solution has zero residual in every row - so the
    /// reweighting cannot move the minimum, only shorten the path to it.
    void compute_row_scaling() {
        for (auto& cache : patchResidualCaches_)
            cache.rowScale = torch::ones({cache.eval.numeval, 3}, tensorOptions_);
        for (auto& cache : tractionCaches_)
            cache.rowScale = torch::ones({cache.eval.numeval, 3}, tensorOptions_);
        for (auto& cache : interfaceCaches_)
            cache.rowScale = torch::ones({cache.eval1.numeval, 3}, tensorOptions_);

        if (!cfg_.rowScaling || cfg_.rowScalingProbes < 1) {
            return;
        }

        torch::NoGradGuard noGrad;
        const int64_t ndofs = this->outputs(0).size(0);
        const auto base = constraints_.apply(torch::zeros({ndofs}, tensorOptions_));

        // Constant part of each residual, subtracted to isolate the operator.
        std::vector<torch::Tensor> collBase, tracBase, ifaceBase;
        std::vector<torch::Tensor> collAcc, tracAcc, ifaceAcc;
        for (const auto& cache : patchResidualCaches_) {
            collBase.push_back(strong_form_residual(
                cache.patchIndex, base, cache.eval, cache.invJ, cache.hessG, cache.body));
            collAcc.push_back(torch::zeros_like(collBase.back()));
        }
        for (const auto& cache : tractionCaches_) {
            tracBase.push_back(traction_on_boundary(
                cache.patchIndex, cache.side, base, cache.eval, cache.J, cache.invJ));
            tracAcc.push_back(torch::zeros_like(tracBase.back()));
        }
        for (const auto& cache : interfaceCaches_) {
            ifaceBase.push_back(
                traction_on_boundary(cache.patch1, cache.side1, base, cache.eval1,
                                     cache.J1, cache.invJ1) +
                traction_on_boundary(cache.patch2, cache.side2, base, cache.eval2,
                                     cache.J2, cache.invJ2));
            ifaceAcc.push_back(torch::zeros_like(ifaceBase.back()));
        }

        for (int probe = 0; probe < cfg_.rowScalingProbes; ++probe) {
            const auto v = constraints_.apply(
                2.0 * torch::randint(0, 2, {ndofs}, tensorOptions_) - 1.0);

            for (std::size_t i = 0; i < patchResidualCaches_.size(); ++i) {
                const auto& cache = patchResidualCaches_[i];
                if (cache.eval.numeval == 0) continue;
                const auto d = strong_form_residual(cache.patchIndex, v, cache.eval,
                                                    cache.invJ, cache.hessG, cache.body) -
                               collBase[i];
                collAcc[i] = collAcc[i] + d * d;
            }
            for (std::size_t i = 0; i < tractionCaches_.size(); ++i) {
                const auto& cache = tractionCaches_[i];
                if (cache.eval.numeval == 0) continue;
                const auto d = traction_on_boundary(cache.patchIndex, cache.side, v,
                                                    cache.eval, cache.J, cache.invJ) -
                               tracBase[i];
                tracAcc[i] = tracAcc[i] + d * d;
            }
            for (std::size_t i = 0; i < interfaceCaches_.size(); ++i) {
                const auto& cache = interfaceCaches_[i];
                if (cache.eval1.numeval == 0 || cache.eval2.numeval == 0) continue;
                const auto d = traction_on_boundary(cache.patch1, cache.side1, v,
                                                    cache.eval1, cache.J1, cache.invJ1) +
                               traction_on_boundary(cache.patch2, cache.side2, v,
                                                    cache.eval2, cache.J2, cache.invJ2) -
                               ifaceBase[i];
                ifaceAcc[i] = ifaceAcc[i] + d * d;
            }
        }

        // Row norms, then scales normalised to geometric mean one.
        const double inv = 1.0 / static_cast<double>(cfg_.rowScalingProbes);
        std::vector<torch::Tensor> norms;
        for (auto& acc : collAcc)  norms.push_back((acc * inv).sqrt().clamp_min(1e-30));
        for (auto& acc : tracAcc)  norms.push_back((acc * inv).sqrt().clamp_min(1e-30));
        for (auto& acc : ifaceAcc) norms.push_back((acc * inv).sqrt().clamp_min(1e-30));

        double logSum = 0.0;
        int64_t count = 0;
        for (const auto& n : norms) {
            if (n.numel() == 0) continue;
            logSum += n.log().sum().template item<double>();
            count += n.numel();
        }
        const double geoMean = count > 0 ? std::exp(logSum / static_cast<double>(count)) : 1.0;

        std::size_t k = 0;
        for (auto& cache : patchResidualCaches_) cache.rowScale = geoMean / norms[k++];
        for (auto& cache : tractionCaches_)      cache.rowScale = geoMean / norms[k++];
        for (auto& cache : interfaceCaches_)     cache.rowScale = geoMean / norms[k++];

        double lo = std::numeric_limits<double>::max(), hi = 0.0;
        for (const auto& n : norms) {
            if (n.numel() == 0) continue;
            lo = std::min(lo, n.min().template item<double>());
            hi = std::max(hi, n.max().template item<double>());
        }
        std::cout << "row scaling: " << count << " residual rows, row norms "
                  << lo << " ... " << hi << " (spread factor " << hi / lo
                  << "), " << cfg_.rowScalingProbes << " probes\n";
    }

    LossParts loss_parts(const torch::Tensor& displacementTensor) const {
        if (cfg_.linearOperator && linearReady_) {
            return loss_parts_linear(displacementTensor);
        }
        return loss_parts_patchwise(displacementTensor);
    }

    /// @brief Supervised term: fit the reference solution directly. Unlike the
    /// physics terms it weights every direction equally, which is why it can
    /// reach the soft modes the residual is nearly blind to. It is ADDED to
    /// the physics loss, not a replacement for it.
    torch::Tensor supervised_loss(const torch::Tensor& displacementTensor) const {
        if (cfg_.supervisedLearning && supervisedTarget_.defined()) {
            return cfg_.supervisedWeight *
                   torch::mse_loss(displacementTensor, supervisedTarget_);
        }
        return torch::zeros({}, tensorOptions_);
    }

    /// @brief The loss evaluated patch by patch, point set by point set. Used
    /// when linearOperator is off, and once at start-up to verify the linear
    /// operator against it.
    LossParts loss_parts_patchwise(const torch::Tensor& displacementTensor) const {
        auto collocationLoss = torch::zeros({}, tensorOptions_);
        auto tractionLoss = torch::zeros({}, tensorOptions_);
        auto tfbcLoss = torch::zeros({}, tensorOptions_);
        auto interfaceLoss = torch::zeros({}, tensorOptions_);

        for (const auto& cache : patchResidualCaches_) {
            if (cache.eval.numeval == 0) {
                continue;
            }
            const auto residual = cache.rowScale * strong_form_residual(
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
            const auto residual =
                cache.rowScale * (traction - cache.target.repeat({traction.size(0), 1}));
            const auto mse = torch::mse_loss(residual, torch::zeros_like(residual));
            if (cache.isForce) {
                tractionLoss = tractionLoss + mse;
            } else {
                tfbcLoss = tfbcLoss + mse;
            }
        }

        for (const auto& cache : interfaceCaches_) {
            if (cache.eval1.numeval == 0 || cache.eval2.numeval == 0) {
                continue;
            }
            const auto t1 = traction_on_boundary(
                cache.patch1, cache.side1, displacementTensor, cache.eval1, cache.J1, cache.invJ1);
            const auto t2 = traction_on_boundary(
                cache.patch2, cache.side2, displacementTensor, cache.eval2, cache.J2, cache.invJ2);
            const auto residual = cache.rowScale * (t1 + t2);
            interfaceLoss = interfaceLoss +
                            torch::mse_loss(residual, torch::zeros_like(residual));
        }

        const auto supervisedLoss = supervised_loss(displacementTensor);

        return {
            cfg_.collocationWeight * collocationLoss + tractionLoss + tfbcLoss +
                interfaceLoss + supervisedLoss,
            collocationLoss,
            tractionLoss,
            tfbcLoss,
            interfaceLoss,
            supervisedLoss};
    }

    // -------------------------------------------------------------------------
    // Linear operator
    //
    // Every residual of the loss - equilibrium, prescribed and zero traction,
    // interface balance - is linear in the global displacement coefficients u:
    //
    //     r = A u + b,
    //
    // with A depending on the geometry, the basis and the material only, and b
    // holding the body force and the prescribed tractions. The patch-wise
    // evaluation rebuilds A u from scratch in every loss evaluation: it cuts the
    // patch coefficients out of u, evaluates three first and six second
    // derivatives at every point set and maps them to physical space - several
    // thousand small tensor operations whose backward pass dominated the run
    // time (about 50 s per loss evaluation on the bone with implant).
    //
    // Here A is assembled once, row-scaled, as a sparse matrix per loss term.
    // A loss evaluation is then one sparse product per term, and its gradient
    // one product with the transpose (SparseMatVec).
    //
    // Assembly. A prepared point set stores, per point q and per basis function
    // k with support there, the parametric basis derivatives B_d[k,q] and
    // B_de[k,q] and the local coefficient index. With the inverse Jacobian
    // invJ[q,d,j] = dxi_d/dx_j and the geometry Hessians hessG[l][q,d,e] =
    // d2x_l/dxi_d dxi_e the physical derivatives of basis function k are
    //
    //     N_j[k,q]  = sum_d B_d[k,q] invJ[q,d,j]
    //     M_ij[k,q] = sum_de invJ[q,d,i] (B_de[k,q] - sum_l N_l[k,q] hessG[l][q,d,e]) invJ[q,e,j]
    //
    // exactly the transformation strong_form_residual() applies to u. The
    // equilibrium row i and the traction row i then carry, for displacement
    // component c,
    //
    //     div sigma:  mu delta_ic sum_j M_jj + (lambda + mu) M_ic
    //     sigma n:    mu delta_ic (N . n) + mu N_i n_c + lambda n_i N_c
    //
    // in column c * ndofs + (global id of k). An interface row adds the
    // traction rows of both patches; columns shared by them are summed.
    //
    // Row scaling. The patch-wise path estimates the norm of every row of A
    // (restricted to the free coefficients) from 64 random probes. With A at
    // hand the norm is computed exactly; scales are normalised to geometric
    // mean one as before.
    //
    // Check. After assembly the loss is evaluated both ways for a random u,
    // with the same row scales. The run stops if they differ by more than
    // 1e-6 relative; the deviation is printed.
    // -------------------------------------------------------------------------

    /// @brief Rows of one point set in COO form: row = 3 * point + component.
    struct LinearRows {
        torch::Tensor rows;
        torch::Tensor cols;
        torch::Tensor vals;
        torch::Tensor b;
        int64_t nrows{0};
    };

    /// @brief Physical first (N, 3 x nb x n) and, if hessG is given, second
    /// (M, 3 x 3 x nb x n) derivatives of the basis functions at a point set,
    /// with the global DOF id of every basis function (cols, nb x n).
    struct PhysicalBasis {
        torch::Tensor cols;
        torch::Tensor N;
        torch::Tensor M;
    };

    PhysicalBasis physical_basis(std::size_t patchIndex, const PreparedPointSet& ev,
                                 const torch::Tensor& invJ,
                                 const std::array<torch::Tensor, 3>* hessG) const {
        const int64_t n = ev.numeval;
        const auto local = ev.coeff_indices.flatten().to(torch::kInt64).view({-1, n});
        const auto& ids = displacement().dof_map().local_to_global[patchIndex];
        const auto globalIds =
            torch::tensor(ids, torch::TensorOptions().dtype(torch::kInt64))
                .to(tensorOptions_.device());

        PhysicalBasis result;
        result.cols = globalIds.index_select(0, local.flatten()).view_as(local);

        const auto basis = [&](iganet::deriv d) {
            return ev.basfunc(d).to(tensorOptions_).reshape({-1, n});
        };
        const auto Bd = torch::stack({basis(iganet::deriv::dx), basis(iganet::deriv::dy),
                                      basis(iganet::deriv::dz)});
        result.N = torch::einsum("dkq,qdj->jkq", {Bd, invJ});

        if (hessG != nullptr) {
            const auto bxx = basis(iganet::deriv::dx ^ 2);
            const auto bxy = basis(iganet::deriv::dx + iganet::deriv::dy);
            const auto bxz = basis(iganet::deriv::dx + iganet::deriv::dz);
            const auto byy = basis(iganet::deriv::dy ^ 2);
            const auto byz = basis(iganet::deriv::dy + iganet::deriv::dz);
            const auto bzz = basis(iganet::deriv::dz ^ 2);
            const auto Bde = torch::stack({torch::stack({bxx, bxy, bxz}),
                                           torch::stack({bxy, byy, byz}),
                                           torch::stack({bxz, byz, bzz})});
            const auto H = torch::stack({(*hessG)[0], (*hessG)[1], (*hessG)[2]});
            const auto corrected = Bde - torch::einsum("lkq,lqde->dekq", {result.N, H});
            result.M = torch::einsum("qdi,dekq,qej->ijkq", {invJ, corrected, invJ});
        }
        return result;
    }

    /// @brief Appends the entries of row block i, displacement component c.
    void append_entries(LinearRows& out, const PhysicalBasis& basis, int i, int c,
                        const torch::Tensor& vals, int64_t nscalar) const {
        const int64_t n = vals.size(1);
        const auto rows = (3 * torch::arange(n, basis.cols.options()) + i)
                              .view({1, n}).expand_as(basis.cols);
        out.rows = torch::cat({out.rows, rows.flatten()});
        out.cols = torch::cat({out.cols, (basis.cols + c * nscalar).flatten()});
        out.vals = torch::cat({out.vals, vals.flatten()});
    }

    LinearRows empty_rows(int64_t n) const {
        const auto idx = torch::TensorOptions().dtype(torch::kInt64).device(tensorOptions_.device());
        return {torch::empty({0}, idx), torch::empty({0}, idx),
                torch::empty({0}, tensorOptions_), torch::zeros({3 * n}, tensorOptions_), 3 * n};
    }

    /// @brief Rows of div(sigma) + f at one patch's interior points.
    LinearRows equilibrium_rows(const PatchResidualCache& cache, int64_t nscalar) const {
        const int64_t n = cache.eval.numeval;
        auto out = empty_rows(n);
        const auto basis = physical_basis(cache.patchIndex, cache.eval, cache.invJ, &cache.hessG);
        const double mu = mu_[cache.patchIndex];
        const double lambda = lambda_[cache.patchIndex];
        const auto& M = basis.M;
        const auto laplace = M[0][0] + M[1][1] + M[2][2];
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < 3; ++c) {
                auto vals = (lambda + mu) * M[i][c];
                if (i == c) {
                    vals = vals + mu * laplace;
                }
                append_entries(out, basis, i, c, vals, nscalar);
            }
        }
        out.b = cache.body.repeat({n, 1}).flatten();
        return out;
    }

    /// @brief Rows of sigma(u) n on one side, appended to `out` (no b).
    void traction_rows(LinearRows& out, std::size_t patchIndex, iganet::short_t side,
                       const PreparedPointSet& ev, const torch::Tensor& invJ,
                       int64_t nscalar) const {
        const auto basis = physical_basis(patchIndex, ev, invJ, nullptr);
        const double mu = mu_[patchIndex];
        const double lambda = lambda_[patchIndex];

        // Same outward normal as traction_on_boundary().
        const auto fixed = static_cast<iganet::short_t>((side - 1) / 2);
        auto normal = invJ.index({torch::indexing::Slice(), fixed, torch::indexing::Slice()});
        if (!MultiPatch::interface_type::side_parameter(side)) {
            normal = -normal;
        }
        normal = normal / normal.norm(2, 1).clamp_min(1e-12).view({-1, 1});
        const auto nT = normal.transpose(0, 1).unsqueeze(1);      // (3, 1, n)
        const auto& N = basis.N;                                   // (3, nb, n)
        const auto Ndotn = (N * nT).sum(0);                        // (nb, n)

        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < 3; ++c) {
                auto vals = mu * N[i] * nT[c] + lambda * nT[i] * N[c];
                if (i == c) {
                    vals = vals + mu * Ndotn;
                }
                append_entries(out, basis, i, c, vals, nscalar);
            }
        }
    }

    /// @brief Builds A and b per loss term, the exact row scaling, and checks
    /// the result against the patch-wise evaluation.
    void assemble_linear_operator() {
        torch::NoGradGuard noGrad;
        const auto start = std::chrono::steady_clock::now();
        const int64_t ntotal = this->outputs(0).size(0);
        const int64_t nscalar = ntotal / 3;

        // Rows per point set, in the order of the caches.
        std::vector<LinearRows> collRows, tracRows, ifaceRows;
        for (const auto& cache : patchResidualCaches_) {
            collRows.push_back(cache.eval.numeval == 0 ? empty_rows(0)
                                                       : equilibrium_rows(cache, nscalar));
        }
        for (const auto& cache : tractionCaches_) {
            auto out = empty_rows(cache.eval.numeval);
            if (cache.eval.numeval > 0) {
                traction_rows(out, cache.patchIndex, cache.side, cache.eval, cache.invJ, nscalar);
                out.b = -cache.target.repeat({cache.eval.numeval, 1}).flatten();
            }
            tracRows.push_back(std::move(out));
        }
        for (const auto& cache : interfaceCaches_) {
            const bool active = cache.eval1.numeval > 0 && cache.eval2.numeval > 0;
            auto out = empty_rows(cache.eval1.numeval);
            if (active) {
                traction_rows(out, cache.patch1, cache.side1, cache.eval1, cache.invJ1, nscalar);
                traction_rows(out, cache.patch2, cache.side2, cache.eval2, cache.invJ2, nscalar);
            }
            ifaceRows.push_back(std::move(out));
        }

        // Merge duplicate entries (interface columns shared by both patches).
        const auto coalesce = [&](LinearRows& r) {
            if (r.vals.numel() == 0) return;
            auto coo = torch::sparse_coo_tensor(torch::stack({r.rows, r.cols}), r.vals,
                                                {r.nrows, ntotal})
                           .coalesce();
            r.rows = coo.indices()[0];
            r.cols = coo.indices()[1];
            r.vals = coo.values();
        };
        for (auto* group : {&collRows, &tracRows, &ifaceRows})
            for (auto& r : *group) coalesce(r);

        // Exact row norms over the free coefficients, as the probes estimated.
        const auto freeMask =
            (constraints_.apply(torch::ones({ntotal}, tensorOptions_)) -
             constraints_.apply(torch::zeros({ntotal}, tensorOptions_))) > 0.5;
        const auto rowNorm = [&](const LinearRows& r) {
            auto sq = torch::zeros({r.nrows}, tensorOptions_);
            if (r.vals.numel() > 0) {
                const auto keep = freeMask.index_select(0, r.cols).to(tensorOptions_.dtype());
                sq.index_add_(0, r.rows, r.vals * r.vals * keep);
            }
            return sq.sqrt().clamp_min(1e-30);
        };

        std::vector<torch::Tensor> collNorm, tracNorm, ifaceNorm;
        for (const auto& r : collRows) collNorm.push_back(rowNorm(r));
        for (const auto& r : tracRows) tracNorm.push_back(rowNorm(r));
        for (const auto& r : ifaceRows) ifaceNorm.push_back(rowNorm(r));

        // Scales normalised to geometric mean one, stored in the caches as well
        // so that the patch-wise evaluation uses identical weights.
        double logSum = 0.0, lo = std::numeric_limits<double>::max(), hi = 0.0;
        int64_t count = 0;
        for (const auto* group : {&collNorm, &tracNorm, &ifaceNorm}) {
            for (const auto& norm : *group) {
                if (norm.numel() == 0) continue;
                logSum += norm.log().sum().template item<double>();
                lo = std::min(lo, norm.min().template item<double>());
                hi = std::max(hi, norm.max().template item<double>());
                count += norm.numel();
            }
        }
        const double geoMean = count > 0 ? std::exp(logSum / static_cast<double>(count)) : 1.0;
        const auto scaleOf = [&](const torch::Tensor& norm) {
            return cfg_.rowScaling ? geoMean / norm : torch::ones_like(norm);
        };
        for (std::size_t k = 0; k < patchResidualCaches_.size(); ++k)
            patchResidualCaches_[k].rowScale = scaleOf(collNorm[k]).view({-1, 3});
        for (std::size_t k = 0; k < tractionCaches_.size(); ++k)
            tractionCaches_[k].rowScale = scaleOf(tracNorm[k]).view({-1, 3});
        for (std::size_t k = 0; k < interfaceCaches_.size(); ++k)
            interfaceCaches_[k].rowScale = scaleOf(ifaceNorm[k]).view({-1, 3});

        // One sparse matrix per loss term. mse_loss averages over the 3n rows of
        // each point set, which the weight w = 1 / (3n) per row reproduces.
        const auto build = [&](const std::vector<const LinearRows*>& parts,
                               const std::vector<torch::Tensor>& scales) {
            LinearTerm term;
            std::vector<torch::Tensor> rows, cols, vals, b, w;
            int64_t offset = 0;
            for (std::size_t k = 0; k < parts.size(); ++k) {
                const auto& r = *parts[k];
                if (r.nrows == 0 || r.vals.numel() == 0) continue;
                const auto s = scales[k].flatten();
                rows.push_back(r.rows + offset);
                cols.push_back(r.cols);
                vals.push_back(r.vals * s.index_select(0, r.rows));
                b.push_back(r.b * s);
                w.push_back(torch::full({r.nrows}, 1.0 / static_cast<double>(r.nrows),
                                        tensorOptions_));
                offset += r.nrows;
            }
            if (offset == 0) return term;
            const auto idx = torch::stack({torch::cat(rows), torch::cat(cols)});
            const auto v = torch::cat(vals);
            term.A = torch::sparse_coo_tensor(idx, v, {offset, ntotal}).coalesce().to_sparse_csr();
            term.AT = torch::sparse_coo_tensor(idx.flip(0), v, {ntotal, offset})
                          .coalesce().to_sparse_csr();
            term.b = torch::cat(b);
            term.w = torch::cat(w);
            term.nnz = v.numel();
            return term;
        };

        std::vector<const LinearRows*> forceParts, tfbcParts;
        std::vector<torch::Tensor> forceScales, tfbcScales;
        for (std::size_t k = 0; k < tractionCaches_.size(); ++k) {
            auto& parts = tractionCaches_[k].isForce ? forceParts : tfbcParts;
            auto& scales = tractionCaches_[k].isForce ? forceScales : tfbcScales;
            parts.push_back(&tracRows[k]);
            scales.push_back(tractionCaches_[k].rowScale);
        }
        std::vector<const LinearRows*> collParts, ifaceParts;
        std::vector<torch::Tensor> collScales, ifaceScales;
        for (std::size_t k = 0; k < collRows.size(); ++k) {
            collParts.push_back(&collRows[k]);
            collScales.push_back(patchResidualCaches_[k].rowScale);
        }
        for (std::size_t k = 0; k < ifaceRows.size(); ++k) {
            ifaceParts.push_back(&ifaceRows[k]);
            ifaceScales.push_back(interfaceCaches_[k].rowScale);
        }
        collocationTerm_ = build(collParts, collScales);
        tractionTerm_ = build(forceParts, forceScales);
        tfbcTerm_ = build(tfbcParts, tfbcScales);
        interfaceTerm_ = build(ifaceParts, ifaceScales);
        linearReady_ = true;

        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "linear operator: " << count << " residual rows, "
                  << collocationTerm_.nnz + tractionTerm_.nnz + tfbcTerm_.nnz +
                         interfaceTerm_.nnz
                  << " nonzeros, assembled in " << std::fixed << std::setprecision(1)
                  << seconds << " s" << std::defaultfloat << std::setprecision(6) << "\n";
        if (cfg_.rowScaling) {
            std::cout << "row scaling: exact row norms " << lo << " ... " << hi
                      << " (spread factor " << hi / lo << ")\n";
        }
        check_linear_operator();
    }

    /// @brief Compares the linear operator with the patch-wise evaluation for a
    /// random displacement and stops if the two disagree.
    void check_linear_operator() const {
        torch::NoGradGuard noGrad;
        const int64_t ntotal = this->outputs(0).size(0);
        const auto u = constraints_.apply(torch::randn({ntotal}, tensorOptions_));
        const auto linear = loss_parts_linear(u);
        const auto patchwise = loss_parts_patchwise(u);
        double worst = 0.0;
        for (const auto& [a, b] : {std::pair{linear.collocation, patchwise.collocation},
                                   std::pair{linear.traction, patchwise.traction},
                                   std::pair{linear.tfbc, patchwise.tfbc},
                                   std::pair{linear.interfaceTraction,
                                             patchwise.interfaceTraction}}) {
            const double x = a.template item<double>();
            const double y = b.template item<double>();
            worst = std::max(worst, std::abs(x - y) / std::max(std::abs(y), 1e-300));
        }
        std::cout << "linear operator check against patch-wise evaluation: "
                     "max. relative deviation "
                  << worst << "\n";
        if (!(worst < 1e-6)) {
            throw std::runtime_error(
                "Linear operator disagrees with the patch-wise loss evaluation "
                "(relative deviation " + std::to_string(worst) + "). Set "
                "multipatch.linear_operator to false to train without it.");
        }
    }

    /// @brief One loss term as weighted squared residual of r = A u + b.
    torch::Tensor linear_term_loss(const torch::Tensor& displacementTensor,
                                   const auto& term) const {
        if (!term.A.defined()) {
            return torch::zeros({}, tensorOptions_);
        }
        const auto r = SparseMatVec::apply(displacementTensor, term.A, term.AT) + term.b;
        return (term.w * r * r).sum();
    }

    LossParts loss_parts_linear(const torch::Tensor& displacementTensor) const {
        const auto collocationLoss = linear_term_loss(displacementTensor, collocationTerm_);
        const auto tractionLoss = linear_term_loss(displacementTensor, tractionTerm_);
        const auto tfbcLoss = linear_term_loss(displacementTensor, tfbcTerm_);
        const auto interfaceLoss = linear_term_loss(displacementTensor, interfaceTerm_);
        const auto supervisedLoss = supervised_loss(displacementTensor);
        return {
            cfg_.collocationWeight * collocationLoss + tractionLoss + tfbcLoss +
                interfaceLoss + supervisedLoss,
            collocationLoss,
            tractionLoss,
            tfbcLoss,
            interfaceLoss,
            supervisedLoss};
    }

    torch::Tensor traction_on_boundary(
        std::size_t patchIndex,
        iganet::short_t side,
        const torch::Tensor& displacementTensor,
        const PreparedPointSet& cache,
        const torch::Tensor& /*J*/,
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

        // Each side of an interface contributes with ITS OWN material, which is
        // exactly traction continuity across a material jump.
        auto stress = 2.0 * mu_[patchIndex] * strain;
        for (iganet::short_t c = 0; c < 3; ++c) {
            stress.index_put_(
                {torch::indexing::Slice(), c, c},
                stress.index({torch::indexing::Slice(), c, c}) + lambda_[patchIndex] * trace);
        }

        const auto fixed = static_cast<iganet::short_t>((side - 1) / 2);
        
        // Outward normal as the gradient of the fixed parametric coordinate,
        // i.e. row `fixed` of J^-1. The cross product of the two in-face
        // tangents used before points inward on left-handed patches (11 of the
        // 16 bone patches are parametrised left-handed), which reverses
        // prescribed tractions and silently turns the interface coupling into a
        // zero-traction condition wherever two neighbours differ in handedness.
        auto normal = invJ.index({torch::indexing::Slice(), fixed, torch::indexing::Slice()});
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
    std::vector<int64_t> epochHistory_;
    std::vector<PatchResidualCache> patchResidualCaches_;
    std::vector<BoundaryTractionCache> tractionCaches_;
    std::vector<InterfaceCache> interfaceCaches_;
    std::array<double, 6> lastLossParts_{};
    torch::Tensor supervisedTarget_;
    double matchingTolerance_{5e-5};   // see MATCHING_TOLERANCE in the examples
    std::vector<double> lambda_;
    std::vector<double> mu_;

    /// @brief Row-scaled r = A u + b of one loss term, see
    /// assemble_linear_operator(). w averages each point set over its rows.
    struct LinearTerm {
        torch::Tensor A;
        torch::Tensor AT;
        torch::Tensor b;
        torch::Tensor w;
        int64_t nnz{0};
    };
    LinearTerm collocationTerm_;
    LinearTerm tractionTerm_;
    LinearTerm tfbcTerm_;
    LinearTerm interfaceTerm_;
    bool linearReady_{false};
};

} // namespace iganet_elasticity::multipatch
