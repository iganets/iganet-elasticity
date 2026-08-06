/*
 * Example: 3D multi-patch elasticity from an XML spline geometry.
 *
 * This example focuses on loading an existing spline geometry, constructing a
 * multi-patch description from it, and then running a collocation-based
 * elasticity solve across all patches. It is a useful smoke/integration case
 * for complex geometries such as the bone example.
 */

#include "headers/lin_elasticity_utils.hpp"

#include <iganet.h>

#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
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

std::string resolveBoundaryLabel(const nlohmann::json& j) {
    if (j.contains("geometry") && j["geometry"].contains("boundary_label")) {
        return j["geometry"]["boundary_label"].get<std::string>();
    }
    return "BID1";
}

int resolveMultiPatchId(const nlohmann::json& j) {
    if (j.contains("geometry") && j["geometry"].contains("multipatch_id")) {
        return j["geometry"]["multipatch_id"].get<int>();
    }
    return 0;
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

struct MultipatchConfig {
    double youngModulus = 210.0;
    double poissonRatio = 0.25;
    int maxEpoch = 20;
    double minLoss = 1e-6;
    double learningRate = 1e-2;
    double dirichletWeight = 1e5;
    std::array<double, 3> bodyForce{0.0, 0.0, -100.0};
    std::vector<std::tuple<int, double, double, double>> diriSides;
    std::vector<std::tuple<int, double, double, double>> forceSides;
};

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
        if (mj.contains("dirichlet_weight")) {
            cfg.dirichletWeight = mj["dirichlet_weight"].get<double>();
        }
        if (mj.contains("boundary_conditions")) {
            const auto& bc = mj["boundary_conditions"];
            if (bc.contains("diri_sides")) {
                cfg.diriSides.clear();
                for (const auto& ds : bc["diri_sides"]) {
                    cfg.diriSides.emplace_back(ds[0].get<int>(), ds[1].get<double>(),
                                               ds[2].get<double>(), ds[3].get<double>());
                }
            }
            if (bc.contains("force_sides")) {
                cfg.forceSides.clear();
                for (const auto& fs : bc["force_sides"]) {
                    cfg.forceSides.emplace_back(fs[0].get<int>(), fs[1].get<double>(),
                                                fs[2].get<double>(), fs[3].get<double>());
                }
            }
        }
    }

    if (j.contains("body_force")) {
        const auto& bf = require(j, "body_force");
        cfg.bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};
    }

    if (j.contains("multipatch") && j["multipatch"].contains("body_force")) {
        const auto& bf = j["multipatch"]["body_force"];
        cfg.bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};
    }

    if (j.contains("boundary_conditions")) {
        for (const auto& ds : require(j, "boundary_conditions.diri_sides")) {
            cfg.diriSides.emplace_back(ds[0].get<int>(), ds[1].get<double>(),
                                       ds[2].get<double>(), ds[3].get<double>());
        }
        for (const auto& fs : require(j, "boundary_conditions.force_sides")) {
            cfg.forceSides.emplace_back(fs[0].get<int>(), fs[1].get<double>(),
                                        fs[2].get<double>(), fs[3].get<double>());
        }
    }

    if (j.contains("multipatch") && j["multipatch"].contains("boundary_conditions")) {
        const auto& bc = j["multipatch"]["boundary_conditions"];
        if (bc.contains("diri_sides")) {
            cfg.diriSides.clear();
            for (const auto& ds : bc["diri_sides"]) {
                cfg.diriSides.emplace_back(ds[0].get<int>(), ds[1].get<double>(),
                                           ds[2].get<double>(), ds[3].get<double>());
            }
        }
        if (bc.contains("force_sides")) {
            cfg.forceSides.clear();
            for (const auto& fs : bc["force_sides"]) {
                cfg.forceSides.emplace_back(fs[0].get<int>(), fs[1].get<double>(),
                                            fs[2].get<double>(), fs[3].get<double>());
            }
        }
    }

    return cfg;
}

std::optional<std::array<double, 3>>
boundaryTarget(int side, const std::vector<std::tuple<int, double, double, double>>& values) {
    for (const auto& entry : values) {
        if (std::get<0>(entry) == side) {
            return std::array<double, 3>{
                std::get<1>(entry), std::get<2>(entry), std::get<3>(entry)};
        }
    }
    return std::nullopt;
}

template <typename BoundaryPointSets>
std::size_t countBoundaryMatches(
    const BoundaryPointSets& pointSets,
    const std::vector<std::tuple<int, double, double, double>>& values) {
    return static_cast<std::size_t>(std::count_if(pointSets.begin(), pointSets.end(),
        [&](const auto& entry) {
            return boundaryTarget(entry.first.side, values).has_value();
        }));
}

template <typename Eval>
torch::Tensor stackVector(const Eval& value) {
    return torch::stack({*value[0], *value[1], *value[2]}, 1);
}

template <typename EvalXi0, typename EvalXi1, typename EvalXi2>
torch::Tensor stackParametricJacobian(const EvalXi0& dx,
                                      const EvalXi1& dy,
                                      const EvalXi2& dz) {
    return torch::stack({
        torch::stack({*dx[0], *dy[0], *dz[0]}, 1),
        torch::stack({*dx[1], *dy[1], *dz[1]}, 1),
        torch::stack({*dx[2], *dy[2], *dz[2]}, 1)}, 1);
}

template <typename EvalXX, typename EvalXY, typename EvalXZ,
          typename EvalYY, typename EvalYZ, typename EvalZZ>
std::array<torch::Tensor, 3> stackParametricHessians(const EvalXX& xx,
                                                      const EvalXY& xy,
                                                      const EvalXZ& xz,
                                                      const EvalYY& yy,
                                                      const EvalYZ& yz,
                                                      const EvalZZ& zz) {
    std::array<torch::Tensor, 3> result;
    for (short c = 0; c < 3; ++c) {
        result[c] = torch::stack({
            torch::stack({*xx[c], *xy[c], *xz[c]}, 1),
            torch::stack({*xy[c], *yy[c], *yz[c]}, 1),
            torch::stack({*xz[c], *yz[c], *zz[c]}, 1)}, 1);
    }
    return result;
}

template <typename MultiPatch>
typename MultiPatch::patch_type localPatchWithTensor(const MultiPatch& space,
                                                     std::size_t patchIndex,
                                                     const torch::Tensor& tensor) {
    auto patch = space.patch(patchIndex);
    patch.from_tensor(space.local_tensor(patchIndex, tensor));
    return patch;
}

template <typename Patch>
std::array<torch::Tensor, 3>
parametricHessians(const Patch& patch,
                   const iganet::utils::TensorArray<3>& xi) {
    const auto xx = patch.template eval<iganet::deriv::dx ^ 2>(xi);
    const auto xy = patch.template eval<iganet::deriv::dx + iganet::deriv::dy>(xi);
    const auto xz = patch.template eval<iganet::deriv::dx + iganet::deriv::dz>(xi);
    const auto yy = patch.template eval<iganet::deriv::dy ^ 2>(xi);
    const auto yz = patch.template eval<iganet::deriv::dy + iganet::deriv::dz>(xi);
    const auto zz = patch.template eval<iganet::deriv::dz ^ 2>(xi);
    return stackParametricHessians(xx, xy, xz, yy, yz, zz);
}

template <typename GeometryMultiPatch, typename SolutionMultiPatch>
class MultipatchElasticitySolver {
public:
    using value_type = typename GeometryMultiPatch::value_type;

    MultipatchElasticitySolver(const GeometryMultiPatch& geometry,
                               const SolutionMultiPatch& displacement,
                               iganet::StrongDirichletConstraints<value_type> constraints,
                               MultipatchConfig cfg,
                               torch::TensorOptions options)
        : geometry_(geometry)
        , displacement_(displacement)
        , constraints_(std::move(constraints))
        , cfg_(std::move(cfg))
        , options_(options) {
        lambda_ = (cfg_.youngModulus * cfg_.poissonRatio) /
                  ((1.0 + cfg_.poissonRatio) * (1.0 - 2.0 * cfg_.poissonRatio));
        mu_ = cfg_.youngModulus / (2.0 * (1.0 + cfg_.poissonRatio));
    }

    struct LossParts {
        torch::Tensor total;
        torch::Tensor collocation;
        torch::Tensor dirichlet;
        torch::Tensor traction;
    };

    LossParts loss(const torch::Tensor& displacementTensor) const {
        auto collocationLoss = torch::zeros({}, options_);
        auto dirichletLoss = torch::zeros({}, options_);
        auto tractionLoss = torch::zeros({}, options_);

        for (std::size_t patchIndex = 0; patchIndex < geometry_.npatches(); ++patchIndex) {
            auto xi = geometry_.patch(patchIndex).greville(true);
            if (xi[0].numel() == 0) {
                xi = geometry_.patch(patchIndex).greville(false);
            }
            for (auto& x : xi) {
                x = x.to(options_.device());
            }

            const auto G = geometry_.patch(patchIndex);
            const auto U = localPatchWithTensor(displacement_, patchIndex, displacementTensor);

            const auto gdx = G.template eval<iganet::deriv::dx>(xi);
            const auto gdy = G.template eval<iganet::deriv::dy>(xi);
            const auto gdz = G.template eval<iganet::deriv::dz>(xi);
            const auto udx = U.template eval<iganet::deriv::dx>(xi);
            const auto udy = U.template eval<iganet::deriv::dy>(xi);
            const auto udz = U.template eval<iganet::deriv::dz>(xi);

            const auto J = stackParametricJacobian(gdx, gdy, gdz);
            const auto invJ = torch::linalg_inv(J);
            const auto gradUxi = stackParametricJacobian(udx, udy, udz);
            const auto gradU = torch::matmul(gradUxi, invJ);

            const auto hessG = parametricHessians(G, xi);
            const auto hessUxi = parametricHessians(U, xi);
            std::array<torch::Tensor, 3> hessU;
            for (short c = 0; c < 3; ++c) {
                auto corrected = hessUxi[c].clone();
                for (short k = 0; k < 3; ++k) {
                    corrected = corrected -
                        gradU.index({torch::indexing::Slice(), c, k}).view({-1, 1, 1}) *
                        hessG[k];
                }
                hessU[c] = torch::matmul(invJ.transpose(1, 2),
                                         torch::matmul(corrected, invJ));
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

            const auto divStressX =
                (lambda_ + 2.0 * mu_) * ux_xx + mu_ * ux_yy + mu_ * ux_zz +
                (lambda_ + mu_) * (uy_xy + uz_xz);
            const auto divStressY =
                mu_ * uy_xx + (lambda_ + 2.0 * mu_) * uy_yy + mu_ * uy_zz +
                (lambda_ + mu_) * (ux_yx + uz_yz);
            const auto divStressZ =
                mu_ * uz_xx + mu_ * uz_yy + (lambda_ + 2.0 * mu_) * uz_zz +
                (lambda_ + mu_) * (ux_zx + uy_zy);

            const auto divStress =
                torch::stack({divStressX, divStressY, divStressZ}, 1);
            const auto body = torch::tensor({cfg_.bodyForce[0], cfg_.bodyForce[1],
                                             cfg_.bodyForce[2]}, options_).view({1, 3});
            collocationLoss =
                collocationLoss +
                torch::mse_loss(divStress, -body.repeat({divStress.size(0), 1}));
        }

        const auto boundaryPointSets = geometry_.boundary_greville();
        for (const auto& [boundary, xiRaw] : boundaryPointSets) {
            const auto diri = boundaryTarget(boundary.side, cfg_.diriSides);
            const auto force = boundaryTarget(boundary.side, cfg_.forceSides);
            if (!diri && !force) {
                continue;
            }

            auto xi = xiRaw;
            for (auto& x : xi) {
                x = x.to(options_.device());
            }

            const auto G = geometry_.patch(boundary.patch);
            const auto U = localPatchWithTensor(displacement_, boundary.patch, displacementTensor);
            const auto uval = U.eval(xi);
            const auto u = stackVector(uval);

            if (diri) {
                const auto target = torch::tensor({(*diri)[0], (*diri)[1], (*diri)[2]},
                                                  options_).view({1, 3});
                dirichletLoss = dirichletLoss + torch::mse_loss(u, target.repeat({u.size(0), 1}));
            }

            if (force) {
                const auto gdx = G.template eval<iganet::deriv::dx>(xi);
                const auto gdy = G.template eval<iganet::deriv::dy>(xi);
                const auto gdz = G.template eval<iganet::deriv::dz>(xi);
                const auto udx = U.template eval<iganet::deriv::dx>(xi);
                const auto udy = U.template eval<iganet::deriv::dy>(xi);
                const auto udz = U.template eval<iganet::deriv::dz>(xi);
                const auto J = stackParametricJacobian(gdx, gdy, gdz);
                const auto invJ = torch::linalg_inv(J);
                const auto gradUxi = stackParametricJacobian(udx, udy, udz);
                const auto gradU = torch::matmul(gradUxi, invJ);
                const auto strain = 0.5 * (gradU + gradU.transpose(1, 2));
                const auto trace = strain.index({torch::indexing::Slice(), 0, 0}) +
                                   strain.index({torch::indexing::Slice(), 1, 1}) +
                                   strain.index({torch::indexing::Slice(), 2, 2});
                auto stress = 2.0 * mu_ * strain;
                for (short c = 0; c < 3; ++c) {
                    stress.index_put_({torch::indexing::Slice(), c, c},
                                      stress.index({torch::indexing::Slice(), c, c}) +
                                      lambda_ * trace);
                }
                const short fixed = static_cast<short>((boundary.side - 1) / 2);
                const short t0 = fixed == 0 ? 1 : 0;
                const short t1 = fixed == 2 ? 1 : 2;
                const auto a = J.index({torch::indexing::Slice(), torch::indexing::Slice(), t0});
                const auto b = J.index({torch::indexing::Slice(), torch::indexing::Slice(), t1});
                auto normal = torch::cross(a, b, 1);
                if (!iganet::MultiPatchInterface<3>::side_parameter(boundary.side)) {
                    normal = -normal;
                }
                normal = normal / normal.norm(2, 1).clamp_min(1e-12).view({-1, 1});
                const auto traction =
                    torch::matmul(stress, normal.unsqueeze(2)).squeeze(2);
                const auto target = torch::tensor({(*force)[0], (*force)[1], (*force)[2]},
                                                  options_).view({1, 3});
                tractionLoss =
                    tractionLoss + torch::mse_loss(traction, target.repeat({traction.size(0), 1}));
            }
        }

        const auto total = collocationLoss + tractionLoss;
        return {total, collocationLoss, dirichletLoss, tractionLoss};
    }

    std::vector<double> train(torch::Tensor& freeDisplacementTensor) const {
        torch::optim::Adam optimizer({freeDisplacementTensor},
                                     torch::optim::AdamOptions(cfg_.learningRate));
        std::vector<double> history;
        history.reserve(static_cast<std::size_t>(cfg_.maxEpoch));

        for (int epoch = 0; epoch < cfg_.maxEpoch; ++epoch) {
            optimizer.zero_grad();
            const auto displacementTensor =
                constraints_.expand_free_tensor(freeDisplacementTensor);
            const auto parts = loss(displacementTensor);
            parts.total.backward();
            optimizer.step();

            const double value = parts.total.detach().template item<double>();
            history.push_back(value);
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | total " << std::setw(14) << value
                      << " | coll " << std::setw(14)
                      << parts.collocation.detach().template item<double>()
                      << " | diri " << std::setw(14)
                      << parts.dirichlet.detach().template item<double>()
                      << " | traction " << std::setw(14)
                      << parts.traction.detach().template item<double>() << "\n";

            if (std::abs(value) < cfg_.minLoss) {
                break;
            }
        }

        return history;
    }

private:
    const GeometryMultiPatch& geometry_;
    const SolutionMultiPatch& displacement_;
    iganet::StrongDirichletConstraints<value_type> constraints_;
    MultipatchConfig cfg_;
    torch::TensorOptions options_;
    double lambda_{0.0};
    double mu_{0.0};
};

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
                            "sim_config_3D_multi_patch.json";
    const auto resultPath =
        repoRoot / "results" / "result_iganet_lin_elasticity_3D_multipatch.json";

    nlohmann::json j;
    try {
        std::ifstream cfgFile(configPath);
        if (cfgFile) {
            cfgFile >> j;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse config JSON: " << e.what() << "\n";
        return 1;
    }

    const auto xmlPath = resolveXmlPath(argc, argv, repoRoot, j);
    const int multipatchId = resolveMultiPatchId(j);
    const std::string boundaryLabel = resolveBoundaryLabel(j);
    const auto cfg = loadMultipatchConfig(j);

    const torch::Device computeDevice =
        torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                    : torch::Device(torch::kCPU);
    const auto options = iganet::Options<real_t>{}.device(computeDevice);
    const auto tensorOptions =
        torch::TensorOptions().dtype(torch::kFloat64).device(computeDevice);

    try {
        auto start = std::chrono::high_resolution_clock::now();

        multipatch_t geometry;
        geometry.set_matching_tolerance(1e-6, 1e-6);
        geometry.from_xml(xmlPath.string(), multipatchId);
        geometry.from_tensor(geometry.as_tensor().to(computeDevice));

        auto displacement = geometry.make_isoparametric_solution_space<3>(options);
        auto geometryTensor = geometry.as_tensor();
        iganet::StrongDirichletConstraints<real_t> constraints(displacement);
        for (const auto& entry : cfg.diriSides) {
            const int side = std::get<0>(entry);
            for (const auto& boundary : displacement.boundaries()) {
                if (boundary.side != side) {
                    continue;
                }
                constraints.fix_boundary(displacement, boundary, 0, std::get<1>(entry));
                constraints.fix_boundary(displacement, boundary, 1, std::get<2>(entry));
                constraints.fix_boundary(displacement, boundary, 2, std::get<3>(entry));
            }
        }

        auto initialDisplacementTensor =
            torch::zeros({displacement.as_tensor_size()}, tensorOptions);
        auto freeDisplacementTensor =
            constraints.extract_free_tensor(initialDisplacementTensor)
                .clone()
                .detach();
        freeDisplacementTensor.set_requires_grad(true);
        auto displacementTensor = constraints.expand_free_tensor(freeDisplacementTensor);

        const int64_t allGreville = countPatchGrevillePoints(geometry, false);
        const int64_t interiorGreville = countPatchGrevillePoints(geometry, true);
        auto boundaryPointSets = geometry.boundary_greville(boundaryLabel);
        if (boundaryPointSets.empty()) {
            boundaryPointSets = geometry.boundary_greville();
        }
        const int64_t boundaryGreville = countBoundaryPoints(boundaryPointSets);

        for (std::size_t patchIndex = 0; patchIndex < geometry.npatches(); ++patchIndex) {
            auto xi = geometry.patch(patchIndex).greville(true);
            if (xi[0].numel() == 0) {
                xi = geometry.patch(patchIndex).greville(false);
            }

            auto knotIndices = geometry.patch(patchIndex).find_knot_indices(xi);
            auto coeffIndices = geometry.patch(patchIndex).find_coeff_indices(knotIndices);
            auto globalIndices = geometry.global_coeff_indices(patchIndex, coeffIndices);

            auto G = geometry.eval_patch(patchIndex, xi, geometryTensor);
            auto dGdxi = geometry.template eval_patch<iganet::deriv::dx>(
                patchIndex, xi, geometryTensor);
            auto U = displacement.eval_patch(patchIndex, xi, displacementTensor);

            (void)globalIndices;
            (void)G;
            (void)dGdxi;
            (void)U;
        }

        MultipatchElasticitySolver solver(geometry, displacement, constraints, cfg,
                                          tensorOptions);
        auto lossHistory = solver.train(freeDisplacementTensor);
        displacementTensor =
            constraints.expand_free_tensor(freeDisplacementTensor).detach();
        displacement.from_tensor(displacementTensor);

        auto stop = std::chrono::high_resolution_clock::now();
        const double elapsed =
            std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

        nlohmann::json summary;
        summary["xml_path"] = xmlPath.string();
        summary["multipatch_id"] = multipatchId;
        summary["npatches"] = geometry.npatches();
        summary["ninterfaces"] = geometry.ninterfaces();
        summary["nboundaries"] = geometry.nboundaries();
        summary["geometry_scalar_dofs"] = geometry.ndofs();
        summary["displacement_scalar_dofs"] = displacement.ndofs();
        summary["geometry_tensor_size"] = geometryTensor.numel();
        summary["displacement_tensor_size"] = displacementTensor.numel();
        summary["patch_greville_points"] = allGreville;
        summary["patch_interior_greville_points"] = interiorGreville;
        summary["boundary_label"] = boundaryLabel;
        summary["boundary_point_sets"] = boundaryPointSets.size();
        summary["boundary_greville_points"] = boundaryGreville;
        summary["dirichlet_boundary_sets"] =
            countBoundaryMatches(boundaryPointSets, cfg.diriSides);
        summary["force_boundary_sets"] =
            countBoundaryMatches(boundaryPointSets, cfg.forceSides);
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
        summary["patches"] = patchesToJson(geometry, displacement, displacementTensor);
        summary["elapsed_seconds"] = elapsed;
        summary["note"] =
            "Strong-form collocation over global C0 displacement control-point DOFs.";

        {
            std::ofstream out(resultPath);
            if (!out) {
                throw std::runtime_error("Could not open result file: " + resultPath.string());
            }
            nlohmann::json result;
            result["multipatch_elasticity"] = summary;
            out << result.dump(1);
        }

        std::cout << "\n=== MULTIPATCH ELASTICITY SOLVE ===\n"
                  << "XML: " << xmlPath << "\n"
                  << "patches: " << geometry.npatches() << "\n"
                  << "interfaces: " << geometry.ninterfaces() << "\n"
                  << "outer boundaries: " << geometry.nboundaries() << "\n"
                  << "geometry scalar DOFs: " << geometry.ndofs() << "\n"
                  << "displacement scalar DOFs: " << displacement.ndofs() << "\n"
                  << "patch Greville points: " << allGreville << "\n"
                  << "interior Greville points: " << interiorGreville << "\n"
                  << "boundary Greville points: " << boundaryGreville << "\n"
                  << "loss initial: " << summary["loss_initial"] << "\n"
                  << "loss final: " << summary["loss_final"] << "\n"
                  << "result: " << resultPath << "\n"
                  << "==================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Multipatch preflight failed: " << e.what() << "\n";
        iganet::finalize();
        return 1;
    }

    iganet::finalize();
    return 0;
}
