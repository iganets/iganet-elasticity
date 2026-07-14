#pragma once

#include "headers/lin_elasticity_utils.hpp"

#include <iganet.h>
#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Shared aliases used by the linear-elasticity example code
// -----------------------------------------------------------------------------

using namespace iganet::literals;
using iganet_elasticity::utils::config::require;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using optimizer_config_t = iganet_elasticity::utils::config::optimizer_config;
using optimizer_type_t = iganet_elasticity::utils::config::optimizer_type;

template <typename GeometryMap>
using linear_elasticity_geometry_patch_t =
    std::decay_t<decltype(std::declval<GeometryMap>().template space<0>())>;

// -----------------------------------------------------------------------------
// 2D single-patch implementation
// -----------------------------------------------------------------------------

template <typename Optimizer, typename GeometryMap, typename Variable>
class linear_elasticity_2d_impl
    : public iganet::IgANet<Optimizer, std::tuple<GeometryMap>, std::tuple<Variable>>,
      public iganet::IgANetCustomizable<std::tuple<GeometryMap>, std::tuple<Variable>>
{
private:
    using Inputs  = std::tuple<GeometryMap>;
    using Outputs = std::tuple<Variable>;

    using Base = iganet::IgANet<Optimizer, Inputs, Outputs>;
    using geometry_patch_t = std::decay_t<decltype(std::declval<GeometryMap>().template space<0>())>;
    using variable_patch_t = std::decay_t<decltype(std::declval<Variable>().template space<0>())>;
    using prepared_eval_t = typename geometry_patch_t::PreparedEvaluation;

    struct CachedPointSet {
        prepared_eval_t eval;
        torch::Tensor J;
        torch::Tensor invJ;
        std::array<torch::Tensor, 2> hessG;
    };

    typename Base::template collPts_t<0> collPts_;
    typename Base::template collPts_t<0> interiorCollPts_;

    CachedPointSet collocationCache_;
    CachedPointSet interiorResidualCache_;
    std::optional<CachedPointSet> tractionBoundaryCache_;
    std::array<torch::Tensor, 2> tractionCollPts_;
    std::vector<int> tractionIntersectionCuts_;

    // material properties - lame's parameters
    double lambda_;
    double mu_;

    int nrCollPts_; 
    typename std::tuple_element_t<0, Outputs> ref_;


    // simulation parameters
    int MAX_EPOCH_;
    double MIN_LOSS_;
    int64_t NR_CTRL_PTS_;
    std::vector<int> TFBC_SIDES_;
    std::string JSON_PATH_;
    std::pair<double, double> BODY_FORCE_;
    std::vector<std::tuple<int, double, double>> FORCE_SIDES_;
    std::vector<std::tuple<int, double, double>> DIRI_SIDES_;
    bool SUPERVISED_LEARNING_;

public:
    /// @brief Constructor
    template <typename... Args>
    linear_elasticity_2d_impl(double lambda, double mu, bool SUPERVISED_LEARNING, int MAX_EPOCH, 
                    double MIN_LOSS, std::pair<double, double> BODY_FORCE, std::vector<int> TFBC_SIDES,
                    std::vector<std::tuple<int, double, double>> FORCE_SIDES,
                    std::vector<std::tuple<int, double, double>> DIRI_SIDES, 
                    int64_t NR_CTRL_PTS, std::string JSON_PATH, std::vector<int64_t> &&layers, 
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
        : Base( std::forward<std::vector<int64_t>>(layers),
                std::forward<std::vector<std::vector<std::any>>>(activations),
                std::forward<Args>(args)...),
                lambda_(lambda), mu_(mu), SUPERVISED_LEARNING_(SUPERVISED_LEARNING), MAX_EPOCH_(MAX_EPOCH), 
                MIN_LOSS_(MIN_LOSS), BODY_FORCE_(BODY_FORCE), TFBC_SIDES_(TFBC_SIDES), FORCE_SIDES_(FORCE_SIDES), 
                DIRI_SIDES_(DIRI_SIDES), NR_CTRL_PTS_(NR_CTRL_PTS), JSON_PATH_(std::move(JSON_PATH)), 
                ref_(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)) {}

    // /// @brief Returns a constant reference to the collocation points
    // auto const &collPts() const { return collPts_; }

    // /// @brief Returns a constant reference to the interior collocation points
    // auto const &interiorCollPts() const { return interiorCollPts_; }

    /// @brief Returns a constant reference to the reference solution
    auto const &ref() const { return ref_; }

    /// @brief Returns a non-constant reference to the reference solution
    auto &ref() { return ref_; }

private:
    const geometry_patch_t& geometryPatch() const {
        return this->template input<0>().template space<0>();
    }

    const variable_patch_t& displacementPatch() const {
        return this->template output<0>().template space<0>();
    }

    template <typename EvalXi0, typename EvalXi1>
    static torch::Tensor stack_parametric_jacobian(const EvalXi0& dx,
                                                   const EvalXi1& dy) {
        return torch::stack({
            torch::stack({*dx[0], *dy[0]}, 1),
            torch::stack({*dx[1], *dy[1]}, 1)}, 1);
    }

    template <typename EvalXX, typename EvalXY, typename EvalYY>
    static std::array<torch::Tensor, 2> stack_parametric_hessians(
        const EvalXX& xx,
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

    CachedPointSet prepareCachedPointSet(
        const iganet::utils::TensorArray<2>& xi) const {
        CachedPointSet cache;
        cache.eval = geometryPatch().template prepare_evaluation<
            iganet::deriv::dx,
            iganet::deriv::dy,
            iganet::deriv::dx ^ 2,
            iganet::deriv::dx + iganet::deriv::dy,
            iganet::deriv::dy ^ 2>(xi);

        if (cache.eval.numeval == 0) {
            const auto options = geometryPatch().as_tensor().options();
            cache.J = torch::empty({0, 2, 2}, options);
            cache.invJ = torch::empty({0, 2, 2}, options);
            cache.hessG = {
                torch::empty({0, 2, 2}, options),
                torch::empty({0, 2, 2}, options)};
            return cache;
        }

        const auto gdx = geometryPatch().template eval_from_prepared<iganet::deriv::dx>(cache.eval);
        const auto gdy = geometryPatch().template eval_from_prepared<iganet::deriv::dy>(cache.eval);
        cache.J = stack_parametric_jacobian(gdx, gdy);
        cache.invJ = torch::linalg_inv(cache.J);

        const auto gxx =
            geometryPatch().template eval_from_prepared<iganet::deriv::dx ^ 2>(cache.eval);
        const auto gxy =
            geometryPatch().template eval_from_prepared<iganet::deriv::dx + iganet::deriv::dy>(
                cache.eval);
        const auto gyy =
            geometryPatch().template eval_from_prepared<iganet::deriv::dy ^ 2>(cache.eval);
        cache.hessG = stack_parametric_hessians(gxx, gxy, gyy);
        return cache;
    }

    torch::Tensor evaluateParametricGradient(const CachedPointSet& cache) const {
        const auto udx =
            displacementPatch().template eval_from_prepared<iganet::deriv::dx>(cache.eval);
        const auto udy =
            displacementPatch().template eval_from_prepared<iganet::deriv::dy>(cache.eval);
        return stack_parametric_jacobian(udx, udy);
    }

    std::array<torch::Tensor, 2> evaluateParametricHessians(
        const CachedPointSet& cache) const {
        const auto uxx =
            displacementPatch().template eval_from_prepared<iganet::deriv::dx ^ 2>(cache.eval);
        const auto uxy = displacementPatch().template eval_from_prepared<
            iganet::deriv::dx + iganet::deriv::dy>(cache.eval);
        const auto uyy =
            displacementPatch().template eval_from_prepared<iganet::deriv::dy ^ 2>(cache.eval);
        return stack_parametric_hessians(uxx, uxy, uyy);
    }

    std::array<torch::Tensor, 2> evaluatePhysicalHessians(
        const CachedPointSet& cache,
        const torch::Tensor& gradU) const {
        const auto hessUxi = evaluateParametricHessians(cache);
        std::array<torch::Tensor, 2> hessU;
        for (iganet::short_t c = 0; c < 2; ++c) {
            auto corrected = hessUxi[c].clone();
            for (iganet::short_t k = 0; k < 2; ++k) {
                corrected = corrected -
                            gradU.index({torch::indexing::Slice(), c, k}).view({-1, 1, 1}) *
                                cache.hessG[k];
            }
            hessU[c] =
                torch::matmul(cache.invJ.transpose(1, 2), torch::matmul(corrected, cache.invJ));
        }
        return hessU;
    }

    torch::Tensor evaluateTractionValues(const CachedPointSet& cache) const {
        const auto gradUxi = evaluateParametricGradient(cache);
        const auto gradU = torch::matmul(gradUxi, cache.invJ);

        const auto ux_x = gradU.index({torch::indexing::Slice(), 0, 0});
        const auto ux_y = gradU.index({torch::indexing::Slice(), 0, 1});
        const auto uy_x = gradU.index({torch::indexing::Slice(), 1, 0});
        const auto uy_y = gradU.index({torch::indexing::Slice(), 1, 1});

        torch::Tensor tractionValuesX =
            torch::zeros({cache.eval.numeval}, gradU.options());
        torch::Tensor tractionValuesY =
            torch::zeros({cache.eval.numeval}, gradU.options());

        int pointCtr = 0;
        int sideCtr = 0;
        std::vector<int> neumannSides = TFBC_SIDES_;
        for (const auto& force : FORCE_SIDES_) {
            neumannSides.push_back(std::get<0>(force));
        }

        for (const int side : neumannSides) {
            const int nVals = nrCollPts_ - tractionIntersectionCuts_[sideCtr];
            for (int i = 0; i < nVals; ++i) {
                const int idx = pointCtr + i;
                if (side == 1) {
                    tractionValuesX[idx] =
                        -lambda_ * (ux_x[idx] + uy_y[idx]) - 2.0 * mu_ * ux_x[idx];
                    tractionValuesY[idx] = -mu_ * (uy_x[idx] + ux_y[idx]);
                } else if (side == 2) {
                    tractionValuesX[idx] =
                        lambda_ * (ux_x[idx] + uy_y[idx]) + 2.0 * mu_ * ux_x[idx];
                    tractionValuesY[idx] = mu_ * (uy_x[idx] + ux_y[idx]);
                } else if (side == 3) {
                    tractionValuesX[idx] = -mu_ * (uy_x[idx] + ux_y[idx]);
                    tractionValuesY[idx] =
                        -lambda_ * (ux_x[idx] + uy_y[idx]) - 2.0 * mu_ * uy_y[idx];
                } else if (side == 4) {
                    tractionValuesX[idx] = mu_ * (uy_x[idx] + ux_y[idx]);
                    tractionValuesY[idx] =
                        lambda_ * (ux_x[idx] + uy_y[idx]) + 2.0 * mu_ * uy_y[idx];
                }
            }
            pointCtr += nVals;
            ++sideCtr;
        }

        return torch::stack({tractionValuesX, tractionValuesY}, 1);
    }

    void prepareTractionBoundaryCache() {
        tractionBoundaryCache_.reset();
        tractionIntersectionCuts_.clear();

        if (TFBC_SIDES_.empty() && FORCE_SIDES_.empty()) {
            return;
        }

        std::vector<int> neumannSides;
        std::vector<int> diriOrForceSides;
        for (const auto& tuple : DIRI_SIDES_) {
            diriOrForceSides.push_back(std::get<0>(tuple));
        }

        neumannSides.reserve(TFBC_SIDES_.size() + FORCE_SIDES_.size());
        neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
        for (const auto& force : FORCE_SIDES_) {
            neumannSides.push_back(std::get<0>(force));
            diriOrForceSides.push_back(std::get<0>(force));
        }

        const auto boundaryOpts = std::get<0>(collPts_.boundary())[0].options();
        std::vector<torch::Tensor> tractionCollPtsX;
        std::vector<torch::Tensor> tractionCollPtsY;

        for (const int side : neumannSides) {
            if (side == 1) {
                auto collPtsY_tensor = std::get<0>(collPts_.boundary())[0];
                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) !=
                        diriOrForceSides.end() &&
                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) ==
                        diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}, boundaryOpts));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) ==
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}, boundaryOpts));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) !=
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 2}, boundaryOpts));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                    tractionIntersectionCuts_.push_back(2);
                } else {
                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_}, boundaryOpts));
                    tractionCollPtsY.push_back(std::get<0>(collPts_.boundary())[0]);
                    tractionIntersectionCuts_.push_back(0);
                }
            } else if (side == 2) {
                auto collPtsY_tensor = std::get<0>(collPts_.boundary())[0];
                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) !=
                        diriOrForceSides.end() &&
                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) ==
                        diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}, boundaryOpts));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) ==
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}, boundaryOpts));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) !=
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 2}, boundaryOpts));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                    tractionIntersectionCuts_.push_back(2);
                } else {
                    tractionCollPtsX.push_back(torch::ones({nrCollPts_}, boundaryOpts));
                    tractionCollPtsY.push_back(std::get<0>(collPts_.boundary())[0]);
                    tractionIntersectionCuts_.push_back(0);
                }
            } else if (side == 3) {
                auto collPtsX_tensor = std::get<0>(collPts_.boundary())[0];
                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) !=
                        diriOrForceSides.end() &&
                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) ==
                        diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) ==
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) !=
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 2}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(2);
                } else {
                    tractionCollPtsX.push_back(std::get<0>(collPts_.boundary())[0]);
                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(0);
                }
            } else if (side == 4) {
                auto collPtsX_tensor = std::get<0>(collPts_.boundary())[0];
                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) !=
                        diriOrForceSides.end() &&
                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) ==
                        diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) ==
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(1);
                } else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) !=
                               diriOrForceSides.end() &&
                           std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) !=
                               diriOrForceSides.end()) {
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 2}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(2);
                } else {
                    tractionCollPtsX.push_back(std::get<0>(collPts_.boundary())[0]);
                    tractionCollPtsY.push_back(torch::ones({nrCollPts_}, boundaryOpts));
                    tractionIntersectionCuts_.push_back(0);
                }
            } else {
                throw std::invalid_argument("Side for traction BC has to be 1, 2, 3 or 4.");
            }
        }

        tractionCollPts_ = {
            torch::cat(tractionCollPtsX, 0),
            torch::cat(tractionCollPtsY, 0)};
        tractionBoundaryCache_ = prepareCachedPointSet(tractionCollPts_);
    }
    
public:
    /// @brief Writes data to a JSON file
    void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
    
        // create json object
        nlohmann::json jsonData;

        // try to read the JSON data from the file
        try {
            std::ifstream json_file_in(JSON_PATH_);
            if (json_file_in.is_open()) {
                json_file_in >> jsonData;
                json_file_in.close();
            } else {
                std::cerr << "Warning: Could not open file for reading: " 
                        << JSON_PATH_ << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading JSON file: " << JSON_PATH_ 
                    << ". Exception: " << e.what() << "\n";
        }

        // add new data to the JSON object
        try {
            jsonData[key] = data;
        } catch (const std::exception& e) {
            std::cerr << "Error adding key to JSON object: " << e.what() << "\n";
            return;
        }

        // write the JSON data to the file
        try {
            std::ofstream json_file_out(JSON_PATH_);
            if (json_file_out.is_open()) {
                json_file_out << jsonData.dump(1);
                json_file_out.close();
            } else {
                std::cerr << "Error: Could not open file for writing: " 
                        << JSON_PATH_ << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error writing JSON file: " << JSON_PATH_ 
                    << ". Exception: " << e.what() << "\n";
        }
    }

    /// @brief helper function to load the std collocation displacements from a JSON file
    torch::Tensor loadDisplacements() {
        // create options for the tensor
        auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
    
        // open the JSON file
        std::ifstream file(JSON_PATH_);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + JSON_PATH_);
        }
    
        // parse the JSON file
        nlohmann::json jsonData;
        file >> jsonData;
        file.close();
    
        // extract the stdCollDisplacement array
        auto stdCollDisplacements_j = jsonData["stdCollDisplacement"];
        int nrStdCollCtrlPts = stdCollDisplacements_j.size();
    
        // create a tensor for the displacements
        torch::Tensor stdCollDisplacement = torch::empty({nrStdCollCtrlPts, 2}, options);
    
        // fill the tensor with data from the JSON file
        for (int i = 0; i < nrStdCollCtrlPts; ++i) {
            stdCollDisplacement[i][0] = stdCollDisplacements_j[i][0].get<double>();
            stdCollDisplacement[i][1] = stdCollDisplacements_j[i][1].get<double>();
        }
    
        return stdCollDisplacement;
    }

    /// @brief helper function to calculate the Greville abscissae
    static std::vector<double> computeGrevilleAbscissae
        (const std::vector<double>& knotVector, int degree, int numCtrlPts) {
        std::vector<double> greville(numCtrlPts, 0.0);
        
        for (int i = 0; i < numCtrlPts; ++i) {
            double sum = 0.0;
            for (int j = i + 1; j <= i + degree; ++j) {
                sum += knotVector.at(j);
            }
            greville[i] = sum / degree;
        }
        return greville;
    }

#ifdef IGANET_WITH_GISMO
    /// @brief GISMO workflow (returns torch tensors)
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RunGismoSimulation(
        int64_t NR_CTRL_PTS, int DEGREE, double YOUNG_MODULUS, double POISSON_RATIO,
        const std::vector<std::tuple<int, double, double>>& DIRI_SIDES,
        const std::vector<std::tuple<int, double, double>>& FORCE_SIDES,
        const std::pair<double, double>& BODY_FORCE){
            
        // --- torch options (CPU, double)
        auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

        const int64_t nPts = NR_CTRL_PTS * NR_CTRL_PTS;

        // torch outputs
        torch::Tensor ctrlPts       = torch::empty({nPts, 2}, opts);
        torch::Tensor displacements = torch::empty({nPts, 2}, opts);
        torch::Tensor stresses      = torch::empty({nPts, 1}, opts);

        // Accessors for fast element access (no .item in loops!)
        auto ctrlA = ctrlPts.accessor<double, 2>();
        auto dispA = displacements.accessor<double, 2>();
        auto strA  = stresses.accessor<double, 2>();

        // We still need a gsMatrix for GISMO geometry construction + point queries
        gismo::gsMatrix<double> ctrlPts_gs(nPts, 2);

        // create knot vectors
        gismo::gsKnotVector<double> knotVector_u(0.0, 1.0, NR_CTRL_PTS - DEGREE - 1, DEGREE + 1);
        gismo::gsKnotVector<double> knotVector_v(0.0, 1.0, NR_CTRL_PTS - DEGREE - 1, DEGREE + 1);

        // calculation of the Greville points
        std::vector<double> grevilleU = computeGrevilleAbscissae(knotVector_u, DEGREE, NR_CTRL_PTS);
        std::vector<double> grevilleV = computeGrevilleAbscissae(knotVector_v, DEGREE, NR_CTRL_PTS);

        // systematic placement of control points according to greville abscissae
        int64_t index = 0;
        for (int j = 0; j < NR_CTRL_PTS; ++j) {
            for (int i = 0; i < NR_CTRL_PTS; ++i) {
                const double x = grevilleU[i];
                const double y = grevilleV[j];

                // write into torch tensor
                ctrlA[index][0] = x;
                ctrlA[index][1] = y;

                // write into gismo matrix
                ctrlPts_gs(index, 0) = x;
                ctrlPts_gs(index, 1) = y;

                ++index;
            }
        }

        // create geometry
        gismo::gsTensorBSpline<2, double> geometry(knotVector_u, knotVector_v, ctrlPts_gs);

        // create multipatch and add the geometry
        gismo::gsMultiPatch<double> multiPatch;
        multiPatch.addPatch(geometry);
        gismo::gsMultiBasis<> basis(multiPatch);

        // helper to map 1-4 to gs boundary enums
        auto getGsBoundarySide = [](int side) -> gismo::boundary::side {
            switch (side) {
                case 1: return gismo::boundary::west;
                case 2: return gismo::boundary::east;
                case 3: return gismo::boundary::south;
                case 4: return gismo::boundary::north;
                default:
                    throw std::invalid_argument("Invalid side number (must be 1 to 4)");
            }
        };

        // define boundary conditions
        gismo::gsBoundaryConditions<double> bcInfo;

        // Dirichlet BCs
        for (const auto& d : DIRI_SIDES) {
            int side = std::get<0>(d);
            double xVal = std::get<1>(d);
            double yVal = std::get<2>(d);
            auto gsSide = getGsBoundarySide(side);

            bcInfo.addCondition(0, gsSide, gismo::condition_type::dirichlet,
                                gismo::gsConstantFunction<double>(xVal, 2), 0);
            bcInfo.addCondition(0, gsSide, gismo::condition_type::dirichlet,
                                gismo::gsConstantFunction<double>(yVal, 2), 1);
        }

        // Neumann (Traction) BCs
        for (const auto& f : FORCE_SIDES) {
            int side = std::get<0>(f);
            double tx = std::get<1>(f);
            double ty = std::get<2>(f);
            auto gsSide = getGsBoundarySide(side);

            gismo::gsFunctionExpr<> traction(std::to_string(tx), std::to_string(ty), 2);
            bcInfo.addCondition(0, gsSide, gismo::condition_type::neumann, traction);
        }

        // body force
        gismo::gsConstantFunction<double> bodyForce(BODY_FORCE.first, BODY_FORCE.second, 2);

        // initialize the elasticity assembler
        gismo::gsElasticityAssembler<double> assembler(geometry, basis, bcInfo, bodyForce);
        assembler.options().setReal("YoungsModulus", YOUNG_MODULUS);
        assembler.options().setReal("PoissonsRatio", POISSON_RATIO);
        assembler.assemble();

        // solve the system
        gismo::gsSparseSolver<>::CGDiagonal solver;
        gismo::gsMatrix<double> solution;
        solver.compute(assembler.matrix());
        solution = solver.solve(assembler.rhs());

        // create a multipatch object for the solution
        gismo::gsMultiPatch<double> solutionPatch;
        assembler.constructSolution(solution, assembler.allFixedDofs(), solutionPatch);

        // create a piecewise function for the stresses
        gismo::gsPiecewiseFunction<double> stressFunction;

        // calculate von Mises stresses (cauchy form)
        assembler.constructCauchyStresses(solutionPatch, stressFunction,
                                        gismo::stress_components::von_mises);

        // loop all control points
        for (int i = 0; i < ctrlPts_gs.rows(); ++i) {
            // create temp point
            gismo::gsMatrix<double> point(2, 1);
            point(0, 0) = ctrlPts_gs(i, 0);
            point(1, 0) = ctrlPts_gs(i, 1);

            // DISPLACEMENT EVALUATION
            auto u = solutionPatch.patch(0).eval(point);
            dispA[i][0] = u(0);
            dispA[i][1] = u(1);

            // STRESS EVALUATION
            const auto &segment = stressFunction.piece(0);
            gismo::gsMatrix<double> s(1, 1);
            segment.eval_into(point, s);
            strA[i][0] = s(0, 0);
        }

        return {ctrlPts, displacements, stresses};
    }
#endif

    /// @brief Initializes the epoch
    bool epoch(int64_t epoch) override {
    
        // print epoch number
        std::cout << "Epoch: " << epoch << std::endl;   

        if (epoch == 0) {
            Base::inputs(epoch);
            collPts_         = Base::template collPts<0>(iganet::collPts::greville);
            interiorCollPts_ = Base::template collPts<0>(iganet::collPts::greville_interior);


            // WARNING, only works for equal number of control points in x and y direction
            nrCollPts_ = static_cast<int>(std::sqrt(collPts_.interior()[0].size(0)));
            torch::Tensor collPtsCoeffs = collPts_.interior()[0].slice(0, 0, nrCollPts_);
            nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
            for (int i = 0; i < collPtsCoeffs.size(0); ++i) {
                collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
            }
            appendToJsonFile("net_collPtsCoeffsRef1", collPtsCoeffs_j);
            appendToJsonFile("net_nrCollPtsRef1", {nrCollPts_});

            collocationCache_ = prepareCachedPointSet(collPts_.interior());
            interiorResidualCache_ = prepareCachedPointSet(interiorCollPts_.interior());
            prepareTractionBoundaryCache();

            return true;
        } 
        else {
            return false;
        }
    }

    /// @brief Computes the loss function
    torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

        // create u_ from the training's outputs
        this->template output<0>().from_tensor(outputs);

        // pre-allocation of the loss values
        torch::Tensor totalLoss; 
        torch::Tensor elastLoss;
        std::optional<torch::Tensor> bcLoss;
        std::optional<torch::Tensor> tfbcLoss;
        std::optional<torch::Tensor> gsLoss;
        std::optional<torch::Tensor> forceLoss;

        // pre-allocation of the tensors for the traction boundary conditions
        std::optional<torch::Tensor> forceValues;
        std::optional<torch::Tensor> targetForce;
        std::optional<torch::Tensor> tractionFreeValues;
        std::optional<torch::Tensor> tractionZeros;

        // TRACTION BOUNDARY CONDITIONS
    
        // only calculate the traction-free boundary conditions if there are any
        if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty())
        {
            const auto tractionValues = evaluateTractionValues(*tractionBoundaryCache_);

            if (!FORCE_SIDES_.empty()) {
                // calculate total cutlength from the cached force-side point counts
                int cutlength = 0;
                int forceSize = FORCE_SIDES_.size();
                for (int i = static_cast<int>(tractionIntersectionCuts_.size()) - forceSize; 
                        i < static_cast<int>(tractionIntersectionCuts_.size()); ++i) {
                    cutlength += nrCollPts_ - tractionIntersectionCuts_[i];
                }
                // separate traction-free and force parts
                tractionFreeValues.emplace
                    (tractionValues.slice(0, 0, tractionValues.size(0) - cutlength));
                tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
                forceValues.emplace(tractionValues.slice(0, tractionValues.size(0) -
                                    cutlength, tractionValues.size(0)));
                targetForce.emplace(torch::zeros_like(*forceValues));
                // fill in the known force values
                int offset = 0;
                int startIdx = static_cast<int>(tractionIntersectionCuts_.size()) - forceSize;
                for (size_t i = 0; i < FORCE_SIDES_.size(); ++i) {
                    int reducedPts = nrCollPts_ - tractionIntersectionCuts_[startIdx + i];
                    auto rowSlice = (*targetForce).slice(0, offset, offset + reducedPts);
                    rowSlice.slice(1, 0, 1).fill_(std::get<1>(FORCE_SIDES_[i]));  // x-value
                    rowSlice.slice(1, 1, 2).fill_(std::get<2>(FORCE_SIDES_[i]));  // y-value
                    offset += reducedPts;
                }       
            }
            else {
                // set the traction-free values
                tractionFreeValues.emplace(tractionValues);
                // set the target values to zero
                tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
            }

        }

        // LINEAR ELASTICITY EQUATION

        const auto gradUxi = evaluateParametricGradient(interiorResidualCache_);
        const auto gradU = torch::matmul(gradUxi, interiorResidualCache_.invJ);
        const auto hessU = evaluatePhysicalHessians(interiorResidualCache_, gradU);

        const auto ux_xx = hessU[0].index({torch::indexing::Slice(), 0, 0});
        const auto ux_xy = hessU[0].index({torch::indexing::Slice(), 0, 1});
        const auto ux_yy = hessU[0].index({torch::indexing::Slice(), 1, 1});

        const auto uy_xx = hessU[1].index({torch::indexing::Slice(), 0, 0});
        const auto uy_xy = hessU[1].index({torch::indexing::Slice(), 0, 1});
        const auto uy_yy = hessU[1].index({torch::indexing::Slice(), 1, 1});

        // pre-allocation of the results
        torch::Tensor divStressX = torch::zeros({interiorResidualCache_.eval.numeval}, gradU.options());
        torch::Tensor divStressY = torch::zeros({interiorResidualCache_.eval.numeval}, gradU.options());

        // calculation of the divergence of the stress tensor, this is what we're trying to minimize
        for (int i = 0; i < interiorResidualCache_.eval.numeval; ++i) {
            // x-direction
            divStressX[i] = (lambda_ + 2 * mu_) * ux_xx[i] + 
                            mu_ * ux_yy[i] + (lambda_ + mu_) * uy_xy[i];

            // y-direction
            divStressY[i] = mu_ * uy_xx[i] + (lambda_ + 2 * mu_) * uy_yy[i] + 
                            (lambda_ + mu_) * ux_xy[i];
            
        }
        
        // create a tensor of the divergence of the stress tensor
        torch::Tensor divStress = torch::stack({divStressX, divStressY}, /*dim=*/1);

        // BODY FORCE: constant vector (fx, fy)
        auto opts = divStress.options();  // device + dtype passend zu divStress

        torch::Tensor bodyForce = torch::tensor(
            {BODY_FORCE_.first, BODY_FORCE_.second},
            opts
        ).view({1, 2}).repeat({divStress.size(0), 1});   // (N,2)

        // UNSUPERVISED LEARNING (default)
        if (SUPERVISED_LEARNING_ == false) {

            // create command line output variable for all the different losses
            std::ostringstream singleLossOutput;

            // calculation of the loss function for double-sided constraint solid
            // div(sigma) + f = 0 --> div(sigma) = -f
            elastLoss = torch::mse_loss(divStress, bodyForce);
            
            // add the elasticity loss to the total loss
            totalLoss = elastLoss;

            // add the elasticity loss to the cmd-output variable
            singleLossOutput << "EL " << std::setw(11) << elastLoss.item<double>();

            // only consider traction-free-bc (tfbc) loss if tfbcs are applied
            if (!TFBC_SIDES_.empty()) {
                tfbcLoss = torch::mse_loss(*tractionFreeValues, *tractionZeros);
                totalLoss += *tfbcLoss;
                singleLossOutput << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
            }

            // only consider force loss if force is applied
            if (!FORCE_SIDES_.empty()) {
                forceLoss = torch::mse_loss(*forceValues, *targetForce);
                totalLoss += *forceLoss;
                singleLossOutput << " + FL " << std::setw(11) << (*forceLoss).item<double>();
            }

            // only consider BC loss if dirichlet BCs are applied
            if (!DIRI_SIDES_.empty()) {
                // add a BC weight for penalization of the training
                int bcWeight = 1e7;
                // initialize bcLoss variable
                bcLoss = torch::tensor(0.0, outputs.options());

                // evaluation of the displacements at the boundary points
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.boundary());
                // evaluation of the displacements at the reference boundary points
                auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.boundary());

                // loop through all dirichlet sides
                for (const auto& side : DIRI_SIDES_) {
                    int sideNr = std::get<0>(side);
                    
                    switch (sideNr) {
                        case 1: 
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                            break;
                        case 2:
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                            break;
                        case 3:
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                            break;
                        case 4:
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
                                torch::mse_loss(*std::get<3>(u_bdr)[1], *std::get<3>(bdr)[1]));
                            break;
                        default:
                            std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                    }
                }
                totalLoss += *bcLoss;
                singleLossOutput << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight 
                                << " * 1e" << static_cast<int>(std::log10(bcWeight));
            }

            // print the loss values
            std::cout << std::setw(11) << 
                totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
        }
        
        // SUPERVISED LEARNING
        else if (SUPERVISED_LEARNING_ == true) {

            // create command line output variable for all the different losses
            std::ostringstream singleLossOutput;
        
            // preprocess the outputs for comparison with the std collocation solution
            torch::Tensor modifiedOutputs = outputs * 1.0;
        
            // create netDisplacements_ from slices of modifiedOutputs
            torch::Tensor netDisplacements_ = torch::stack({
                modifiedOutputs.slice(0, 0, outputs.size(0) / 2),
                modifiedOutputs.slice(0, outputs.size(0) / 2, outputs.size(0)),
            }, 1);

            // load the displacements from the std collocation solution
            torch::Tensor stdCollDisplacements_ = loadDisplacements().to(netDisplacements_.options());

            // supervised loss: MSE of net against standard collocation solution
            gsLoss = 1e9 * torch::mse_loss(netDisplacements_, stdCollDisplacements_);

            // calculation of the loss function for double-sided constraint solid
            // div(sigma) + f = 0 --> div(sigma) = -f
            elastLoss = torch::mse_loss(divStress, bodyForce);

            // add the elasticity loss and supervised loss to the total loss
            totalLoss = *gsLoss + elastLoss;

            // add the elasticity and supervised losses to the cmd-output variable
            singleLossOutput << "GL " << std::setw(11) << (*gsLoss).item<double>()
                            << " + EL " << std::setw(11) << elastLoss.item<double>();

            // only consider traction-free-bc (tfbc) loss if tfbcs are applied
            if (!TFBC_SIDES_.empty()) {
                tfbcLoss = torch::mse_loss(*tractionFreeValues, *tractionZeros);
                totalLoss += *tfbcLoss;
                singleLossOutput << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
            }

            // only consider force loss if force is applied
            if (!FORCE_SIDES_.empty()) {
                forceLoss = torch::mse_loss(*forceValues, *targetForce);
                totalLoss += *forceLoss;
                singleLossOutput << " + FL " << std::setw(11) << (*forceLoss).item<double>();
            }

            // only consider BC loss if dirichlet BCs are applied
            if (!DIRI_SIDES_.empty()) {
                // add a BC weight for penalization of the training
                int bcWeight = 1e0;
                // initialize bcLoss variable
                bcLoss = torch::tensor(0.0, outputs.options());

                // evaluation of the displacements at the boundary points
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.boundary());
                // evaluation of the displacements at the reference boundary points
                auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.boundary());

                // loop through all dirichlet sides
                for (const auto& side : DIRI_SIDES_) {
                    int sideNr = std::get<0>(side);

                    switch (sideNr) {
                        case 1:
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                            break;
                        case 2:
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                            break;
                        case 3:
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                            break;
                        case 4:
                            *bcLoss += bcWeight * 
                                (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
                                torch::mse_loss(*std::get<3>(u_bdr)[1], *std::get<3>(bdr)[1]));
                            break;
                        default:
                            std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                    }
                }
                totalLoss += *bcLoss;
                singleLossOutput << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight 
                                << " * 1e" << static_cast<int>(std::log10(bcWeight));
            }

            // print the loss values
            std::cout << std::setw(11) << 
                totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
        }

        else {
            throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_");
        }

        // POSTPROCESSING PREPARATION - WRITING DATA TO JSON FILE

        // only calculate this at the end of the simulation
        if ((epoch == MAX_EPOCH_ - 1) || (totalLoss.item<double>() <= MIN_LOSS_)) {
            
            // STRESS CALCULATION

            const auto gradUxiColl = evaluateParametricGradient(collocationCache_);
            const auto gradUColl = torch::matmul(gradUxiColl, collocationCache_.invJ);

            const auto ux_x = gradUColl.index({torch::indexing::Slice(), 0, 0});
            const auto ux_y = gradUColl.index({torch::indexing::Slice(), 0, 1});
            const auto uy_x = gradUColl.index({torch::indexing::Slice(), 1, 0});
            const auto uy_y = gradUColl.index({torch::indexing::Slice(), 1, 1});

            // allocate the stress tensor
            torch::Tensor sigma_xx = torch::zeros({collocationCache_.eval.numeval}, gradUColl.options());
            torch::Tensor sigma_xy = torch::zeros({collocationCache_.eval.numeval}, gradUColl.options());
            torch::Tensor sigma_yy = torch::zeros({collocationCache_.eval.numeval}, gradUColl.options()); 
            torch::Tensor sigma_vm = torch::zeros({collocationCache_.eval.numeval}, gradUColl.options());   

            torch::Tensor epsilon_xx = torch::zeros({collocationCache_.eval.numeval}, gradUColl.options());
            torch::Tensor epsilon_yy = torch::zeros({collocationCache_.eval.numeval}, gradUColl.options());
            torch::Tensor poisson_re = torch::zeros({collocationCache_.eval.numeval}, gradUColl.options());

            // create json object for the stresses
            nlohmann::json netVmStresses_j = nlohmann::json::array();
            nlohmann::json netXStresses_j = nlohmann::json::array();
            nlohmann::json netYStresses_j = nlohmann::json::array();
            nlohmann::json netPoisson_j = nlohmann::json::array();

            // calculate the stress tensor
            for (int i = 0; i < collocationCache_.eval.numeval; ++i) {
                // calculate the stress values for all collocation points
                sigma_xx[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * ux_x[i];
                sigma_xy[i] = mu_ * (uy_x[i] + ux_y[i]);
                sigma_yy[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * uy_y[i];
                
                // calculate von mises stress at the collocation points
                sigma_vm[i] = sqrt(sigma_xx[i] * sigma_xx[i] + sigma_yy[i] * sigma_yy[i] 
                                - sigma_xx[i] * sigma_yy[i] + sigma_xy[i] * sigma_xy[i] * 3);
                
                // calculate the strains at the collocation points
                epsilon_xx[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) * 
                    (sigma_xx[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_yy[i]);
                epsilon_yy[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) * 
                    (sigma_yy[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_xx[i]);

                // only valid for load in x-direction
                poisson_re[i] = - epsilon_yy[i] / epsilon_xx[i];
                
                // add the stresses to the json objects
                netVmStresses_j.push_back({sigma_vm[i].item<double>()});
                netXStresses_j.push_back({sigma_xx[i].item<double>()});
                netYStresses_j.push_back({sigma_yy[i].item<double>()});
                // add the poisson ratio to the json object
                netPoisson_j.push_back({poisson_re[i].item<double>()});
            }

            // write the stresses and poisson ratios to the json file
            appendToJsonFile("net_VmStresses", netVmStresses_j);
            appendToJsonFile("net_XStresses", netXStresses_j);
            appendToJsonFile("net_YStresses", netYStresses_j);
            appendToJsonFile("net_Poisson", netPoisson_j);

            // CALCULATE THE NEW POSITION OF THE COLLPTS

            // create a tensor of the collocation points
            torch::Tensor collPtsFirstAsTensor = torch::stack(
                {std::get<0>(collPts_.interior()), std::get<1>(collPts_.interior())}, 1);
            auto displacementOfCollPts = this->template output<0>().eval(collPts_.interior());
            torch::Tensor displacementAsTensor = torch::stack(
                {*(displacementOfCollPts[0]), *(displacementOfCollPts[1]) }, 1);

            // create json objects for the collocation points' reference and displaced position
            nlohmann::json collPtsFirst_j = nlohmann::json::array();
            nlohmann::json collPtsFirstDispl_j = nlohmann::json::array();
            for (int i = 0; i < collPtsFirstAsTensor.size(0); ++i) {
                // reference position of the collocation points
                collPtsFirst_j.push_back({collPtsFirstAsTensor[i][0].item<double>(), 
                                        collPtsFirstAsTensor[i][1].item<double>()});
                // new position of the collocation points
                collPtsFirstDispl_j.push_back({collPtsFirstAsTensor[i][0].item<double>() + 
                                            displacementAsTensor[i][0].item<double>(), 
                                            collPtsFirstAsTensor[i][1].item<double>() + 
                                            displacementAsTensor[i][1].item<double>()});
            }
            // write the collocation points' original position to the json file
            appendToJsonFile("net_collPtsFirstAsTensor", collPtsFirst_j);
            // write the collocation points' new position to the json file
            appendToJsonFile("net_collPtsFirstAfterDisplacementAsTensor", collPtsFirstDispl_j);

            // WRITING DIVERGENCE OF THE STRESS TENSOR TO JSON FILE

            nlohmann::json netDivergenceX_j = nlohmann::json::array();
            nlohmann::json netDivergenceY_j = nlohmann::json::array();

            for (int i = 0; i < divStressX.size(0); ++i) {
                netDivergenceX_j.push_back({divStressX[i].item<double>()});
                netDivergenceY_j.push_back({divStressY[i].item<double>()});
            }

            // write the divergence of the stress tensor to the json file
            appendToJsonFile("net_DivergenceX", netDivergenceX_j);
            appendToJsonFile("net_DivergenceY", netDivergenceY_j);
        }
        return totalLoss;
    }
}; // class linear_elasticity_2d_impl

// -----------------------------------------------------------------------------
// 3D single-patch implementation
// -----------------------------------------------------------------------------

template <typename Optimizer, typename GeometryMap, typename Variable>
class linear_elasticity_3d_impl
    : public iganet::IgANet<Optimizer, std::tuple<GeometryMap>, std::tuple<Variable>>,
      public iganet::IgANetCustomizable<std::tuple<GeometryMap>, std::tuple<Variable>>
{
private:
    using Inputs       = std::tuple<GeometryMap>;
    using Outputs      = std::tuple<Variable>;
    using Base         = iganet::IgANet<Optimizer, Inputs, Outputs>;
    using Customizable = iganet::IgANetCustomizable<Inputs, Outputs>;

    typename Base::template collPts_t<0> collPts_;
    typename Base::template collPts_t<0> interiorCollPts_;

    // Precomputed index tensors for fast Jacobian/Hessian evaluation.
 
    typename Customizable::template output_interior_knot_indices_t<0> var_knot_indices_;
    typename Customizable::template output_interior_coeff_indices_t<0> var_coeff_indices_;

    typename Customizable::template output_interior_knot_indices_t<0> var_knot_indices_interior_;
    typename Customizable::template output_interior_coeff_indices_t<0> var_coeff_indices_interior_;

    typename Customizable::template output_interior_knot_indices_t<0> var_knot_indices_boundary_;
    typename Customizable::template output_interior_coeff_indices_t<0> var_coeff_indices_boundary_;

    typename Customizable::template input_interior_knot_indices_t<0> G_knot_indices_;
    typename Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_;

    typename Customizable::template input_interior_knot_indices_t<0> G_knot_indices_interior_;
    typename Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_interior_;

    typename Customizable::template input_interior_knot_indices_t<0> G_knot_indices_boundary_;
    typename Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_boundary_;

    
    // Material parameters.
   
    double lambda_; ///< Lamé parameter lambda.
    double mu_;     ///< Lamé parameter mu (shear modulus).

    typename std::tuple_element_t<0, Outputs> ref_;

    // Simulation parameters.
 
    int    MAX_EPOCH_;
    double MIN_LOSS_;
    int64_t NR_CTRL_PTS_;

    std::array<double, 3> BODY_FORCE_; ///< Body force [fx, fy, fz].

    /// Dirichlet boundary conditions: (side, ux, uy, uz)
    std::vector<std::tuple<int, double, double, double>> DIRI_SIDES_;
    /// Neumann boundary conditions / prescribed tractions: (side, tx, ty, tz)
    std::vector<std::tuple<int, double, double, double>> FORCE_SIDES_;
    /// Traction-free sides as side numbers.
    std::vector<int> TFBC_SIDES_;

    std::string JSON_PATH_;
    bool        SUPERVISED_LEARNING_;

    bool                                  tractionPtsInitialized_ = false;
    std::array<torch::Tensor, 3>          tractionCollPts_;
    std::vector<int>                      nPtsPerSide_;
    std::optional<torch::Tensor>          stdDisplacements_;

    int nrCollPts_ = 0;

    /// @brief Returns true if the given side has a Dirichlet condition.
    bool isDirichletSide(int sideNr) const {
        return std::any_of(DIRI_SIDES_.begin(), DIRI_SIDES_.end(),
            [&](const auto& t) { return std::get<0>(t) == sideNr; });
    }

    /// @brief Returns true if the given side has a Neumann traction condition.
    bool isNeumannSide(int sideNr) const {
        return std::any_of(FORCE_SIDES_.begin(), FORCE_SIDES_.end(),
            [&](const auto& t) { return std::get<0>(t) == sideNr; });
    }

    /// @brief Returns boundary-condition priority for conflict resolution at corners.
    /// @details Priority order: Dirichlet = 3, Neumann = 2, traction-free = 1.
    int bc_priority(int sideNr) const {
        if (isDirichletSide(sideNr)) return 3;
        if (isNeumannSide(sideNr))   return 2;
        return 1;
    }

    /// @brief Returns true if another side wins a shared corner against sideNr.
    bool bc_other_wins(int otherSide, int sideNr) const {
        int op = bc_priority(otherSide);
        int tp = bc_priority(sideNr);
        if (op > tp)                    return true;
        if (op == tp && otherSide < sideNr) return true;
        return false;
    }

    /// @brief Returns true if two cube faces intersect geometrically.
    bool sidesIntersect(int a, int b) const {
        if (a == b) return false;
        return !((a==1&&b==2)||(a==2&&b==1)||
                 (a==3&&b==4)||(a==4&&b==3)||
                 (a==5&&b==6)||(a==6&&b==5));
    }

    /// @brief Builds physical face coordinates for one boundary side.
    std::array<torch::Tensor, 3> getFaceBoundaryPoints(int sideNr) const {
        switch (sideNr) {
            case 1: { auto Y=std::get<0>(collPts_.boundary())[0];
                      auto Z=std::get<0>(collPts_.boundary())[1];
                      return {torch::zeros_like(Y), Y, Z}; }
            case 2: { auto Y=std::get<1>(collPts_.boundary())[0];
                      auto Z=std::get<1>(collPts_.boundary())[1];
                      return {torch::ones_like(Y), Y, Z}; }
            case 3: { auto X=std::get<2>(collPts_.boundary())[0];
                      auto Z=std::get<2>(collPts_.boundary())[1];
                      return {X, torch::zeros_like(X), Z}; }
            case 4: { auto X=std::get<3>(collPts_.boundary())[0];
                      auto Z=std::get<3>(collPts_.boundary())[1];
                      return {X, torch::ones_like(X), Z}; }
            case 5: { auto X=std::get<4>(collPts_.boundary())[0];
                      auto Y=std::get<4>(collPts_.boundary())[1];
                      return {X, Y, torch::zeros_like(X)}; }
            case 6: { auto X=std::get<5>(collPts_.boundary())[0];
                      auto Y=std::get<5>(collPts_.boundary())[1];
                      return {X, Y, torch::ones_like(X)}; }
            default:
                throw std::invalid_argument("Boundary side must be 1..6.");
        }
    }

    /// @brief Masks points that lie on a second boundary side.
    torch::Tensor maskPointsOnOtherSide(const std::array<torch::Tensor,3>& pts,
                                         int otherSide) const
    {
        const auto& X=pts[0]; const auto& Y=pts[1]; const auto& Z=pts[2];
        switch (otherSide) {
            case 1: return torch::isclose(X, torch::zeros_like(X));
            case 2: return torch::isclose(X, torch::ones_like(X));
            case 3: return torch::isclose(Y, torch::zeros_like(Y));
            case 4: return torch::isclose(Y, torch::ones_like(Y));
            case 5: return torch::isclose(Z, torch::zeros_like(Z));
            case 6: return torch::isclose(Z, torch::ones_like(Z));
            default:
                throw std::invalid_argument("Boundary side must be 1..6.");
        }
    }

    /// @brief Keeps only boundary points owned by the given side after corner arbitration.
    torch::Tensor buildKeepMaskForSide(int sideNr) const {
        auto pts = getFaceBoundaryPoints(sideNr);
        torch::Tensor keepMask = torch::ones(
            {pts[0].size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(pts[0].device()));

        for (int other = 1; other <= 6; ++other) {
            if (!sidesIntersect(sideNr, other)) continue;
            if (!bc_other_wins(other, sideNr))  continue;
            keepMask = torch::logical_and(
                keepMask, torch::logical_not(maskPointsOnOtherSide(pts, other)));
        }
        return keepMask;
    }

    /// @brief Initializes cached traction collocation points for Neumann-type boundaries.
    void initTractionCollPts(const std::vector<int>& neumannSides,
                              const torch::TensorOptions& opts)
    {
        std::vector<torch::Tensor> xV, yV, zV;
        nPtsPerSide_.clear();

        auto make_face_points = [&](int side) -> std::array<torch::Tensor, 3> {
            return getFaceBoundaryPoints(side);
        };

        for (int side : neumannSides) {
            auto facePts = make_face_points(side);
            auto keepMask = buildKeepMaskForSide(side);
            auto idx = torch::nonzero(keepMask).reshape({-1});

            auto Xf = facePts[0].index_select(0, idx);
            auto Yf = facePts[1].index_select(0, idx);
            auto Zf = facePts[2].index_select(0, idx);

            nPtsPerSide_.push_back(static_cast<int>(Xf.size(0)));
            if (Xf.size(0) > 0) {
                xV.push_back(Xf); yV.push_back(Yf); zV.push_back(Zf);
            }
        }

        if (!xV.empty()) {
            tractionCollPts_ = {
                torch::cat(xV, 0),
                torch::cat(yV, 0),
                torch::cat(zV, 0)};
        } else {
            tractionCollPts_ = {
                torch::empty({0}, opts),
                torch::empty({0}, opts),
                torch::empty({0}, opts)};
        }

        // Precompute indices for boundary evaluations.
        var_knot_indices_boundary_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                tractionCollPts_);
        var_coeff_indices_boundary_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_boundary_);
        G_knot_indices_boundary_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                tractionCollPts_);
        G_coeff_indices_boundary_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_boundary_);

        tractionPtsInitialized_ = true;
    }


public:
    /// @brief Constructs the 3D linear-elasticity network wrapper.
    template <typename... Args>
    linear_elasticity_3d_impl(double lambda, double mu, bool SUPERVISED_LEARNING,
                      int MAX_EPOCH, double MIN_LOSS,
                      std::array<double, 3> BODY_FORCE,
                      std::vector<int> TFBC_SIDES,
                      std::vector<std::tuple<int,double,double,double>> FORCE_SIDES,
                      std::vector<std::tuple<int,double,double,double>> DIRI_SIDES,
                      int64_t NR_CTRL_PTS, std::string JSON_PATH,
                      std::vector<int64_t>&& layers,
                      std::vector<std::vector<std::any>>&& activations,
                      Args&&... args)
        : Base(std::forward<std::vector<int64_t>>(layers),
               std::forward<std::vector<std::vector<std::any>>>(activations),
               std::forward<Args>(args)...)
        , lambda_(lambda), mu_(mu)
        , ref_(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS, NR_CTRL_PTS))
        , MAX_EPOCH_(MAX_EPOCH), MIN_LOSS_(MIN_LOSS), NR_CTRL_PTS_(NR_CTRL_PTS)
        , BODY_FORCE_(BODY_FORCE)
        , DIRI_SIDES_(DIRI_SIDES), FORCE_SIDES_(FORCE_SIDES), TFBC_SIDES_(TFBC_SIDES)
        , JSON_PATH_(std::move(JSON_PATH))
        , SUPERVISED_LEARNING_(SUPERVISED_LEARNING)
    {}

    /// @brief Returns the reference displacement field.
    auto const& ref() const { return ref_; }
    /// @brief Returns the mutable reference displacement field.
    auto&       ref()       { return ref_; }

    /// @brief Writes one result entry into the configured JSON output file.
    void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
        ::appendToJsonFile(JSON_PATH_, key, data);
    }

    /// @brief Precomputes collocation points and knot/coefficient index caches.
    void initialize_problem_data() {
        Base::inputs(0);
        collPts_         = Base::template collPts<0>(iganet::collPts::greville);
        interiorCollPts_ = Base::template collPts<0>(iganet::collPts::greville_interior);

        nrCollPts_ = static_cast<int>(
            std::cbrt(static_cast<double>(collPts_.interior()[0].size(0))));

        torch::Tensor collPtsCoeffs =
            collPts_.interior()[0].slice(0, 0, nrCollPts_);
        nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
        for (int i = 0; i < collPtsCoeffs.size(0); ++i)
            collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
        appendToJsonFile("net_collPtsCoeffsRef1", collPtsCoeffs_j);
        appendToJsonFile("net_nrCollPtsRef1", {nrCollPts_});

        var_knot_indices_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                collPts_.interior());
        var_coeff_indices_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_);

        var_knot_indices_interior_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                interiorCollPts_.interior());
        var_coeff_indices_interior_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_interior_);

        G_knot_indices_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                collPts_.interior());
        G_coeff_indices_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_);

        G_knot_indices_interior_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                interiorCollPts_.interior());
        G_coeff_indices_interior_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_interior_);

        if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty()) {
            std::vector<int> neumannSides;
            neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
            for (const auto& f : FORCE_SIDES_)
                neumannSides.push_back(std::get<0>(f));

            auto dummyOpts = torch::TensorOptions().dtype(torch::kDouble);
            initTractionCollPts(neumannSides, dummyOpts);
        }
    }

    /// @brief Epoch callback used for lightweight logging.
    bool epoch(int64_t epoch) override {
        std::cout << "Epoch: " << epoch << std::endl;
        return epoch == 0;
    }

    /// @brief Computes the training loss for either supervised or unsupervised mode.
    torch::Tensor loss(const torch::Tensor& outputs, int64_t epoch) override {

        this->template output<0>().from_tensor(outputs);

        torch::Tensor totalLoss;
        torch::Tensor elastLoss;
        std::optional<torch::Tensor> bcLoss, tfbcLoss, supLoss, forceLoss;
        std::optional<torch::Tensor> forceValues, targetForce;
        std::optional<torch::Tensor> tractionFreeValues, tractionZeros;

        // Traction / Neumann boundary conditions.
      
        if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty()) {

            std::vector<int> neumannSides;
            neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
            for (const auto& f : FORCE_SIDES_)
                neumannSides.push_back(std::get<0>(f));

            if (!tractionPtsInitialized_)
                initTractionCollPts(neumannSides, outputs.options());

            if (tractionCollPts_[0].numel() > 0) {
                auto jacobianBoundary = this->template output<0>().ijac(
                    this->template input<0>(), tractionCollPts_,
                    var_knot_indices_boundary_, var_coeff_indices_boundary_,
                    G_knot_indices_boundary_,   G_coeff_indices_boundary_);

                auto ux_x = *jacobianBoundary[0]; auto ux_y = *jacobianBoundary[1];
                auto ux_z = *jacobianBoundary[2]; auto uy_x = *jacobianBoundary[3];
                auto uy_y = *jacobianBoundary[4]; auto uy_z = *jacobianBoundary[5];
                auto uz_x = *jacobianBoundary[6]; auto uz_y = *jacobianBoundary[7];
                auto uz_z = *jacobianBoundary[8];

                const int64_t nTrac = tractionCollPts_[0].size(0);
                torch::Tensor tvX = torch::zeros({nTrac}, ux_x.options());
                torch::Tensor tvY = torch::zeros({nTrac}, ux_x.options());
                torch::Tensor tvZ = torch::zeros({nTrac}, ux_x.options());

                int pointCtr = 0;
                int sideCtr = 0;
                for (int side : neumannSides) {
                    const int n = nPtsPerSide_[sideCtr];
                    if (n == 0) {
                        ++sideCtr;
                        continue;
                    }

                    auto xSlice = tvX.slice(0, pointCtr, pointCtr + n);
                    auto ySlice = tvY.slice(0, pointCtr, pointCtr + n);
                    auto zSlice = tvZ.slice(0, pointCtr, pointCtr + n);

                    auto ux_x_s = ux_x.slice(0, pointCtr, pointCtr + n);
                    auto ux_y_s = ux_y.slice(0, pointCtr, pointCtr + n);
                    auto ux_z_s = ux_z.slice(0, pointCtr, pointCtr + n);
                    auto uy_x_s = uy_x.slice(0, pointCtr, pointCtr + n);
                    auto uy_y_s = uy_y.slice(0, pointCtr, pointCtr + n);
                    auto uy_z_s = uy_z.slice(0, pointCtr, pointCtr + n);
                    auto uz_x_s = uz_x.slice(0, pointCtr, pointCtr + n);
                    auto uz_y_s = uz_y.slice(0, pointCtr, pointCtr + n);
                    auto uz_z_s = uz_z.slice(0, pointCtr, pointCtr + n);

                    if (side == 1) {
                        xSlice.copy_(-((lambda_ + 2. * mu_) * ux_x_s + lambda_ * uy_y_s + lambda_ * uz_z_s));
                        ySlice.copy_(-(mu_ * (ux_y_s + uy_x_s)));
                        zSlice.copy_(-(mu_ * (ux_z_s + uz_x_s)));
                    } else if (side == 2) {
                        xSlice.copy_((lambda_ + 2. * mu_) * ux_x_s + lambda_ * uy_y_s + lambda_ * uz_z_s);
                        ySlice.copy_(mu_ * (ux_y_s + uy_x_s));
                        zSlice.copy_(mu_ * (ux_z_s + uz_x_s));
                    } else if (side == 3) {
                        xSlice.copy_(-(mu_ * (ux_y_s + uy_x_s)));
                        ySlice.copy_(-(lambda_ * ux_x_s + (lambda_ + 2. * mu_) * uy_y_s + lambda_ * uz_z_s));
                        zSlice.copy_(-(mu_ * (uy_z_s + uz_y_s)));
                    } else if (side == 4) {
                        xSlice.copy_(mu_ * (ux_y_s + uy_x_s));
                        ySlice.copy_(lambda_ * ux_x_s + (lambda_ + 2. * mu_) * uy_y_s + lambda_ * uz_z_s);
                        zSlice.copy_(mu_ * (uy_z_s + uz_y_s));
                    } else if (side == 5) {
                        xSlice.copy_(-(mu_ * (ux_z_s + uz_x_s)));
                        ySlice.copy_(-(mu_ * (uy_z_s + uz_y_s)));
                        zSlice.copy_(-(lambda_ * ux_x_s + lambda_ * uy_y_s + (lambda_ + 2. * mu_) * uz_z_s));
                    } else if (side == 6) {
                        xSlice.copy_(mu_ * (ux_z_s + uz_x_s));
                        ySlice.copy_(mu_ * (uy_z_s + uz_y_s));
                        zSlice.copy_(lambda_ * ux_x_s + lambda_ * uy_y_s + (lambda_ + 2. * mu_) * uz_z_s);
                    } else {
                        throw std::invalid_argument("Side for 3D traction BC has to be 1..6.");
                    }

                    pointCtr += n;
                    ++sideCtr;
                }

                torch::Tensor tractionValues =
                    torch::stack({tvX, tvY, tvZ}, 1);

                if (!FORCE_SIDES_.empty()) {
                    int cutlength = 0;
                    int forceSize = static_cast<int>(FORCE_SIDES_.size());
                    for (int i = static_cast<int>(nPtsPerSide_.size()) - forceSize;
                         i < static_cast<int>(nPtsPerSide_.size()); ++i)
                        cutlength += nPtsPerSide_[i];

                    tractionFreeValues.emplace(
                        tractionValues.slice(0, 0, tractionValues.size(0) - cutlength));
                    tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));

                    forceValues.emplace(
                        tractionValues.slice(0, tractionValues.size(0) - cutlength));
                    targetForce.emplace(torch::zeros_like(*forceValues));

                    int offset   = 0;
                    int startIdx = static_cast<int>(nPtsPerSide_.size()) - forceSize;
                    for (size_t i = 0; i < FORCE_SIDES_.size(); ++i) {
                        int rPts = nPtsPerSide_[startIdx + static_cast<int>(i)];
                        auto row = (*targetForce).slice(0, offset, offset + rPts);
                        row.slice(1,0,1).fill_(std::get<1>(FORCE_SIDES_[i]));
                        row.slice(1,1,2).fill_(std::get<2>(FORCE_SIDES_[i]));
                        row.slice(1,2,3).fill_(std::get<3>(FORCE_SIDES_[i]));
                        offset += rPts;
                    }
                } else {
                    tractionFreeValues.emplace(tractionValues);
                    tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
                }
            }
        }

        auto hessianColl = this->template output<0>().ihess(
            this->template input<0>(), interiorCollPts_.interior(),
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_,   G_coeff_indices_interior_);

        auto& ux_xx=hessianColl(0,0,0); auto& ux_yy=hessianColl(1,1,0); auto& ux_zz=hessianColl(2,2,0);
        auto& uy_xy=hessianColl(0,1,1); auto& uz_xz=hessianColl(0,2,2);
        auto& uy_xx=hessianColl(0,0,1); auto& uy_yy=hessianColl(1,1,1); auto& uy_zz=hessianColl(2,2,1);
        auto& ux_yx=hessianColl(1,0,0); auto& uz_yz=hessianColl(1,2,2);
        auto& uz_xx=hessianColl(0,0,2); auto& uz_yy=hessianColl(1,1,2); auto& uz_zz=hessianColl(2,2,2);
        auto& ux_zx=hessianColl(2,0,0); auto& uy_zy=hessianColl(2,1,1);

        auto opts = hessianColl(0,0,0).options();

        torch::Tensor divStressX =
            (lambda_ + 2. * mu_) * ux_xx + mu_ * ux_yy + mu_ * ux_zz
            + (lambda_ + mu_) * (uy_xy + uz_xz);
        torch::Tensor divStressY =
            mu_ * uy_xx + (lambda_ + 2. * mu_) * uy_yy + mu_ * uy_zz
            + (lambda_ + mu_) * (ux_yx + uz_yz);
        torch::Tensor divStressZ =
            mu_ * uz_xx + mu_ * uz_yy + (lambda_ + 2. * mu_) * uz_zz
            + (lambda_ + mu_) * (ux_zx + uy_zy);

        torch::Tensor divStress = torch::stack({divStressX, divStressY, divStressZ}, 1);

        torch::Tensor bodyForce = torch::tensor(
            {BODY_FORCE_[0], BODY_FORCE_[1], BODY_FORCE_[2]}, opts)
            .view({1,3}).repeat({divStress.size(0), 1});

 
        auto masked_side_loss = [&](const torch::Tensor& u0, const torch::Tensor& u1,
                                    const torch::Tensor& u2, const torch::Tensor& b0,
                                    const torch::Tensor& b1, const torch::Tensor& b2,
                                    int sideNr) -> torch::Tensor {
            auto keepMask = buildKeepMaskForSide(sideNr);
            auto keepIdx  = torch::nonzero(keepMask).reshape({-1});
            if (keepIdx.numel() == 0) return torch::zeros({}, outputs.options());
            return torch::mse_loss(u0.index_select(0,keepIdx), b0.index_select(0,keepIdx))
                 + torch::mse_loss(u1.index_select(0,keepIdx), b1.index_select(0,keepIdx))
                 + torch::mse_loss(u2.index_select(0,keepIdx), b2.index_select(0,keepIdx));
        };

        auto add_masked_side_loss = [&](const auto& u_side, const auto& b_side, int sNr) {
            *bcLoss += static_cast<double>(SUPERVISED_LEARNING_ ? 1 : 100000)
                * masked_side_loss(*u_side[0],*u_side[1],*u_side[2],
                                   *b_side[0],*b_side[1],*b_side[2], sNr);
        };

        // Unsupervised learning.
  
        if (!SUPERVISED_LEARNING_) {
            std::ostringstream log;
            elastLoss = torch::mse_loss(divStress, -bodyForce);
            totalLoss = elastLoss;
            log << "EL " << std::setw(11) << elastLoss.item<double>();

            if (!TFBC_SIDES_.empty()) {
                tfbcLoss  = torch::mse_loss(*tractionFreeValues, *tractionZeros);
                totalLoss += *tfbcLoss;
                log << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
            }
            if (!FORCE_SIDES_.empty()) {
                forceLoss  = torch::mse_loss(*forceValues, *targetForce);
                totalLoss += *forceLoss;
                log << " + FL " << std::setw(11) << (*forceLoss).item<double>();
            }
            if (!DIRI_SIDES_.empty()) {
                const double bcWeight = 1e5;
                bcLoss = torch::tensor(0.0, outputs.options());
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.boundary());
                auto bdr   = ref_.template eval<iganet::functionspace::boundary>(collPts_.boundary());
                for (const auto& side : DIRI_SIDES_) {
                    int sNr = std::get<0>(side);
                    switch (sNr) {
                        case 1: add_masked_side_loss(std::get<0>(u_bdr),std::get<0>(bdr),1); break;
                        case 2: add_masked_side_loss(std::get<1>(u_bdr),std::get<1>(bdr),2); break;
                        case 3: add_masked_side_loss(std::get<2>(u_bdr),std::get<2>(bdr),3); break;
                        case 4: add_masked_side_loss(std::get<3>(u_bdr),std::get<3>(bdr),4); break;
                        case 5: add_masked_side_loss(std::get<4>(u_bdr),std::get<4>(bdr),5); break;
                        case 6: add_masked_side_loss(std::get<5>(u_bdr),std::get<5>(bdr),6); break;
                        default: std::cerr << "Invalid Dirichlet side!\n";
                    }
                }
                totalLoss += *bcLoss;
                log << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight
                    << " * 1e" << static_cast<int>(std::log10(bcWeight));
            }
            std::cout << std::setw(11) << totalLoss.item<double>()
                      << " = " << log.str() << std::endl;
        }

        // Supervised learning.

        else if (SUPERVISED_LEARNING_) {
            std::ostringstream log;

            torch::Tensor netDisp = torch::stack({
                outputs.slice(0, 0,              outputs.size(0)/3),
                outputs.slice(0, outputs.size(0)/3,   2*outputs.size(0)/3),
                outputs.slice(0, 2*outputs.size(0)/3, outputs.size(0))}, 1);

            if (!stdDisplacements_.has_value()) {
                stdDisplacements_ = loadDisplacements(JSON_PATH_).to(netDisp.options());
            }
            torch::Tensor stdDisp = *stdDisplacements_;

            const double supWeight = 1e7;
            supLoss   = supWeight * torch::mse_loss(netDisp, stdDisp);
            elastLoss = torch::mse_loss(divStress, -bodyForce);
            totalLoss = *supLoss + elastLoss;

            log << "SL " << std::setw(11) << (*supLoss).item<double>() / supWeight
                << " * 1e" << static_cast<int>(std::log10(supWeight))
                << " + EL " << std::setw(11) << elastLoss.item<double>();

            if (!TFBC_SIDES_.empty()) {
                tfbcLoss  = torch::mse_loss(*tractionFreeValues, *tractionZeros);
                totalLoss += *tfbcLoss;
                log << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
            }
            if (!FORCE_SIDES_.empty()) {
                forceLoss  = torch::mse_loss(*forceValues, *targetForce);
                totalLoss += *forceLoss;
                log << " + FL " << std::setw(11) << (*forceLoss).item<double>();
            }
            if (!DIRI_SIDES_.empty()) {
                const double bcWeight = 1e0;
                bcLoss = torch::tensor(0.0, outputs.options());
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.boundary());
                auto bdr   = ref_.template eval<iganet::functionspace::boundary>(collPts_.boundary());
                for (const auto& side : DIRI_SIDES_) {
                    int sNr = std::get<0>(side);
                    switch (sNr) {
                        case 1: add_masked_side_loss(std::get<0>(u_bdr),std::get<0>(bdr),1); break;
                        case 2: add_masked_side_loss(std::get<1>(u_bdr),std::get<1>(bdr),2); break;
                        case 3: add_masked_side_loss(std::get<2>(u_bdr),std::get<2>(bdr),3); break;
                        case 4: add_masked_side_loss(std::get<3>(u_bdr),std::get<3>(bdr),4); break;
                        case 5: add_masked_side_loss(std::get<4>(u_bdr),std::get<4>(bdr),5); break;
                        case 6: add_masked_side_loss(std::get<5>(u_bdr),std::get<5>(bdr),6); break;
                        default: std::cerr << "Invalid Dirichlet side!\n";
                    }
                }
                totalLoss += *bcLoss;
                log << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight
                    << " * 1e" << static_cast<int>(std::log10(bcWeight));
            }
            std::cout << std::setw(11) << totalLoss.item<double>()
                      << " = " << log.str() << std::endl;
        } else {
            throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_");
        }

        return totalLoss;
    }

    /// @brief Exports derived stresses, displaced collocation points, and residual fields.
    void PostProc() {

        // Jacobian at all collocation points.
        auto jacobian = this->template output<0>().ijac(
            this->template input<0>(), collPts_.interior(),
            var_knot_indices_,  var_coeff_indices_,
            G_knot_indices_,    G_coeff_indices_);

        auto ux_x=*jacobian[0]; auto ux_y=*jacobian[1]; auto ux_z=*jacobian[2];
        auto uy_x=*jacobian[3]; auto uy_y=*jacobian[4]; auto uy_z=*jacobian[5];
        auto uz_x=*jacobian[6]; auto uz_y=*jacobian[7]; auto uz_z=*jacobian[8];

        const int64_t nPts = jacobian[0]->size(0);

        // Stress tensor components.
        torch::Tensor sigma_xx = torch::zeros({nPts});
        torch::Tensor sigma_xy = torch::zeros({nPts});
        torch::Tensor sigma_xz = torch::zeros({nPts});
        torch::Tensor sigma_yy = torch::zeros({nPts});
        torch::Tensor sigma_yz = torch::zeros({nPts});
        torch::Tensor sigma_zz = torch::zeros({nPts});
        torch::Tensor sigma_vm = torch::zeros({nPts});

        nlohmann::json netVmStresses_j = nlohmann::json::array();
        nlohmann::json netXStresses_j  = nlohmann::json::array();
        nlohmann::json netYStresses_j  = nlohmann::json::array();
        nlohmann::json netZStresses_j  = nlohmann::json::array();

        for (int i = 0; i < nPts; ++i) {
            // Hooke's law for isotropic linear elasticity.
            sigma_xx[i] = lambda_*(ux_x[i]+uy_y[i]+uz_z[i]) + 2.*mu_*ux_x[i];
            sigma_xy[i] = mu_*(uy_x[i]+ux_y[i]);
            sigma_xz[i] = mu_*(uz_x[i]+ux_z[i]);
            sigma_yy[i] = lambda_*(ux_x[i]+uy_y[i]+uz_z[i]) + 2.*mu_*uy_y[i];
            sigma_yz[i] = mu_*(uz_y[i]+uy_z[i]);
            sigma_zz[i] = lambda_*(ux_x[i]+uy_y[i]+uz_z[i]) + 2.*mu_*uz_z[i];

            // Von Mises equivalent stress.
            sigma_vm[i] = sqrt(0.5*(
                (sigma_xx[i]-sigma_yy[i])*(sigma_xx[i]-sigma_yy[i]) +
                (sigma_yy[i]-sigma_zz[i])*(sigma_yy[i]-sigma_zz[i]) +
                (sigma_zz[i]-sigma_xx[i])*(sigma_zz[i]-sigma_xx[i]) +
                6.*(sigma_xy[i]*sigma_xy[i]+sigma_yz[i]*sigma_yz[i]+sigma_xz[i]*sigma_xz[i])));

            netVmStresses_j.push_back({sigma_vm[i].item<double>()});
            netXStresses_j.push_back( {sigma_xx[i].item<double>()});
            netYStresses_j.push_back( {sigma_yy[i].item<double>()});
            netZStresses_j.push_back( {sigma_zz[i].item<double>()});
        }

        appendToJsonFile("net_VmStresses", netVmStresses_j);
        appendToJsonFile("net_XStresses",  netXStresses_j);
        appendToJsonFile("net_YStresses",  netYStresses_j);
        appendToJsonFile("net_ZStresses",  netZStresses_j);

        // Collocation points: reference and deformed positions.
        torch::Tensor cpRef = torch::stack(
            {std::get<0>(collPts_.interior()),
             std::get<1>(collPts_.interior()),
             std::get<2>(collPts_.interior())}, 1);
        auto displ = this->template output<0>().eval(collPts_.interior());
        torch::Tensor cpDispl = torch::stack({*displ[0],*displ[1],*displ[2]}, 1);

        nlohmann::json collPtsFirst_j      = nlohmann::json::array();
        nlohmann::json collPtsFirstDispl_j = nlohmann::json::array();
        for (int i = 0; i < cpRef.size(0); ++i) {
            collPtsFirst_j.push_back({
                cpRef[i][0].item<double>(),
                cpRef[i][1].item<double>(),
                cpRef[i][2].item<double>()});
            collPtsFirstDispl_j.push_back({
                cpRef[i][0].item<double>() + cpDispl[i][0].item<double>(),
                cpRef[i][1].item<double>() + cpDispl[i][1].item<double>(),
                cpRef[i][2].item<double>() + cpDispl[i][2].item<double>()});
        }
        appendToJsonFile("net_collPtsFirstAsTensor",
                         collPtsFirst_j);
        appendToJsonFile("net_collPtsFirstAfterDisplacementAsTensor",
                         collPtsFirstDispl_j);

        // Stress divergence for residual analysis.
        auto hessianColl = this->template output<0>().ihess(
            this->template input<0>(), interiorCollPts_.interior(),
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_,   G_coeff_indices_interior_);

        auto& dux_xx=hessianColl(0,0,0); auto& dux_yy=hessianColl(1,1,0); auto& dux_zz=hessianColl(2,2,0);
        auto& duy_xy=hessianColl(0,1,1); auto& duz_xz=hessianColl(0,2,2);
        auto& duy_xx=hessianColl(0,0,1); auto& duy_yy=hessianColl(1,1,1); auto& duy_zz=hessianColl(2,2,1);
        auto& dux_yx=hessianColl(1,0,0); auto& duz_yz=hessianColl(1,2,2);
        auto& duz_xx=hessianColl(0,0,2); auto& duz_yy=hessianColl(1,1,2); auto& duz_zz=hessianColl(2,2,2);
        auto& dux_zx=hessianColl(2,0,0); auto& duy_zy=hessianColl(2,1,1);

        const int64_t szInner = hessianColl(0,0,0).size(0);
        auto optsInner = hessianColl(0,0,0).options();
        torch::Tensor divX = torch::zeros({szInner}, optsInner);
        torch::Tensor divY = torch::zeros({szInner}, optsInner);
        torch::Tensor divZ = torch::zeros({szInner}, optsInner);

        for (int i = 0; i < szInner; ++i) {
            divX[i] = (lambda_+2.*mu_)*dux_xx[i]+mu_*dux_yy[i]+mu_*dux_zz[i]+(lambda_+mu_)*(duy_xy[i]+duz_xz[i]);
            divY[i] = mu_*duy_xx[i]+(lambda_+2.*mu_)*duy_yy[i]+mu_*duy_zz[i]+(lambda_+mu_)*(dux_yx[i]+duz_yz[i]);
            divZ[i] = mu_*duz_xx[i]+mu_*duz_yy[i]+(lambda_+2.*mu_)*duz_zz[i]+(lambda_+mu_)*(dux_zx[i]+duy_zy[i]);
        }

        nlohmann::json divX_j = nlohmann::json::array();
        nlohmann::json divY_j = nlohmann::json::array();
        nlohmann::json divZ_j = nlohmann::json::array();
        for (int i = 0; i < szInner; ++i) {
            divX_j.push_back({divX[i].item<double>()});
            divY_j.push_back({divY[i].item<double>()});
            divZ_j.push_back({divZ[i].item<double>()});
        }
        appendToJsonFile("net_DivergenceX", divX_j);
        appendToJsonFile("net_DivergenceY", divY_j);
        appendToJsonFile("net_DivergenceZ", divZ_j);
    }


#ifdef IGANET_WITH_GISMO
    /// @brief Runs the original GISMO-based 2D reference simulation.
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RunGismoSimulation(
        int64_t NR_CTRL_PTS, int DEGREE, double YOUNG_MODULUS, double POISSON_RATIO,
        const std::vector<std::tuple<int,double,double>>& DIRI_SIDES,
        const std::vector<std::tuple<int,double,double>>& FORCE_SIDES,
        const std::pair<double,double>& BODY_FORCE)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
        const int64_t nPts = NR_CTRL_PTS * NR_CTRL_PTS;
        torch::Tensor ctrlPts       = torch::empty({nPts,2}, opts);
        torch::Tensor displacements = torch::empty({nPts,2}, opts);
        torch::Tensor stresses      = torch::empty({nPts,1}, opts);
        auto ctrlA = ctrlPts.accessor<double,2>();
        auto dispA = displacements.accessor<double,2>();
        auto strA  = stresses.accessor<double,2>();
        gismo::gsMatrix<double> ctrlPts_gs(nPts, 2);
        gismo::gsKnotVector<double> kv_u(0.,1., NR_CTRL_PTS-DEGREE-1, DEGREE+1);
        gismo::gsKnotVector<double> kv_v(0.,1., NR_CTRL_PTS-DEGREE-1, DEGREE+1);
        std::vector<double> gU = computeGrevilleAbscissae(
            std::vector<double>(kv_u.begin(),kv_u.end()), DEGREE, NR_CTRL_PTS);
        std::vector<double> gV = computeGrevilleAbscissae(
            std::vector<double>(kv_v.begin(),kv_v.end()), DEGREE, NR_CTRL_PTS);
        int64_t idx = 0;
        for (int j=0; j<NR_CTRL_PTS; ++j) for (int i=0; i<NR_CTRL_PTS; ++i) {
            ctrlA[idx][0]=gU[i]; ctrlA[idx][1]=gV[j];
            ctrlPts_gs(idx,0)=gU[i]; ctrlPts_gs(idx,1)=gV[j]; ++idx;
        }
        gismo::gsTensorBSpline<2,double> geometry(kv_u, kv_v, ctrlPts_gs);
        gismo::gsMultiPatch<double> mp; mp.addPatch(geometry);
        gismo::gsMultiBasis<> basis(mp);
        auto getGsSide = [](int s) -> gismo::boundary::side {
            switch(s){
                case 1: return gismo::boundary::west;
                case 2: return gismo::boundary::east;
                case 3: return gismo::boundary::south;
                case 4: return gismo::boundary::north;
                default: throw std::invalid_argument("Invalid side (must be 1..4)");
            }
        };
        gismo::gsBoundaryConditions<double> bcInfo;
        for (const auto& d : DIRI_SIDES) {
            auto gs = getGsSide(std::get<0>(d));
            bcInfo.addCondition(0, gs, gismo::condition_type::dirichlet,
                gismo::gsConstantFunction<double>(std::get<1>(d),2), 0);
            bcInfo.addCondition(0, gs, gismo::condition_type::dirichlet,
                gismo::gsConstantFunction<double>(std::get<2>(d),2), 1);
        }
        for (const auto& f : FORCE_SIDES) {
            auto gs = getGsSide(std::get<0>(f));
            gismo::gsFunctionExpr<> t(std::to_string(std::get<1>(f)),
                                      std::to_string(std::get<2>(f)), 2);
            bcInfo.addCondition(0, gs, gismo::condition_type::neumann, t);
        }
        gismo::gsConstantFunction<double> bf(BODY_FORCE.first, BODY_FORCE.second, 2);
        gismo::gsElasticityAssembler<double> asm_(geometry, basis, bcInfo, bf);
        asm_.options().setReal("YoungsModulus",  YOUNG_MODULUS);
        asm_.options().setReal("PoissonsRatio",  POISSON_RATIO);
        asm_.assemble();
        gismo::gsSparseSolver<>::CGDiagonal solver;
        gismo::gsMatrix<double> sol;
        solver.compute(asm_.matrix()); sol = solver.solve(asm_.rhs());
        gismo::gsMultiPatch<double> solPatch;
        asm_.constructSolution(sol, asm_.allFixedDofs(), solPatch);
        gismo::gsPiecewiseFunction<double> stressFn;
        asm_.constructCauchyStresses(solPatch, stressFn, gismo::stress_components::von_mises);
        for (int i=0; i<ctrlPts_gs.rows(); ++i) {
            gismo::gsMatrix<double> pt(2,1);
            pt(0,0)=ctrlPts_gs(i,0); pt(1,0)=ctrlPts_gs(i,1);
            auto u = solPatch.patch(0).eval(pt);
            dispA[i][0]=u(0); dispA[i][1]=u(1);
            gismo::gsMatrix<double> s(1,1);
            stressFn.piece(0).eval_into(pt, s);
            strA[i][0]=s(0,0);
        }
        return {ctrlPts, displacements, stresses};
    }
#endif
}; // class linear_elasticity_3d_impl

// -----------------------------------------------------------------------------
// Public dispatcher
// -----------------------------------------------------------------------------

template <typename Optimizer, typename GeometryMap, typename Variable,
          int GeoDim = linear_elasticity_geometry_patch_t<GeometryMap>::geoDim()>
struct linear_elasticity_selector;

template <typename Optimizer, typename GeometryMap, typename Variable>
struct linear_elasticity_selector<Optimizer, GeometryMap, Variable, 2> {
    using type = linear_elasticity_2d_impl<Optimizer, GeometryMap, Variable>;
};

template <typename Optimizer, typename GeometryMap, typename Variable>
struct linear_elasticity_selector<Optimizer, GeometryMap, Variable, 3> {
    using type = linear_elasticity_3d_impl<Optimizer, GeometryMap, Variable>;
};

template <typename Optimizer, typename GeometryMap, typename Variable>
using linear_elasticity =
    typename linear_elasticity_selector<Optimizer, GeometryMap, Variable>::type;
