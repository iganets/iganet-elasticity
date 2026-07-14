/*
 * Example: 2D single-patch linear elasticity with collocation-based IgANet.
 *
 * This file contains a fairly explicit implementation of the full workflow:
 *   1. read a configuration file,
 *   2. build a geometry spline and a displacement spline,
 *   3. train the neural network representation of the displacement field,
 *   4. evaluate stresses and deformed control points,
 *   5. write all relevant data to a JSON result file.
 *
 * Compared to the optimized examples, this file keeps more logic locally.
 * That makes it useful as a reference when one wants to understand the
 * complete single-patch 2D pipeline without jumping through too many helpers.
 */

#include <iganet.h>
#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <optional>

using namespace iganet::literals;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using iganet_elasticity::utils::config::require;
using optimizer_config_t = iganet_elasticity::utils::config::optimizer_config;
using optimizer_type_t = iganet_elasticity::utils::config::optimizer_type;

// -----------------------------------------------------------------------------
// Network wrapper used by this example
// -----------------------------------------------------------------------------

/// @brief Specialization of the IgANet class for linear elasticity in 2D
template <typename Optimizer, typename GeometryMap, typename Variable>
class linear_elasticity
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
    linear_elasticity(double lambda, double mu, bool SUPERVISED_LEARNING, int MAX_EPOCH, 
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
            // The first epoch is used as a setup stage. We create the fixed
            // collocation sets once and cache geometry-dependent quantities
            // that will be reused in every later loss evaluation.
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

        // Interpret the neural-network output as the current coefficient vector
        // of the displacement spline. After this assignment, all PDE terms are
        // evaluated by standard spline differentiation.
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

        // -----------------------------------------------------------------
        // Boundary contribution: traction-free and prescribed-force sides
        // -----------------------------------------------------------------
    
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

        // -----------------------------------------------------------------
        // Interior contribution: strong form of linear elasticity
        // -----------------------------------------------------------------

        // First compute derivatives in the parametric coordinates of the
        // spline, then map them to physical space through the geometry map.
        const auto gradUxi = evaluateParametricGradient(interiorResidualCache_);
        const auto gradU = torch::matmul(gradUxi, interiorResidualCache_.invJ);
        const auto hessU = evaluatePhysicalHessians(interiorResidualCache_, gradU);

        // hessU[0] contains second derivatives of the x-displacement u_x,
        // hessU[1] contains second derivatives of the y-displacement u_y.
        // We extract only the combinations that are needed by div(sigma).
        const auto ux_xx = hessU[0].index({torch::indexing::Slice(), 0, 0});
        const auto ux_xy = hessU[0].index({torch::indexing::Slice(), 0, 1});
        const auto ux_yy = hessU[0].index({torch::indexing::Slice(), 1, 1});

        const auto uy_xx = hessU[1].index({torch::indexing::Slice(), 0, 0});
        const auto uy_xy = hessU[1].index({torch::indexing::Slice(), 0, 1});
        const auto uy_yy = hessU[1].index({torch::indexing::Slice(), 1, 1});

        // pre-allocation of the results
        torch::Tensor divStressX = torch::zeros({interiorResidualCache_.eval.numeval}, gradU.options());
        torch::Tensor divStressY = torch::zeros({interiorResidualCache_.eval.numeval}, gradU.options());

        // These formulas are the 2D isotropic linear-elasticity operator in
        // strong form. They produce the two components of div(sigma(u)).
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

        // The body force is constant in this example, so we replicate one
        // 2-vector to all interior collocation points.
        torch::Tensor bodyForce = torch::tensor(
            {BODY_FORCE_.first, BODY_FORCE_.second},
            opts
        ).view({1, 2}).repeat({divStress.size(0), 1});   // (N,2)

        // -----------------------------------------------------------------
        // Loss assembly
        //
        // Unsupervised mode uses only physics and boundary data.
        // Supervised mode additionally matches a standard collocation solution.
        // -----------------------------------------------------------------

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

            // Traction-free boundaries are enforced by driving sigma*n to zero.
            if (!TFBC_SIDES_.empty()) {
                tfbcLoss = torch::mse_loss(*tractionFreeValues, *tractionZeros);
                totalLoss += *tfbcLoss;
                singleLossOutput << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
            }

            // Prescribed traction boundaries are enforced by matching sigma*n
            // against the user-specified force values.
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

                // Evaluate both the predicted displacement and the prescribed
                // Dirichlet target on the same boundary collocation points.
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.boundary());
                // evaluation of the displacements at the reference boundary points
                auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.boundary());

                // Every configured side contributes its own boundary penalty.
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
        
            // Reorganize the flat network output into one displacement vector
            // per control point so it can be compared to the reference field.
            torch::Tensor modifiedOutputs = outputs * 1.0;
        
            // create netDisplacements_ from slices of modifiedOutputs
            torch::Tensor netDisplacements_ = torch::stack({
                modifiedOutputs.slice(0, 0, outputs.size(0) / 2),
                modifiedOutputs.slice(0, outputs.size(0) / 2, outputs.size(0)),
            }, 1);

            // load the displacements from the std collocation solution
            torch::Tensor stdCollDisplacements_ = loadDisplacements().to(netDisplacements_.options());

            // In supervised mode this term anchors the learned displacement
            // coefficients to a precomputed standard-collocation solution.
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

        // -----------------------------------------------------------------
        // Final export step
        //
        // At convergence or at the last epoch we write all post-processing
        // quantities required by the show scripts and debugging workflow.
        // -----------------------------------------------------------------

        // only calculate this at the end of the simulation
        if ((epoch == MAX_EPOCH_ - 1) || (totalLoss.item<double>() <= MIN_LOSS_)) {
            
            // -------------------------
            // Stress output on all collocation points
            // -------------------------

            const auto gradUxiColl = evaluateParametricGradient(collocationCache_);
            const auto gradUColl = torch::matmul(gradUxiColl, collocationCache_.invJ);

            const auto ux_x = gradUColl.index({torch::indexing::Slice(), 0, 0});
            const auto ux_y = gradUColl.index({torch::indexing::Slice(), 0, 1});
            const auto uy_x = gradUColl.index({torch::indexing::Slice(), 1, 0});
            const auto uy_y = gradUColl.index({torch::indexing::Slice(), 1, 1});

            // These arrays store the final field quantities that will be
            // written to JSON for later plotting.
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

            // Convert displacement gradients to Cauchy stress and, additionally,
            // compute a scalar von Mises stress and an apparent Poisson value.
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

            // -------------------------
            // Deformed collocation points
            // -------------------------

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

            // -------------------------
            // Residual diagnostics
            // -------------------------

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
};

int main() {
    iganet::init();
    iganet::verbose(std::cout);

    // resolve paths relative to repo root
    std::filesystem::path repo_root;
    try {
        repo_root = repo_root_from_build_exe();
    } catch (const std::exception& e) {
        std::cerr << "Could not determine repo root: " << e.what() << "\n";
        return 1;
    }

    const std::filesystem::path CONFIG_PATH =
        repo_root / "src" / "examples2D" / "singlePatch" / "sim_config_2D_single_patch.json";
    const std::filesystem::path RESULT_JSON_PATH =
        repo_root / "results" / "result_iganet_lin_elasticity_2D.json";  // output file

    // load config
    std::ifstream file(CONFIG_PATH);
    if (!file) {
        std::cerr << "Could not open config file: " << CONFIG_PATH << "\n";
        return 1;
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse config JSON: " << e.what() << "\n";
        return 1;
    }

    // material parameters
    double YOUNG_MODULUS = 0.0;
    double POISSON_RATIO = 0.0;

    // simulation parameters
    int MAX_EPOCH = 0;
    double MIN_LOSS = 0.0;
    bool SUPERVISED_LEARNING = false;
    std::string JSON_PATH;  // output json path
    optimizer_config_t OPTIMIZER_CFG;

    // reference simulation parameters
    bool RUN_GS_REF_SIM = false;
    bool RUN_COLL_REF_SIM = false;
    int NR_CTRL_PTS_REF = 0;
    int DEGREE_REF = 0;

    // spline parameters
    int64_t NR_CTRL_PTS = 0;
    int DEGREE_CFG = 0;

    // boundary conditions
    std::vector<std::tuple<int, double, double>> FORCE_SIDES;
    std::vector<std::tuple<int, double, double>> DIRI_SIDES;
    std::vector<int> TFBC_SIDES;

    // body force
    std::pair<double, double> BODY_FORCE{0.0, 0.0};

    try {
        // material
        YOUNG_MODULUS = require(j, "material.young_modulus").get<double>();
        POISSON_RATIO = require(j, "material.poisson_ratio").get<double>();

        // simulation
        MAX_EPOCH = require(j, "simulation.max_epoch").get<int>();
        MIN_LOSS = require(j, "simulation.min_loss").get<double>();
        SUPERVISED_LEARNING = require(j, "simulation.supervised_learning").get<bool>();
        OPTIMIZER_CFG = iganet_elasticity::utils::config::load_optimizer_config(j);

        // IMPORTANT: output json is fixed in results/
        JSON_PATH = RESULT_JSON_PATH.string();

        // spline
        const auto solutionSplineCfg =
            iganet_elasticity::utils::config::load_solution_spline_config(j);
        NR_CTRL_PTS = solutionSplineCfg.nr_ctrl_pts;
        DEGREE_CFG = solutionSplineCfg.degree;

        const auto patch_cfg =
            iganet_elasticity::utils::config::load_single_patch_config_2d(j);

        FORCE_SIDES.clear();
        for (const auto& bc : patch_cfg.force_sides) {
            FORCE_SIDES.emplace_back(bc.side, bc.x, bc.y);
        }

        DIRI_SIDES.clear();
        for (const auto& bc : patch_cfg.diri_sides) {
            DIRI_SIDES.emplace_back(bc.side, bc.x, bc.y);
        }

        TFBC_SIDES = patch_cfg.tfbc_sides;
        BODY_FORCE.first = patch_cfg.body_force[0];
        BODY_FORCE.second = patch_cfg.body_force[1];

        // reference simulation (only if present in config)
        if (j.contains("reference_simulation")) {
            RUN_GS_REF_SIM = require(j, "reference_simulation.run_gs_ref_sim").get<bool>();
            RUN_COLL_REF_SIM = require(j, "reference_simulation.run_coll_ref_sim").get<bool>();
            NR_CTRL_PTS_REF = require(j, "reference_simulation.nr_ctrl_pts_ref").get<int>();
            DEGREE_REF = require(j, "reference_simulation.degree_ref").get<int>();
        }

    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    if (RUN_COLL_REF_SIM) {
        const std::string cmd =
            "cd \"" + repo_root.string() + "\" && python3 -m std_collocation_python.run_std_coll \"" +
            CONFIG_PATH.string() + "\" \"" + RESULT_JSON_PATH.string() + "\"";

        const int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "ERROR: python reference run (std_collocation_python/run_std_coll.py) failed. system() returned " << ret << "\n";
            return 1;
        }
    }
        
    // calculation of lame parameters
    double lambda = (YOUNG_MODULUS * POISSON_RATIO) / 
                    ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
    double mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO));

    auto run = [&]<int DEGREE, typename optimizer_t>() -> int {
        using real_t = double;
        using namespace iganet::literals;
        using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
        using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
        using linear_elasticity_t = linear_elasticity<optimizer_t, geometry_t, variable_t>;

        linear_elasticity_t net(//simulation parameters 
            lambda, mu, SUPERVISED_LEARNING, MAX_EPOCH, MIN_LOSS, 
            BODY_FORCE, TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, NR_CTRL_PTS, JSON_PATH, 
            // Number of neurons per layer 
            {25, 25}, 
            // Activation functions 
            {{iganet::activation::sigmoid}, {iganet::activation::sigmoid}, {iganet::activation::none}}, 
            // Number of B-spline coefficients of the geometry 
            std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)), 
            // Number of B-spline coefficients of the variable 
            std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)) );

        // imposing body force
        net.template output<0>().transform([=](const std::array<real_t, 2> xi) {
            return std::array<real_t, 2>{BODY_FORCE.first, BODY_FORCE.second};
        });

        // get the coefficients of the control points
        torch::Tensor ctrlPtsCoeffs = net.template input<0>().as_tensor().slice(0, 0, NR_CTRL_PTS);
        nlohmann::json ctrlPtsCoeffs_j = nlohmann::json::array();
        for (int i = 0; i < NR_CTRL_PTS; ++i) {
            ctrlPtsCoeffs_j.push_back({ctrlPtsCoeffs[i].item<double>()});
        }
        net.appendToJsonFile("net_ctrlPtsCoeffs", ctrlPtsCoeffs_j);

        // run through all DIRI_SIDES
        for (const auto& side : DIRI_SIDES) {
            int sideNr = std::get<0>(side);
            double xDispl = std::get<1>(side);
            double yDispl = std::get<2>(side);

            switch (sideNr) {
                case 1:
                    net.ref().boundary().template side<1>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<1>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    break;
                case 2:
                    net.ref().boundary().template side<2>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<2>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    break;
                case 3:
                    net.ref().boundary().template side<3>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<3>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    break;
                case 4:
                    net.ref().boundary().template side<4>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<4>().template transform<1>(
                        [=](const std::array<real_t, 1> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    break;
                default:
                    std::cerr << "Error: Invalid side number " << sideNr << std::endl;
            }
        }

        // Set maximum number of epochs
        net.options().max_epoch(MAX_EPOCH);

        // Set tolerance for the loss functions
        net.options().min_loss(MIN_LOSS);

        // Start time measurement
        auto t1 = std::chrono::high_resolution_clock::now();

        // Train network
        net.train();

        // Stop time measurement
        auto t2 = std::chrono::high_resolution_clock::now();
        iganet::Log(iganet::log::info)
            << "Training took "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                    .count()
            << " seconds\n";

        // PROCESSING NETWORK OUTPUT FOR SPLINEPY

        // get the geometry and displacement as tensors
        torch::Tensor geometryAsTensor = net.template input<0>().as_tensor();
        torch::Tensor displacementAsTensor = net.template output<0>().as_tensor();
        
        // creating collection matrix for all the control points (iganet)
        torch::Tensor netCtrlPts = torch::zeros({NR_CTRL_PTS * NR_CTRL_PTS, 2});
        // creating collection matrix for all the displacements (iganet)
        torch::Tensor netDisplacements = torch::zeros({NR_CTRL_PTS * NR_CTRL_PTS, 2});

        // filling the collection matrices with the values from the tensors
        for (int i = 0; i < NR_CTRL_PTS * NR_CTRL_PTS; ++i) {
            double x = geometryAsTensor[i].item<double>();          
            double y = geometryAsTensor[i + NR_CTRL_PTS * NR_CTRL_PTS].item<double>();
            netCtrlPts[i][0] = x;
            netCtrlPts[i][1] = y;
                
            double ux = displacementAsTensor[i].item<double>();
            double uy = displacementAsTensor[i + NR_CTRL_PTS * NR_CTRL_PTS].item<double>();
            netDisplacements[i][0] = ux;
            netDisplacements[i][1] = uy;
        }

        // deformed position of the control points
        torch::Tensor displacedNetCtrlPts = netCtrlPts + netDisplacements;

        // json objects for deformed positions
        nlohmann::json displacedNetCtrlPts_j = nlohmann::json::array();
        for (int i = 0; i < displacedNetCtrlPts.size(0); ++i) {
            displacedNetCtrlPts_j.push_back({
                displacedNetCtrlPts[i][0].item<double>(),
                displacedNetCtrlPts[i][1].item<double>()
            });
        }

        // write net data
        net.appendToJsonFile("net_CtrlPts", displacedNetCtrlPts_j);
        net.appendToJsonFile("net_Degree", DEGREE);

        #ifdef IGANET_WITH_GISMO
        
            torch::Tensor gsOriginCtrlPts    = torch::empty({0, 2}, 
                torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
            torch::Tensor gsDisplacements    = torch::empty({0, 2}, 
                torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
            torch::Tensor gsCtrlPts          = torch::empty({0, 2}, 
                torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
            torch::Tensor gsStresses         = torch::empty({0, 1}, 
                torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
            
            nlohmann::json gsOriginCtrlPts_j = nlohmann::json::array();
            nlohmann::json gsDisplacements_j = nlohmann::json::array();
            nlohmann::json gsCtrlPts_j       = nlohmann::json::array();
            nlohmann::json gsStresses_j      = nlohmann::json::array();

            std::tie(gsOriginCtrlPts, gsDisplacements, gsStresses) =
            linear_elasticity_t::RunGismoSimulation(
                NR_CTRL_PTS, DEGREE, YOUNG_MODULUS, POISSON_RATIO,
                DIRI_SIDES, FORCE_SIDES, BODY_FORCE);
            
            // calculate the new position of the control points after displacement
            gsCtrlPts = gsOriginCtrlPts + gsDisplacements;

            for (int i = 0; i < gsCtrlPts.size(0); ++i) {
                gsOriginCtrlPts_j.push_back({
                    gsOriginCtrlPts[i][0].item<double>(),
                    gsOriginCtrlPts[i][1].item<double>()
                });
                
                gsDisplacements_j.push_back({
                    gsDisplacements[i][0].item<double>(),
                    gsDisplacements[i][1].item<double>()
                });

                gsCtrlPts_j.push_back({
                    gsCtrlPts[i][0].item<double>(),
                    gsCtrlPts[i][1].item<double>()
                });

                gsStresses_j.push_back({ 
                    gsStresses[i][0].item<double>() 
                });
            }

            net.appendToJsonFile("gsOriginCtrlPts", gsOriginCtrlPts_j);
            net.appendToJsonFile("gsDisplacements", gsDisplacements_j);
            net.appendToJsonFile("gsCtrlPts", gsCtrlPts_j);
            net.appendToJsonFile("gsStresses", gsStresses_j);

            if (RUN_GS_REF_SIM) {
                torch::Tensor gsRefOriginCtrlPts    = torch::empty({0, 2}, 
                    torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
                torch::Tensor gsRefCtrlPts          = torch::empty({0, 2}, 
                    torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
                torch::Tensor gsRefDisplacements    = torch::empty({0, 2}, 
                    torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
                torch::Tensor gsRefStresses         = torch::empty({0, 1}, 
                    torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

                nlohmann::json gsRefOriginCtrlPts_j = nlohmann::json::array();
                nlohmann::json gsRefCtrlPts_j       = nlohmann::json::array();
                nlohmann::json gsRefDisplacements_j = nlohmann::json::array();
                nlohmann::json gsRefStresses_j      = nlohmann::json::array();

                std::tie(gsRefOriginCtrlPts, gsRefDisplacements, gsRefStresses) =
                    linear_elasticity_t::RunGismoSimulation(
                        NR_CTRL_PTS_REF, DEGREE,
                        YOUNG_MODULUS, POISSON_RATIO,
                        DIRI_SIDES, FORCE_SIDES, BODY_FORCE);

                // calculate the new position of the reference solution's control points after displacement        
                gsRefCtrlPts = gsRefOriginCtrlPts + gsRefDisplacements;

                for (int i = 0; i < gsRefCtrlPts.size(0); ++i) {
                    gsRefOriginCtrlPts_j.push_back({
                        gsRefOriginCtrlPts[i][0].item<double>(),
                        gsRefOriginCtrlPts[i][1].item<double>()
                    });

                    gsRefCtrlPts_j.push_back({
                        gsRefCtrlPts[i][0].item<double>(),
                        gsRefCtrlPts[i][1].item<double>()
                    });

                    gsRefDisplacements_j.push_back({
                        gsRefDisplacements[i][0].item<double>(),
                        gsRefDisplacements[i][1].item<double>()
                    });
                    
                    gsRefStresses_j.push_back({ 
                        gsRefStresses[i][0].item<double>() 
                    });
                }

                net.appendToJsonFile("gsRefOriginCtrlPts", gsRefOriginCtrlPts_j);
                net.appendToJsonFile("gsRefCtrlPts", gsRefCtrlPts_j);
                net.appendToJsonFile("gsRefDisplacements", gsRefDisplacements_j);
                net.appendToJsonFile("gsRefStresses", gsRefStresses_j);
                net.appendToJsonFile("gsRefDegree", DEGREE_REF);
            }
        #endif
        return 0;
    };
    
    const auto dispatch = [&]<typename optimizer_t>() -> int {
        switch (DEGREE_CFG) {
        case 2: return run.template operator()<2, optimizer_t>();
        case 3: return run.template operator()<3, optimizer_t>();
        case 4: return run.template operator()<4, optimizer_t>();
        case 5: return run.template operator()<5, optimizer_t>();
        case 6: return run.template operator()<6, optimizer_t>();
        default:
            std::cerr << "Error: Invalid degree " << DEGREE_CFG << " (2..6)\n" << std::endl;
            return 1;
        }
    };

    switch (OPTIMIZER_CFG.type) {
    case optimizer_type_t::adam:
        return dispatch.template operator()<torch::optim::Adam>();
    case optimizer_type_t::lbfgs:
        return dispatch.template operator()<torch::optim::LBFGS>();
    default:
        std::cerr << "Unsupported optimizer selection in sim_config_2D_single_patch.json\n";
        return 1;
    }

    iganet::finalize();
    return 0;
}
