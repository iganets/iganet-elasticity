#include <iganet.h>
#include <iostream>
#include <fstream>
#include <utils/config.hpp>
#include <utils/paths.hpp>

using namespace iganet::literals;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using iganet_elasticity::utils::config::require;

namespace {

struct SideVectorBoundaryCondition {
    int side;
    double x;
    double y;
};

struct PatchBoundaryConditions {
    std::vector<SideVectorBoundaryCondition> diri_sides;
    std::vector<SideVectorBoundaryCondition> force_sides;
    std::vector<int> tfbc_sides;
};

struct PatchConfig {
    int id;
    int64_t nr_ctrl_pts;
    PatchBoundaryConditions boundary_conditions;
};

struct PatchInterfaceConfig {
    int patch_a;
    int side_a;
    int patch_b;
    int side_b;
    std::string orientation;
};

PatchBoundaryConditions parse_patch_boundary_conditions(const nlohmann::json& bc_json) {
    PatchBoundaryConditions bc;

    for (const auto& dsj : require(bc_json, "diri_sides")) {
        bc.diri_sides.push_back({
            dsj.at(0).get<int>(),
            dsj.at(1).get<double>(),
            dsj.at(2).get<double>()
        });
    }

    for (const auto& fsj : require(bc_json, "force_sides")) {
        bc.force_sides.push_back({
            fsj.at(0).get<int>(),
            fsj.at(1).get<double>(),
            fsj.at(2).get<double>()
        });
    }

    bc.tfbc_sides = require(bc_json, "tfbc_sides").get<std::vector<int>>();
    return bc;
}

std::vector<PatchConfig> parse_patch_configs(const nlohmann::json& j, int64_t default_nr_ctrl_pts) {
    std::vector<PatchConfig> patches;

    if (!j.contains("patches")) {
        PatchConfig patch;
        patch.id = 0;
        patch.nr_ctrl_pts = default_nr_ctrl_pts;
        patch.boundary_conditions = parse_patch_boundary_conditions(require(j, "boundary_conditions"));
        patches.push_back(std::move(patch));
        return patches;
    }

    for (const auto& patch_json : require(j, "patches")) {
        PatchConfig patch;
        patch.id = require(patch_json, "id").get<int>();
        patch.nr_ctrl_pts = patch_json.contains("spline")
            ? require(patch_json, "spline.nr_ctrl_pts").get<int64_t>()
            : default_nr_ctrl_pts;
        patch.boundary_conditions = parse_patch_boundary_conditions(
            require(patch_json, "boundary_conditions"));
        patches.push_back(std::move(patch));
    }

    return patches;
}

std::vector<PatchInterfaceConfig> parse_patch_interfaces(const nlohmann::json& j) {
    std::vector<PatchInterfaceConfig> interfaces;

    if (!j.contains("interfaces")) {
        return interfaces;
    }

    for (const auto& interface_json : require(j, "interfaces")) {
        interfaces.push_back({
            require(interface_json, "patch_a").get<int>(),
            require(interface_json, "side_a").get<int>(),
            require(interface_json, "patch_b").get<int>(),
            require(interface_json, "side_b").get<int>(),
            interface_json.value("orientation", "aligned")
        });
    }

    return interfaces;
}

void append_json_key(const std::string& jsonPath, const std::string& key, const nlohmann::json& data) {
    nlohmann::json jsonData;

    try {
        std::ifstream json_file_in(jsonPath);
        if (json_file_in.is_open()) {
            json_file_in >> jsonData;
            json_file_in.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading JSON file: " << jsonPath
                  << ". Exception: " << e.what() << "\n";
    }

    try {
        jsonData[key] = data;
    } catch (const std::exception& e) {
        std::cerr << "Error adding key to JSON object: " << e.what() << "\n";
        return;
    }

    try {
        std::ofstream json_file_out(jsonPath);
        if (json_file_out.is_open()) {
            json_file_out << jsonData.dump(1);
            json_file_out.close();
        } else {
            std::cerr << "Error: Could not open file for writing: "
                      << jsonPath << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing JSON file: " << jsonPath
                  << ". Exception: " << e.what() << "\n";
    }
}

} // namespace

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
    using Customizable = iganet::IgANetCustomizable<Inputs, Outputs>;

    typename Base::template collPts_t<0> collPts_;
    typename Base::template collPts_t<0> interiorCollPts_;

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

    /// @brief Prepares collocation points and cached index data once before training
    void initialize_problem_data() {
        Base::inputs(0);
        collPts_         = Base::template collPts<0>(iganet::collPts::greville);
        interiorCollPts_ = Base::template collPts<0>(iganet::collPts::greville_interior);

        // WARNING, only works for equal number of control points in x and y direction
        nrCollPts_ = static_cast<int>(std::sqrt(std::get<0>(collPts_)[0].size(0)));
        torch::Tensor collPtsCoeffs = std::get<0>(collPts_)[0].slice(0, 0, nrCollPts_);
        nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
        for (int i = 0; i < collPtsCoeffs.size(0); ++i) {
            collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
        }
        appendToJsonFile("net_collPtsCoeffsRef1", collPtsCoeffs_j);
        appendToJsonFile("net_nrCollPtsRef1", {nrCollPts_});

        var_knot_indices_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                collPts_.first);
        var_coeff_indices_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_);

        var_knot_indices_interior_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                    interiorCollPts_.first);
        var_coeff_indices_interior_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_interior_);

        G_knot_indices_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                collPts_.first);
        G_coeff_indices_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_);

        G_knot_indices_interior_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                interiorCollPts_.first);
        G_coeff_indices_interior_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_interior_);
    }
    
    /// @brief Writes data to a JSON file
    void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
        append_json_key(JSON_PATH_, key, data);
    }

    /// @brief helper function to load the std collocation displacements from a JSON file
    torch::Tensor loadDisplacements(const torch::TensorOptions& options) {
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
        return epoch == 0;
    }

    /// @brief Computes the loss function
    torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {
        // create u_ from the training's outputs
        this->template output<0>().from_tensor(outputs);

        // pre-allocation of the loss values
        torch::Tensor totalLoss; 
        torch::Tensor lossPDE;
        std::optional<torch::Tensor> lossBC;
        std::optional<torch::Tensor> lossINTER;
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
            
            // intersecCtr is used to determine an intersection of dirichlet/force and trac.free sides
            static std::vector<int> intersecCtr(0);
            // allocate tensors for the traction-free boundary conditions
            static std::array<torch::Tensor, 2ul> tractionCollPts;
            // collect sides of traction-free and force BCs
            std::vector<int> neumannSides;

            // collect sides of Dirichlet or force BCs
            std::vector<int> diriOrForceSides;
            for (const auto& tuple : DIRI_SIDES_) {
                // extract only the int-values from DIRI_SIDES_
                diriOrForceSides.push_back(std::get<0>(tuple));
            }       
            
            // add the two vectors of force- and traction-free-BCs
            neumannSides.reserve(TFBC_SIDES_.size() + FORCE_SIDES_.size());
            neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
            // add the force sides to the neumannSides and diriOrForceSides
            for (const auto& force : FORCE_SIDES_) {
                // add the force sides to the neumannSides
                neumannSides.push_back(std::get<0>(force));
                // add the force sides to the diriOrForceSides
                diriOrForceSides.push_back(std::get<0>(force));
            }

            // calculate the tractionCollocationPoints once in the beginning of the simulation
            if (epoch == 0 && intersecCtr.empty()) {
                // allocate tensors for the traction-free boundary conditions
                std::vector<torch::Tensor> tractionCollPtsX;
                std::vector<torch::Tensor> tractionCollPtsY;

                // evaluate the boundary points depending on traction-free sides
                for (int side : neumannSides) {
                    if (side == 1) {
                        // check if diriOrForceSides has only side 3 as side
                        if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                            == diriOrForceSides.end()) {     

                            at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}, collPtsY_tensor.options()));
                            tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has only side 4 as side
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                                == diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                                != diriOrForceSides.end()) {
            
                            at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}, collPtsY_tensor.options()));
                            tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has side 3 and side 4
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                                != diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                                != diriOrForceSides.end()) {
                            
                            at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 2}, collPtsY_tensor.options()));  
                            tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                            // 2 collPts have to be removed
                            intersecCtr.push_back(2);
                        }
                        else {
                            tractionCollPtsX.push_back(torch::zeros(nrCollPts_, std::get<0>(collPts_.second)[0].options()));
                            tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                            // no collPt has to be removed
                            intersecCtr.push_back(0);
                        }
                    }
                    else if (side == 2) {
                        // check if diriOrForceSides has only side 3 as side
                        if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                            == diriOrForceSides.end()) {    

                            at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}, collPtsY_tensor.options()));
                            tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has only side 4 as side
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                                == diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                                != diriOrForceSides.end()) {

                            at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}, collPtsY_tensor.options()));
                            tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has side 3 and side 4
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                                != diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                                != diriOrForceSides.end()) {

                            at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 2}, collPtsY_tensor.options()));
                            tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                            // 2 collPts have to be removed
                            intersecCtr.push_back(2);
                        }
                        else {
                            tractionCollPtsX.push_back(torch::ones(nrCollPts_, std::get<0>(collPts_.second)[0].options()));
                            tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                            // no collPt has to be removed
                            intersecCtr.push_back(0);
                        }
                        
                    }
                    else if (side == 3) {
                        // check if diriOrForceSides has only side 1 as side
                        if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                            == diriOrForceSides.end()) {   

                            at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                            tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}, collPtsX_tensor.options()));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has only side 2 as side
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                                == diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                                != diriOrForceSides.end()) {   

                            at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                            tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}, collPtsX_tensor.options()));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has side 1 and side 2
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                                != diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                                != diriOrForceSides.end()) {   

                            at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                            tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 2}, collPtsX_tensor.options()));
                            // 2 collPts have to be removed
                            intersecCtr.push_back(2);
                        }
                        else {
                            tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                            tractionCollPtsY.push_back(torch::zeros(nrCollPts_, std::get<0>(collPts_.second)[0].options()));
                            // no collPt has to be removed
                            intersecCtr.push_back(0);
                        }
                    }
                    else if (side == 4) {
                        // check if diriOrForceSides has only side 1 as side
                        if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                            == diriOrForceSides.end()) {   

                            at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                            tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}, collPtsX_tensor.options()));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has only side 2 as side
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                                == diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                                != diriOrForceSides.end()) {   

                            at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                            tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}, collPtsX_tensor.options()));
                            // 1 collPt has to be removed
                            intersecCtr.push_back(1);
                        }
                        // check if diriOrForceSides has side 1 and side 2
                        else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                                != diriOrForceSides.end() &&
                                std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                                != diriOrForceSides.end()) {   

                            at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                            tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                            tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 2}, collPtsX_tensor.options()));
                            // 2 collPts have to be removed
                            intersecCtr.push_back(2);
                        }
                        else {
                            tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                            tractionCollPtsY.push_back(torch::ones(nrCollPts_, std::get<0>(collPts_.second)[0].options()));
                            // no collPt has to be removed
                            intersecCtr.push_back(0);
                        }
                    }
                    
                    else {
                        throw std::invalid_argument("Side for traction BC has to be 1, 2, 3 or 4.");
                    }
                }
                
                // merge the tensors to get a (nrTractionCollPts, 2) tensor
                if (!tractionCollPtsX.empty() && !tractionCollPtsY.empty()) {
                    tractionCollPts = {
                        torch::cat(tractionCollPtsX, 0), 
                        torch::cat(tractionCollPtsY, 0)
                    };
                } 
                var_knot_indices_boundary_ =
                    Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                    tractionCollPts);
                var_coeff_indices_boundary_ =
                    Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                    var_knot_indices_boundary_);
                G_knot_indices_boundary_ =
                    this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                        tractionCollPts);
                G_coeff_indices_boundary_ =
                this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                    G_knot_indices_boundary_);
            }  

            // calculate the jacobian of the affected boundary points
            auto jacobianBoundary = this->template output<0>().ijac(this->template input<0>(), tractionCollPts, 
                var_knot_indices_boundary_, var_coeff_indices_boundary_,
                G_knot_indices_boundary_, G_coeff_indices_boundary_);
            auto ux_x = *jacobianBoundary[0];
            auto ux_y = *jacobianBoundary[1];
            auto uy_x = *jacobianBoundary[2];
            auto uy_y = *jacobianBoundary[3];

            // allocate tensors for the traction-free boundary conditions (tfbc)
            torch::Tensor tractionValuesX = torch::zeros({tractionCollPts[0].size(0)}, ux_x.options());
            torch::Tensor tractionValuesY = torch::zeros({tractionCollPts[0].size(0)}, ux_x.options());
            // calculate the traction values at the boundary points
            int pointCtr = 0;
            int sideCtr = 0; 

            for (int side : neumannSides) {
                int n_vals = nrCollPts_ - intersecCtr[sideCtr];
                auto ux_x_slice = ux_x.slice(0, pointCtr, pointCtr + n_vals);
                auto ux_y_slice = ux_y.slice(0, pointCtr, pointCtr + n_vals);
                auto uy_x_slice = uy_x.slice(0, pointCtr, pointCtr + n_vals);
                auto uy_y_slice = uy_y.slice(0, pointCtr, pointCtr + n_vals);
                auto tractionValuesXSlice = tractionValuesX.slice(0, pointCtr, pointCtr + n_vals);
                auto tractionValuesYSlice = tractionValuesY.slice(0, pointCtr, pointCtr + n_vals);

                if (side == 1) {
                    tractionValuesXSlice.copy_(-lambda_ * (ux_x_slice + uy_y_slice) - 2 * mu_ * ux_x_slice);
                    tractionValuesYSlice.copy_(-mu_ * (uy_x_slice + ux_y_slice));
                }
                else if (side == 2) {
                    tractionValuesXSlice.copy_(lambda_ * (ux_x_slice + uy_y_slice) + 2 * mu_ * ux_x_slice);
                    tractionValuesYSlice.copy_(mu_ * (uy_x_slice + ux_y_slice));
                }
                else if (side == 3) {
                    tractionValuesXSlice.copy_(-mu_ * (uy_x_slice + ux_y_slice));
                    tractionValuesYSlice.copy_(-lambda_ * (ux_x_slice + uy_y_slice) - 2 * mu_ * uy_y_slice);
                }
                else if (side == 4) {
                    tractionValuesXSlice.copy_(mu_ * (uy_x_slice + ux_y_slice));
                    tractionValuesYSlice.copy_(lambda_ * (ux_x_slice + uy_y_slice) + 2 * mu_ * uy_y_slice);
                }

                pointCtr += n_vals;
                sideCtr++;
            }

            // merge the traction tensors of x- and y-directions
            torch::Tensor tractionValues = torch::stack({tractionValuesX, tractionValuesY}, 1);

            if (!FORCE_SIDES_.empty()) {
                // calculate total cutlength from last forceSize entries of intersecCtr
                int cutlength = 0;
                int forceSize = FORCE_SIDES_.size();
                for (int i = static_cast<int>(intersecCtr.size()) - forceSize; 
                        i < static_cast<int>(intersecCtr.size()); ++i) {
                    cutlength += nrCollPts_ - intersecCtr[i];
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
                int startIdx = static_cast<int>(intersecCtr.size()) - forceSize;
                for (size_t i = 0; i < FORCE_SIDES_.size(); ++i) {
                    int reducedPts = nrCollPts_ - intersecCtr[startIdx + i];
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

        // calculation of the second derivatives of the displacements (u)
        auto hessianColl = this->template output<0>().ihess(this->template input<0>(), interiorCollPts_.first, 
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_, G_coeff_indices_interior_);

        // partial derivatives of the displacements (u)
        auto& ux_xx = hessianColl(0,0,0);
        auto& ux_xy = hessianColl(0,1,0);
        auto& ux_yx = hessianColl(1,0,0);
        auto& ux_yy = hessianColl(1,1,0);

        auto& uy_xx = hessianColl(0,0,1);
        auto& uy_xy = hessianColl(0,1,1);
        auto& uy_yx = hessianColl(1,0,1);
        auto& uy_yy = hessianColl(1,1,1);

        torch::Tensor divStressX = (lambda_ + 2 * mu_) * ux_xx
                                 + mu_ * ux_yy
                                 + (lambda_ + mu_) * uy_xy;
        torch::Tensor divStressY = mu_ * uy_xx
                                 + (lambda_ + 2 * mu_) * uy_yy
                                 + (lambda_ + mu_) * ux_xy;
        
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
            lossPDE = torch::mse_loss(divStress, bodyForce);
            
            // add the elasticity loss to the total loss
            totalLoss = lossPDE;

            // add the elasticity loss to the cmd-output variable
            singleLossOutput << "PDE " << std::setw(10) << lossPDE.item<double>();

            lossINTER = torch::zeros({}, outputs.options());
            totalLoss += *lossINTER;

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
                lossBC = torch::zeros({}, outputs.options());

                // evaluation of the displacements at the boundary points
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.second);
                // evaluation of the displacements at the reference boundary points
                auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

                // loop through all dirichlet sides
                for (const auto& side : DIRI_SIDES_) {
                    int sideNr = std::get<0>(side);
                    
                    switch (sideNr) {
                        case 1: 
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                            break;
                        case 2:
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                            break;
                        case 3:
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                            break;
                        case 4:
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
                                torch::mse_loss(*std::get<3>(u_bdr)[1], *std::get<3>(bdr)[1]));
                            break;
                        default:
                            std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                    }
                }
                totalLoss += *lossBC;
                singleLossOutput << " + BC " << std::setw(11) << (*lossBC).item<double>() / bcWeight 
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
            torch::Tensor stdCollDisplacements_ = loadDisplacements(netDisplacements_.options());

            // supervised loss: MSE of net against standard collocation solution
            gsLoss = 1e9 * torch::mse_loss(netDisplacements_, stdCollDisplacements_);

            // calculation of the loss function for double-sided constraint solid
            // div(sigma) + f = 0 --> div(sigma) = -f
            lossPDE = torch::mse_loss(divStress, bodyForce);

            // add the elasticity loss and supervised loss to the total loss
            totalLoss = *gsLoss + lossPDE;

            // add the elasticity and supervised losses to the cmd-output variable
            singleLossOutput << "GL " << std::setw(11) << (*gsLoss).item<double>()
                            << " + PDE " << std::setw(10) << lossPDE.item<double>();

            lossINTER = torch::zeros({}, outputs.options());
            totalLoss += *lossINTER;

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
                lossBC = torch::zeros({}, outputs.options());

                // evaluation of the displacements at the boundary points
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.second);
                // evaluation of the displacements at the reference boundary points
                auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

                // loop through all dirichlet sides
                for (const auto& side : DIRI_SIDES_) {
                    int sideNr = std::get<0>(side);

                    switch (sideNr) {
                        case 1:
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                            break;
                        case 2:
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                            break;
                        case 3:
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                            break;
                        case 4:
                            *lossBC += bcWeight * 
                                (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
                                torch::mse_loss(*std::get<3>(u_bdr)[1], *std::get<3>(bdr)[1]));
                            break;
                        default:
                            std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                    }
                }
                totalLoss += *lossBC;
                singleLossOutput << " + BC " << std::setw(11) << (*lossBC).item<double>() / bcWeight 
                                << " * 1e" << static_cast<int>(std::log10(bcWeight));
            }

            // print the loss values
            std::cout << std::setw(11) << 
                totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
        }

        else {
            throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_");
        }
        return totalLoss;
    }

    void PostProc() {
        auto jacobian = this->template output<0>().ijac(this->template input<0>(), collPts_.first,
            var_knot_indices_, var_coeff_indices_, G_knot_indices_, G_coeff_indices_);

        auto ux_x = *jacobian[0];
        auto ux_y = *jacobian[1];
        auto uy_x = *jacobian[2];
        auto uy_y = *jacobian[3];

        auto __jac_opts = jacobian[0]->options();
        torch::Tensor sigma_xx = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor sigma_xy = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor sigma_yy = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor sigma_vm = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor epsilon_xx = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor epsilon_yy = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor poisson_re = torch::zeros({jacobian[0]->size(0)}, __jac_opts);

        nlohmann::json netVmStresses_j = nlohmann::json::array();
        nlohmann::json netXStresses_j = nlohmann::json::array();
        nlohmann::json netYStresses_j = nlohmann::json::array();
        nlohmann::json netPoisson_j = nlohmann::json::array();

        for (int i = 0; i < jacobian[0]->size(0); ++i) {
            sigma_xx[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * ux_x[i];
            sigma_xy[i] = mu_ * (uy_x[i] + ux_y[i]);
            sigma_yy[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * uy_y[i];

            sigma_vm[i] = sqrt(sigma_xx[i] * sigma_xx[i] + sigma_yy[i] * sigma_yy[i]
                            - sigma_xx[i] * sigma_yy[i] + sigma_xy[i] * sigma_xy[i] * 3);

            epsilon_xx[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) *
                (sigma_xx[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_yy[i]);
            epsilon_yy[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) *
                (sigma_yy[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_xx[i]);

            poisson_re[i] = - epsilon_yy[i] / epsilon_xx[i];

            netVmStresses_j.push_back({sigma_vm[i].item<double>()});
            netXStresses_j.push_back({sigma_xx[i].item<double>()});
            netYStresses_j.push_back({sigma_yy[i].item<double>()});
            netPoisson_j.push_back({poisson_re[i].item<double>()});
        }

        appendToJsonFile("net_VmStresses", netVmStresses_j);
        appendToJsonFile("net_XStresses", netXStresses_j);
        appendToJsonFile("net_YStresses", netYStresses_j);
        appendToJsonFile("net_Poisson", netPoisson_j);

        torch::Tensor collPtsFirstAsTensor = torch::stack(
            {std::get<0>(collPts_.first), std::get<1>(collPts_.first)}, 1);
        auto displacementOfCollPts = this->template output<0>().eval(collPts_.first);
        torch::Tensor displacementAsTensor = torch::stack(
            {*(displacementOfCollPts[0]), *(displacementOfCollPts[1]) }, 1);

        nlohmann::json collPtsFirst_j = nlohmann::json::array();
        nlohmann::json collPtsFirstDispl_j = nlohmann::json::array();
        for (int i = 0; i < collPtsFirstAsTensor.size(0); ++i) {
            collPtsFirst_j.push_back({collPtsFirstAsTensor[i][0].item<double>(),
                                    collPtsFirstAsTensor[i][1].item<double>()});
            collPtsFirstDispl_j.push_back({collPtsFirstAsTensor[i][0].item<double>() +
                                        displacementAsTensor[i][0].item<double>(),
                                        collPtsFirstAsTensor[i][1].item<double>() +
                                        displacementAsTensor[i][1].item<double>()});
        }
        appendToJsonFile("net_collPtsFirstAsTensor", collPtsFirst_j);
        appendToJsonFile("net_collPtsFirstAfterDisplacementAsTensor", collPtsFirstDispl_j);

        auto hessianColl = this->template output<0>().ihess(this->template input<0>(), interiorCollPts_.first,
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_, G_coeff_indices_interior_);

        auto& ux_xx = hessianColl(0,0,0);
        auto& ux_yy = hessianColl(1,1,0);
        auto& uy_xx = hessianColl(0,0,1);
        auto& uy_xy = hessianColl(0,1,1);
        auto& uy_yy = hessianColl(1,1,1);
        auto& ux_xy = hessianColl(0,1,0);

        torch::Tensor divStressX = (lambda_ + 2 * mu_) * ux_xx
                                 + mu_ * ux_yy
                                 + (lambda_ + mu_) * uy_xy;
        torch::Tensor divStressY = mu_ * uy_xx
                                 + (lambda_ + 2 * mu_) * uy_yy
                                 + (lambda_ + mu_) * ux_xy;

        nlohmann::json netDivergenceX_j = nlohmann::json::array();
        nlohmann::json netDivergenceY_j = nlohmann::json::array();
        for (int i = 0; i < divStressX.size(0); ++i) {
            netDivergenceX_j.push_back({divStressX[i].item<double>()});
            netDivergenceY_j.push_back({divStressY[i].item<double>()});
        }

        appendToJsonFile("net_DivergenceX", netDivergenceX_j);
        appendToJsonFile("net_DivergenceY", netDivergenceY_j);
    }
};

/// @brief Experimental 2-patch variant with weak interface coupling
template <typename Optimizer, typename GeometryMap, typename Variable>
class linear_elasticity_2patch
    : public iganet::IgANet<Optimizer, std::tuple<GeometryMap, GeometryMap>,
                            std::tuple<Variable, Variable>>,
      public iganet::IgANetCustomizable<std::tuple<GeometryMap, GeometryMap>,
                                        std::tuple<Variable, Variable>>
{
private:
    using Inputs = std::tuple<GeometryMap, GeometryMap>;
    using Outputs = std::tuple<Variable, Variable>;

    using Base = iganet::IgANet<Optimizer, Inputs, Outputs>;
    using Customizable = iganet::IgANetCustomizable<Inputs, Outputs>;

    struct PatchCache {
        typename Base::template collPts_t<0> collPts;
        typename Base::template collPts_t<0> interiorCollPts;

        typename Customizable::template output_interior_knot_indices_t<0> var_knot_indices;
        typename Customizable::template output_interior_coeff_indices_t<0> var_coeff_indices;
        typename Customizable::template output_interior_knot_indices_t<0> var_knot_indices_interior;
        typename Customizable::template output_interior_coeff_indices_t<0> var_coeff_indices_interior;
        typename Customizable::template input_interior_knot_indices_t<0> G_knot_indices;
        typename Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices;
        typename Customizable::template input_interior_knot_indices_t<0> G_knot_indices_interior;
        typename Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_interior;
        int nrCollPts = 0;
    };

    static std::tuple<Variable, Variable> make_reference_tuple(
        const std::array<PatchConfig, 2>& patches) {
        return std::make_tuple(
            Variable(iganet::utils::to_array(
                patches[0].nr_ctrl_pts, patches[0].nr_ctrl_pts)),
            Variable(iganet::utils::to_array(
                patches[1].nr_ctrl_pts, patches[1].nr_ctrl_pts)));
    }

    template <std::size_t Patch>
    auto& patch_output() { return this->template output<Patch>(); }

    template <std::size_t Patch>
    auto const& patch_output() const { return this->template output<Patch>(); }

    template <std::size_t Patch>
    auto& patch_input() { return this->template input<Patch>(); }

    template <std::size_t Patch>
    auto const& patch_input() const { return this->template input<Patch>(); }

    template <std::size_t Patch>
    auto& patch_ref() { return std::get<Patch>(ref_); }

    template <std::size_t Patch>
    auto const& patch_ref() const { return std::get<Patch>(ref_); }

    template <std::size_t Patch>
    auto& patch_cache() { return patchCaches_[Patch]; }

    template <std::size_t Patch>
    auto const& patch_cache() const { return patchCaches_[Patch]; }

    static std::vector<int> collect_occupied_sides(
        const PatchBoundaryConditions& bc, bool include_tfbc = false) {
        std::vector<int> occupied;
        for (const auto& side : bc.diri_sides) occupied.push_back(side.side);
        for (const auto& side : bc.force_sides) occupied.push_back(side.side);
        if (include_tfbc) {
            occupied.insert(occupied.end(), bc.tfbc_sides.begin(), bc.tfbc_sides.end());
        }
        return occupied;
    }

    static bool has_side(const std::vector<int>& sides, int side) {
        return std::find(sides.begin(), sides.end(), side) != sides.end();
    }

    template <std::size_t Patch>
    std::array<torch::Tensor, 2> build_side_collocation_points(
        int side, bool trimCorners, bool reverseDirection = false,
        bool refinedInterface = false) const {
        const auto& cache = patch_cache<Patch>();
        const auto& boundaryCollPts = cache.collPts.second;
        const auto collPts1D = std::get<0>(boundaryCollPts)[0];
        int64_t begin = 0;
        int64_t end = collPts1D.size(0);

        if (trimCorners) {
            const auto occupied = collect_occupied_sides(
                patches_[Patch].boundary_conditions,
                /* include_tfbc = */ refinedInterface);
            switch (side) {
                case 1:
                case 2:
                    if (has_side(occupied, 3)) begin += 1;
                    if (has_side(occupied, 4)) end -= 1;
                    break;
                case 3:
                case 4:
                    if (has_side(occupied, 1)) begin += 1;
                    if (has_side(occupied, 2)) end -= 1;
                    break;
                default:
                    throw std::invalid_argument("Side must be 1, 2, 3 or 4.");
            }
        }

        auto line = collPts1D.slice(0, begin, end);
        if (reverseDirection) {
            line = torch::flip(line, {0});
        }

        switch (side) {
            case 1:
                return {torch::zeros({line.size(0)}, line.options()), line};
            case 2:
                return {torch::ones({line.size(0)}, line.options()), line};
            case 3:
                return {line, torch::zeros({line.size(0)}, line.options())};
            case 4:
                return {line, torch::ones({line.size(0)}, line.options())};
            default:
                throw std::invalid_argument("Side must be 1, 2, 3 or 4.");
        }
    }

    template <std::size_t Patch>
    torch::Tensor displacement_tensor(const std::array<torch::Tensor, 2>& evalPts) {
        auto disp = patch_output<Patch>().eval(evalPts);
        return torch::stack({*disp[0], *disp[1]}, 1);
    }

    template <std::size_t Patch>
    std::vector<int64_t> side_control_point_indices(int side, bool reverseDirection = false) const {
        const int64_t nrCtrlPts = patches_[Patch].nr_ctrl_pts;
        auto geometryTensor = patch_input<Patch>().as_tensor();
        const auto xCoords = geometryTensor.slice(0, 0, nrCtrlPts * nrCtrlPts);
        const auto yCoords = geometryTensor.slice(0, nrCtrlPts * nrCtrlPts, 2 * nrCtrlPts * nrCtrlPts);

        double targetCoord = 0.0;
        bool verticalSide = (side == 1 || side == 2);

        switch (side) {
            case 1:
                targetCoord = xCoords.min().template item<double>();
                break;
            case 2:
                targetCoord = xCoords.max().template item<double>();
                break;
            case 3:
                targetCoord = yCoords.min().template item<double>();
                break;
            case 4:
                targetCoord = yCoords.max().template item<double>();
                break;
            default:
                throw std::invalid_argument("Side must be 1, 2, 3 or 4.");
        }

        constexpr double tol = 1e-10;
        std::vector<std::pair<double, int64_t>> orderedIndices;
        orderedIndices.reserve(nrCtrlPts);

        for (int64_t i = 0; i < nrCtrlPts * nrCtrlPts; ++i) {
            const double x = xCoords[i].template item<double>();
            const double y = yCoords[i].template item<double>();
            if ((verticalSide && std::abs(x - targetCoord) < tol) ||
                (!verticalSide && std::abs(y - targetCoord) < tol)) {
                orderedIndices.emplace_back(verticalSide ? y : x, i);
            }
        }

        std::sort(
            orderedIndices.begin(),
            orderedIndices.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        if (reverseDirection) {
            std::reverse(orderedIndices.begin(), orderedIndices.end());
        }

        std::vector<int64_t> result;
        result.reserve(orderedIndices.size());
        for (const auto& entry : orderedIndices) {
            result.push_back(entry.second);
        }
        return result;
    }

    template <std::size_t Patch>
    torch::Tensor traction_tensor(int side, const std::array<torch::Tensor, 2>& evalPts) {
        auto varKnot =
            patch_output<Patch>().template find_knot_indices<iganet::functionspace::interior>(evalPts);
        auto varCoeff =
            patch_output<Patch>().template find_coeff_indices<iganet::functionspace::interior>(varKnot);
        auto geoKnot =
            patch_input<Patch>().template find_knot_indices<iganet::functionspace::interior>(evalPts);
        auto geoCoeff =
            patch_input<Patch>().template find_coeff_indices<iganet::functionspace::interior>(geoKnot);

        auto jacobianBoundary = patch_output<Patch>().ijac(
            patch_input<Patch>(), evalPts, varKnot, varCoeff, geoKnot, geoCoeff);

        auto ux_x = *jacobianBoundary[0];
        auto ux_y = *jacobianBoundary[1];
        auto uy_x = *jacobianBoundary[2];
        auto uy_y = *jacobianBoundary[3];

        auto tx = torch::zeros({evalPts[0].size(0)}, ux_x.options());
        auto ty = torch::zeros({evalPts[0].size(0)}, ux_x.options());

        switch (side) {
            case 1:
                tx = -lambda_ * (ux_x + uy_y) - 2 * mu_ * ux_x;
                ty = -mu_ * (uy_x + ux_y);
                break;
            case 2:
                tx = lambda_ * (ux_x + uy_y) + 2 * mu_ * ux_x;
                ty = mu_ * (uy_x + ux_y);
                break;
            case 3:
                tx = -mu_ * (uy_x + ux_y);
                ty = -lambda_ * (ux_x + uy_y) - 2 * mu_ * uy_y;
                break;
            case 4:
                tx = mu_ * (uy_x + ux_y);
                ty = lambda_ * (ux_x + uy_y) + 2 * mu_ * uy_y;
                break;
            default:
                throw std::invalid_argument("Side must be 1, 2, 3 or 4.");
        }

        return torch::stack({tx, ty}, 1);
    }

    template <std::size_t Patch>
    torch::Tensor compute_patch_pde_loss() {
        auto& cache = patch_cache<Patch>();
        auto hessianColl = patch_output<Patch>().ihess(
            patch_input<Patch>(), cache.interiorCollPts.first,
            cache.var_knot_indices_interior, cache.var_coeff_indices_interior,
            cache.G_knot_indices_interior, cache.G_coeff_indices_interior);

        auto& ux_xx = hessianColl(0,0,0);
        auto& ux_xy = hessianColl(0,1,0);
        auto& ux_yy = hessianColl(1,1,0);
        auto& uy_xx = hessianColl(0,0,1);
        auto& uy_xy = hessianColl(0,1,1);
        auto& uy_yy = hessianColl(1,1,1);

        torch::Tensor divStressX = (lambda_ + 2 * mu_) * ux_xx
                                 + mu_ * ux_yy
                                 + (lambda_ + mu_) * uy_xy;
        torch::Tensor divStressY = mu_ * uy_xx
                                 + (lambda_ + 2 * mu_) * uy_yy
                                 + (lambda_ + mu_) * ux_xy;

        torch::Tensor divStress = torch::stack({divStressX, divStressY}, 1);
        torch::Tensor bodyForce = torch::tensor(
            {BODY_FORCE_.first, BODY_FORCE_.second},
            divStress.options()).view({1, 2}).repeat({divStress.size(0), 1});

        return torch::mse_loss(divStress, bodyForce);
    }

    template <std::size_t Patch>
    torch::Tensor compute_patch_bc_loss(const torch::TensorOptions& opts) {
        const auto& bc = patches_[Patch].boundary_conditions;
        auto lossBC = torch::zeros({}, opts);

        if (!bc.diri_sides.empty()) {
            const int bcWeight = 1e7;
            auto u_bdr = patch_output<Patch>().template eval<iganet::functionspace::boundary>(
                patch_cache<Patch>().collPts.second);
            auto bdr = patch_ref<Patch>().template eval<iganet::functionspace::boundary>(
                patch_cache<Patch>().collPts.second);

            for (const auto& side : bc.diri_sides) {
                switch (side.side) {
                    case 1:
                        lossBC += bcWeight * (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
                                              torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                        break;
                    case 2:
                        lossBC += bcWeight * (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
                                              torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                        break;
                    case 3:
                        lossBC += bcWeight * (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) +
                                              torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                        break;
                    case 4:
                        lossBC += bcWeight * (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) +
                                              torch::mse_loss(*std::get<3>(u_bdr)[1], *std::get<3>(bdr)[1]));
                        break;
                    default:
                        throw std::invalid_argument("Invalid Dirichlet side.");
                }
            }
        }

        for (int side : bc.tfbc_sides) {
            auto sidePts = build_side_collocation_points<Patch>(side, true);
            auto traction = traction_tensor<Patch>(side, sidePts);
            lossBC += torch::mse_loss(traction, torch::zeros_like(traction));
        }

        for (const auto& force : bc.force_sides) {
            auto sidePts = build_side_collocation_points<Patch>(force.side, true);
            auto traction = traction_tensor<Patch>(force.side, sidePts);
            auto target = torch::zeros_like(traction);
            target.slice(1, 0, 1).fill_(force.x);
            target.slice(1, 1, 2).fill_(force.y);
            lossBC += torch::mse_loss(traction, target);
        }

        return lossBC;
    }

    torch::Tensor compute_interface_loss(const torch::TensorOptions& opts) {
        auto lossINTER = torch::zeros({}, opts);

        for (const auto& interfaceCfg : interfaces_) {
            if (!((interfaceCfg.patch_a == 0 && interfaceCfg.patch_b == 1) ||
                  (interfaceCfg.patch_a == 1 && interfaceCfg.patch_b == 0))) {
                throw std::runtime_error("2-patch prototype currently expects interfaces between patch 0 and 1.");
            }

            const bool reverse = interfaceCfg.orientation == "reversed";

            if (interfaceCfg.patch_a == 0 && interfaceCfg.patch_b == 1) {
                auto sideA = build_side_collocation_points<0>(interfaceCfg.side_a, true, false, true);
                auto sideB = build_side_collocation_points<1>(interfaceCfg.side_b, true, reverse, true);
                auto tracA = traction_tensor<0>(interfaceCfg.side_a, sideA);
                auto tracB = traction_tensor<1>(interfaceCfg.side_b, sideB);
                lossINTER += INTERFACE_TRACTION_WEIGHT_ *
                             torch::mse_loss(tracA + tracB, torch::zeros_like(tracA));
            } else {
                auto sideA = build_side_collocation_points<1>(interfaceCfg.side_a, true, false, true);
                auto sideB = build_side_collocation_points<0>(interfaceCfg.side_b, true, reverse, true);
                auto tracA = traction_tensor<1>(interfaceCfg.side_a, sideA);
                auto tracB = traction_tensor<0>(interfaceCfg.side_b, sideB);
                lossINTER += INTERFACE_TRACTION_WEIGHT_ *
                             torch::mse_loss(tracA + tracB, torch::zeros_like(tracA));
            }
        }

        return lossINTER;
    }

    void assign_outputs_from_tensor(const torch::Tensor& outputs) {
        const auto patch0Size = patch_output<0>().as_tensor().size(0);
        const auto patch1Size = patch_output<1>().as_tensor().size(0);
        auto patch0Tensor = outputs.slice(0, 0, patch0Size);
        auto patch1Tensor = outputs.slice(0, patch0Size, patch0Size + patch1Size).clone();

        for (const auto& interfaceCfg : interfaces_) {
            int sidePatch0 = -1;
            int sidePatch1 = -1;
            bool reverse = interfaceCfg.orientation == "reversed";

            if (interfaceCfg.patch_a == 0 && interfaceCfg.patch_b == 1) {
                sidePatch0 = interfaceCfg.side_a;
                sidePatch1 = interfaceCfg.side_b;
            } else if (interfaceCfg.patch_a == 1 && interfaceCfg.patch_b == 0) {
                sidePatch0 = interfaceCfg.side_b;
                sidePatch1 = interfaceCfg.side_a;
            } else {
                throw std::runtime_error(
                    "2-patch strong coupling currently expects interfaces between patch 0 and 1.");
            }

            auto masterIds = side_control_point_indices<0>(sidePatch0, false);
            auto slaveIds = side_control_point_indices<1>(sidePatch1, reverse);

            auto masterIndexTensor = torch::tensor(
                masterIds,
                torch::TensorOptions().dtype(torch::kInt64).device(patch0Tensor.device()));
            auto slaveIndexTensor = torch::tensor(
                slaveIds,
                torch::TensorOptions().dtype(torch::kInt64).device(patch1Tensor.device()));

            const int64_t nPatch1Cps = patch1Size / 2;
            auto patch0Ux = patch0Tensor.slice(0, 0, patch0Size / 2);
            auto patch0Uy = patch0Tensor.slice(0, patch0Size / 2, patch0Size);
            auto patch1Ux = patch1Tensor.slice(0, 0, nPatch1Cps);
            auto patch1Uy = patch1Tensor.slice(0, nPatch1Cps, patch1Size);

            patch1Ux.index_put_({slaveIndexTensor}, patch0Ux.index_select(0, masterIndexTensor));
            patch1Uy.index_put_({slaveIndexTensor}, patch0Uy.index_select(0, masterIndexTensor));
        }

        patch_output<0>().from_tensor(patch0Tensor);
        patch_output<1>().from_tensor(patch1Tensor);
    }

    template <std::size_t Patch>
    void initialize_patch_data() {
        auto& cache = patch_cache<Patch>();
        cache.collPts = Base::template collPts<Patch>(iganet::collPts::greville);
        cache.interiorCollPts = Base::template collPts<Patch>(iganet::collPts::greville_interior);

        cache.nrCollPts = static_cast<int>(std::sqrt(std::get<0>(cache.collPts.first).size(0)));
        cache.var_knot_indices =
            patch_output<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.collPts.first);
        cache.var_coeff_indices =
            patch_output<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.var_knot_indices);
        cache.var_knot_indices_interior =
            patch_output<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.interiorCollPts.first);
        cache.var_coeff_indices_interior =
            patch_output<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.var_knot_indices_interior);
        cache.G_knot_indices =
            patch_input<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.collPts.first);
        cache.G_coeff_indices =
            patch_input<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.G_knot_indices);
        cache.G_knot_indices_interior =
            patch_input<Patch>().template find_knot_indices<iganet::functionspace::interior>(
                cache.interiorCollPts.first);
        cache.G_coeff_indices_interior =
            patch_input<Patch>().template find_coeff_indices<iganet::functionspace::interior>(
                cache.G_knot_indices_interior);
    }

    void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
        append_json_key(JSON_PATH_, key, data);
    }

    template <std::size_t Patch>
    void export_patch_results(const std::string& suffix) {
        auto& cache = patch_cache<Patch>();
        auto jacobian = patch_output<Patch>().ijac(
            patch_input<Patch>(), cache.collPts.first,
            cache.var_knot_indices, cache.var_coeff_indices,
            cache.G_knot_indices, cache.G_coeff_indices);

        auto ux_x = *jacobian[0];
        auto ux_y = *jacobian[1];
        auto uy_x = *jacobian[2];
        auto uy_y = *jacobian[3];

        auto __jac_opts = jacobian[0]->options();
        torch::Tensor sigma_xx = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor sigma_xy = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor sigma_yy = torch::zeros({jacobian[0]->size(0)}, __jac_opts);
        torch::Tensor sigma_vm = torch::zeros({jacobian[0]->size(0)}, __jac_opts);

        nlohmann::json vm_j = nlohmann::json::array();
        nlohmann::json x_j = nlohmann::json::array();
        nlohmann::json y_j = nlohmann::json::array();

        for (int i = 0; i < jacobian[0]->size(0); ++i) {
            sigma_xx[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * ux_x[i];
            sigma_xy[i] = mu_ * (uy_x[i] + ux_y[i]);
            sigma_yy[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * uy_y[i];
            sigma_vm[i] = sqrt(sigma_xx[i] * sigma_xx[i] + sigma_yy[i] * sigma_yy[i]
                               - sigma_xx[i] * sigma_yy[i] + 3 * sigma_xy[i] * sigma_xy[i]);
            vm_j.push_back({sigma_vm[i].template item<double>()});
            x_j.push_back({sigma_xx[i].template item<double>()});
            y_j.push_back({sigma_yy[i].template item<double>()});
        }

        auto collPtsFirstAsTensor = torch::stack(
            {std::get<0>(cache.collPts.first), std::get<1>(cache.collPts.first)}, 1);
        auto displacementOfCollPts = patch_output<Patch>().eval(cache.collPts.first);
        auto displacementAsTensor = torch::stack(
            {*(displacementOfCollPts[0]), *(displacementOfCollPts[1])}, 1);

        nlohmann::json collPts_j = nlohmann::json::array();
        nlohmann::json collPtsDisp_j = nlohmann::json::array();
        for (int i = 0; i < collPtsFirstAsTensor.size(0); ++i) {
            collPts_j.push_back({
                collPtsFirstAsTensor[i][0].template item<double>(),
                collPtsFirstAsTensor[i][1].template item<double>()
            });
            collPtsDisp_j.push_back({
                (collPtsFirstAsTensor[i][0] + displacementAsTensor[i][0]).template item<double>(),
                (collPtsFirstAsTensor[i][1] + displacementAsTensor[i][1]).template item<double>()
            });
        }

        auto hessianColl = patch_output<Patch>().ihess(
            patch_input<Patch>(), cache.interiorCollPts.first,
            cache.var_knot_indices_interior, cache.var_coeff_indices_interior,
            cache.G_knot_indices_interior, cache.G_coeff_indices_interior);

        auto& ux_xx = hessianColl(0,0,0);
        auto& ux_xy = hessianColl(0,1,0);
        auto& ux_yy = hessianColl(1,1,0);
        auto& uy_xx = hessianColl(0,0,1);
        auto& uy_xy = hessianColl(0,1,1);
        auto& uy_yy = hessianColl(1,1,1);

        torch::Tensor divStressX = (lambda_ + 2 * mu_) * ux_xx
                                 + mu_ * ux_yy
                                 + (lambda_ + mu_) * uy_xy;
        torch::Tensor divStressY = mu_ * uy_xx
                                 + (lambda_ + 2 * mu_) * uy_yy
                                 + (lambda_ + mu_) * ux_xy;

        nlohmann::json divX_j = nlohmann::json::array();
        nlohmann::json divY_j = nlohmann::json::array();
        for (int i = 0; i < divStressX.size(0); ++i) {
            divX_j.push_back({divStressX[i].template item<double>()});
            divY_j.push_back({divStressY[i].template item<double>()});
        }

        const int64_t nrCtrlPts = patches_[Patch].nr_ctrl_pts;
        auto geometryAsTensor = patch_input<Patch>().as_tensor();
        auto displacementTensor = patch_output<Patch>().as_tensor();
        auto netCtrlPts = torch::zeros({nrCtrlPts * nrCtrlPts, 2}, geometryAsTensor.options());
        auto netDisplacements = torch::zeros({nrCtrlPts * nrCtrlPts, 2}, displacementTensor.options());

        for (int i = 0; i < nrCtrlPts * nrCtrlPts; ++i) {
            netCtrlPts[i][0] = geometryAsTensor[i];
            netCtrlPts[i][1] = geometryAsTensor[i + nrCtrlPts * nrCtrlPts];
            netDisplacements[i][0] = displacementTensor[i];
            netDisplacements[i][1] = displacementTensor[i + nrCtrlPts * nrCtrlPts];
        }

        auto displacedNetCtrlPts = netCtrlPts + netDisplacements;
        nlohmann::json ctrlPts_j = nlohmann::json::array();
        nlohmann::json originCtrlPts_j = nlohmann::json::array();
        nlohmann::json dispCtrlPts_j = nlohmann::json::array();
        for (int i = 0; i < displacedNetCtrlPts.size(0); ++i) {
            originCtrlPts_j.push_back({
                netCtrlPts[i][0].template item<double>(),
                netCtrlPts[i][1].template item<double>()
            });
            dispCtrlPts_j.push_back({
                netDisplacements[i][0].template item<double>(),
                netDisplacements[i][1].template item<double>()
            });
            ctrlPts_j.push_back({
                displacedNetCtrlPts[i][0].template item<double>(),
                displacedNetCtrlPts[i][1].template item<double>()
            });
        }

        appendToJsonFile("net_" + suffix + "_VmStresses", vm_j);
        appendToJsonFile("net_" + suffix + "_XStresses", x_j);
        appendToJsonFile("net_" + suffix + "_YStresses", y_j);
        appendToJsonFile("net_" + suffix + "_collPtsFirstAsTensor", collPts_j);
        appendToJsonFile("net_" + suffix + "_collPtsFirstAfterDisplacementAsTensor", collPtsDisp_j);
        appendToJsonFile("net_" + suffix + "_DivergenceX", divX_j);
        appendToJsonFile("net_" + suffix + "_DivergenceY", divY_j);
        appendToJsonFile("net_" + suffix + "_OriginCtrlPts", originCtrlPts_j);
        appendToJsonFile("net_" + suffix + "_Displacements", dispCtrlPts_j);
        appendToJsonFile("net_" + suffix + "_CtrlPts", ctrlPts_j);
        appendToJsonFile("net_" + suffix + "_Degree", DEGREE_);
    }

    template <std::size_t Patch>
    void append_patch_ctrl_data(nlohmann::json& origin, nlohmann::json& disp, nlohmann::json& ctrl) {
        const int64_t nrCtrlPts = patches_[Patch].nr_ctrl_pts;
        auto geometryAsTensor = patch_input<Patch>().as_tensor();
        auto displacementTensor = patch_output<Patch>().as_tensor();

        for (int i = 0; i < nrCtrlPts * nrCtrlPts; ++i) {
            const double x = geometryAsTensor[i].template item<double>();
            const double y = geometryAsTensor[i + nrCtrlPts * nrCtrlPts].template item<double>();
            const double ux = displacementTensor[i].template item<double>();
            const double uy = displacementTensor[i + nrCtrlPts * nrCtrlPts].template item<double>();

            origin.push_back({x, y});
            disp.push_back({ux, uy});
            ctrl.push_back({x + ux, y + uy});
        }
    }

    double lambda_;
    double mu_;
    std::array<PatchConfig, 2> patches_;
    std::vector<PatchInterfaceConfig> interfaces_;
    std::array<PatchCache, 2> patchCaches_;
    std::tuple<Variable, Variable> ref_;
    int DEGREE_;
    int MAX_EPOCH_;
    double MIN_LOSS_;
    std::string JSON_PATH_;
    std::pair<double, double> BODY_FORCE_;
    bool SUPERVISED_LEARNING_;
    double INTERFACE_TRACTION_WEIGHT_ = 5.0;

public:
    template <typename... Args>
    linear_elasticity_2patch(double lambda, double mu, bool supervisedLearning,
                             int degree, int maxEpoch, double minLoss,
                             std::pair<double, double> bodyForce,
                             std::vector<PatchConfig> patches,
                             std::vector<PatchInterfaceConfig> interfaces,
                             std::string jsonPath,
                             std::vector<int64_t>&& layers,
                             std::vector<std::vector<std::any>>&& activations,
                             Args&&... args)
        : Base(std::forward<std::vector<int64_t>>(layers),
               std::forward<std::vector<std::vector<std::any>>>(activations),
               std::forward<Args>(args)...),
          lambda_(lambda),
          mu_(mu),
          patches_{patches.at(0), patches.at(1)},
          interfaces_(std::move(interfaces)),
          patchCaches_{},
          ref_(make_reference_tuple(patches_)),
          DEGREE_(degree),
          MAX_EPOCH_(maxEpoch),
          MIN_LOSS_(minLoss),
          JSON_PATH_(std::move(jsonPath)),
          BODY_FORCE_(bodyForce),
          SUPERVISED_LEARNING_(supervisedLearning) {}

    auto& ref0() { return std::get<0>(ref_); }
    auto& ref1() { return std::get<1>(ref_); }

    void initialize_problem_data() {
        initialize_patch_data<0>();
        initialize_patch_data<1>();
    }

    bool epoch(int64_t epoch) override {
        std::cout << "Epoch: " << epoch << std::endl;
        return epoch == 0;
    }

    torch::Tensor loss(const torch::Tensor& outputs, int64_t) override {
        if (SUPERVISED_LEARNING_) {
            throw std::runtime_error("Supervised learning is not yet implemented for the 2-patch prototype.");
        }

        assign_outputs_from_tensor(outputs);

        auto lossPDE = compute_patch_pde_loss<0>() + compute_patch_pde_loss<1>();
        auto lossBC = compute_patch_bc_loss<0>(outputs.options()) +
                      compute_patch_bc_loss<1>(outputs.options());
        auto lossINTER = compute_interface_loss(outputs.options());
        auto totalLoss = lossPDE + lossBC + lossINTER;

        std::cout << std::setw(11) << totalLoss.template item<double>()
                  << " = PDE " << std::setw(10) << lossPDE.template item<double>()
                  << " + BC " << std::setw(10) << lossBC.template item<double>()
                  << " + INTER " << std::setw(10) << lossINTER.template item<double>()
                  << std::endl;
        return totalLoss;
    }

    void PostProc() {
        export_patch_results<0>("patch0");
        export_patch_results<1>("patch1");

        nlohmann::json allOrigin = nlohmann::json::array();
        nlohmann::json allDisp = nlohmann::json::array();
        nlohmann::json allCtrl = nlohmann::json::array();
        append_patch_ctrl_data<0>(allOrigin, allDisp, allCtrl);
        append_patch_ctrl_data<1>(allOrigin, allDisp, allCtrl);

        nlohmann::json patchIds = nlohmann::json::array();
        for (const auto& patch : patches_) {
            patchIds.push_back(patch.id);
        }

        nlohmann::json interfaces_j = nlohmann::json::array();
        for (const auto& interfaceCfg : interfaces_) {
            interfaces_j.push_back({
                {"patch_a", interfaceCfg.patch_a},
                {"side_a", interfaceCfg.side_a},
                {"patch_b", interfaceCfg.patch_b},
                {"side_b", interfaceCfg.side_b},
                {"orientation", interfaceCfg.orientation}
            });
        }

        appendToJsonFile("net_Patches", patchIds);
        appendToJsonFile("net_Interfaces", interfaces_j);
        appendToJsonFile("net_OriginCtrlPts", allOrigin);
        appendToJsonFile("net_Displacements", allDisp);
        appendToJsonFile("net_CtrlPts", allCtrl);
        appendToJsonFile("net_Degree", DEGREE_);
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

    const std::filesystem::path CONFIG_PATH = repo_root / "sim_config.json";
    const std::filesystem::path RESULT_JSON_PATH = repo_root / "result.json";  // output file

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
    std::string JSON_PATH;  // result.json path (output)

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
    std::vector<PatchConfig> PATCHES;
    std::vector<PatchInterfaceConfig> INTERFACES;

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

        // IMPORTANT: output result.json is fixed in repo root
        JSON_PATH = RESULT_JSON_PATH.string();

        // spline
        NR_CTRL_PTS = require(j, "spline.nr_ctrl_pts").get<int64_t>();
        DEGREE_CFG = require(j, "spline.degree").get<int>();

        PATCHES = parse_patch_configs(j, NR_CTRL_PTS);
        INTERFACES = parse_patch_interfaces(j);

        if (PATCHES.empty()) {
            throw std::runtime_error("At least one patch must be configured.");
        }

        if (PATCHES.size() > 2) {
            throw std::runtime_error(
                "Current multipatch scaffold supports at most 2 patches.");
        }

        FORCE_SIDES.clear();
        DIRI_SIDES.clear();

        for (const auto& side : PATCHES.front().boundary_conditions.force_sides) {
            FORCE_SIDES.emplace_back(side.side, side.x, side.y);
        }

        for (const auto& side : PATCHES.front().boundary_conditions.diri_sides) {
            DIRI_SIDES.emplace_back(side.side, side.x, side.y);
        }

        TFBC_SIDES = PATCHES.front().boundary_conditions.tfbc_sides;

        if (!INTERFACES.empty() && PATCHES.size() != 2) {
            throw std::runtime_error(
                "Interfaces require exactly 2 patches in the current scaffold.");
        }

        // body force
        {
            const auto& bf = require(j, "body_force");
            BODY_FORCE.first = bf.at(0).get<double>();
            BODY_FORCE.second = bf.at(1).get<double>();
        }

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

    {
        std::ofstream reset_result_json(RESULT_JSON_PATH);
        if (!reset_result_json.is_open()) {
            std::cerr << "Could not reset result file: " << RESULT_JSON_PATH << "\n";
            return 1;
        }
        reset_result_json << "{}\n";
    }

    if (PATCHES.size() == 2) {
        std::cout << "Configured 2-patch scaffold with "
                  << INTERFACES.size() << " interface(s). "
                  << "Strong displacement coupling and weak traction coupling are active."
                  << std::endl;
    } else {
        const std::string cmd =
            "cd \"" + repo_root.string() + "\" && python3 run_std_coll.py";

        const int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "ERROR: python reference run (run_std_coll.py) failed. system() returned " << ret << "\n";
            return 1;
        }
    }
        
    // calculation of lame parameters
    double lambda = (YOUNG_MODULUS * POISSON_RATIO) / 
                    ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
    double mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO));

    auto run = [&]<int DEGREE>() -> int {
        using real_t = double;
        using namespace iganet::literals;
        using optimizer_t = torch::optim::LBFGS;
        using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
        using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
        using linear_elasticity_t = linear_elasticity<optimizer_t, geometry_t, variable_t>;
        using linear_elasticity_2patch_t = linear_elasticity_2patch<optimizer_t, geometry_t, variable_t>;

        if (PATCHES.size() == 2) {
            linear_elasticity_2patch_t net(
                lambda, mu, SUPERVISED_LEARNING, DEGREE, MAX_EPOCH, MIN_LOSS,
                BODY_FORCE, PATCHES, INTERFACES, JSON_PATH,
                {40, 40, 40},
                {{iganet::activation::sigmoid}, {iganet::activation::sigmoid},
                 {iganet::activation::sigmoid}, {iganet::activation::none}},
                std::tuple(
                    iganet::utils::to_array(PATCHES[0].nr_ctrl_pts, PATCHES[0].nr_ctrl_pts),
                    iganet::utils::to_array(PATCHES[1].nr_ctrl_pts, PATCHES[1].nr_ctrl_pts)),
                std::tuple(
                    iganet::utils::to_array(PATCHES[0].nr_ctrl_pts, PATCHES[0].nr_ctrl_pts),
                    iganet::utils::to_array(PATCHES[1].nr_ctrl_pts, PATCHES[1].nr_ctrl_pts)));

            net.template output<0>().transform([=](const std::array<real_t, 2>) {
                return std::array<real_t, 2>{BODY_FORCE.first, BODY_FORCE.second};
            });
            net.template output<1>().transform([=](const std::array<real_t, 2>) {
                return std::array<real_t, 2>{BODY_FORCE.first, BODY_FORCE.second};
            });

            // Patch 1 reuses the Greville-based control-point layout of patch 0
            // and is translated by +1 in x so the shared interface stays aligned.
            {
                auto patch0Geometry = net.template input<0>().as_tensor().clone();
                const auto nPatch0 = PATCHES[0].nr_ctrl_pts * PATCHES[0].nr_ctrl_pts;
                patch0Geometry.slice(0, 0, nPatch0).add_(1.0);
                net.template input<1>().from_tensor(patch0Geometry);
            }

            for (const auto& side : PATCHES[0].boundary_conditions.diri_sides) {
                switch (side.side) {
                    case 1:
                        net.ref0().boundary().template side<1>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref0().boundary().template side<1>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    case 2:
                        net.ref0().boundary().template side<2>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref0().boundary().template side<2>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    case 3:
                        net.ref0().boundary().template side<3>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref0().boundary().template side<3>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    case 4:
                        net.ref0().boundary().template side<4>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref0().boundary().template side<4>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    default:
                        throw std::runtime_error("Invalid Dirichlet side on patch 0.");
                }
            }

            for (const auto& side : PATCHES[1].boundary_conditions.diri_sides) {
                switch (side.side) {
                    case 1:
                        net.ref1().boundary().template side<1>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref1().boundary().template side<1>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    case 2:
                        net.ref1().boundary().template side<2>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref1().boundary().template side<2>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    case 3:
                        net.ref1().boundary().template side<3>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref1().boundary().template side<3>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    case 4:
                        net.ref1().boundary().template side<4>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.x}; },
                            std::array<iganet::short_t, 1>{0});
                        net.ref1().boundary().template side<4>().template transform<1>(
                            [=](const std::array<real_t, 1>&) { return std::array<real_t, 1>{side.y}; },
                            std::array<iganet::short_t, 1>{1});
                        break;
                    default:
                        throw std::runtime_error("Invalid Dirichlet side on patch 1.");
                }
            }

            net.options().max_epoch(MAX_EPOCH);
            net.options().min_loss(MIN_LOSS);
            net.options().min_loss_change(0.0);
            net.options().min_loss_rel_change(0.0);
            net.initialize_problem_data();

            auto t1 = std::chrono::high_resolution_clock::now();
            net.train();
            auto t2 = std::chrono::high_resolution_clock::now();
            iganet::Log(iganet::log::info)
                << "Training took "
                << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
                << " seconds\n";

            net.PostProc();
            return 0;
        }

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

        net.initialize_problem_data();

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

        net.PostProc();

        // PROCESSING NETWORK OUTPUT FOR SPLINEPY

        // get the geometry and displacement as tensors
        torch::Tensor geometryAsTensor = net.template input<0>().as_tensor();
        torch::Tensor displacementAsTensor = net.template output<0>().as_tensor();
        
        // creating collection matrix for all the control points (iganet)
        torch::Tensor netCtrlPts = torch::zeros({NR_CTRL_PTS * NR_CTRL_PTS, 2}, geometryAsTensor.options());
        // creating collection matrix for all the displacements (iganet)
        torch::Tensor netDisplacements = torch::zeros({NR_CTRL_PTS * NR_CTRL_PTS, 2}, displacementAsTensor.options());

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
        
            torch::Tensor gsOriginCtrlPts;
            torch::Tensor gsDisplacements;
            torch::Tensor gsCtrlPts;
            torch::Tensor gsStresses;
            
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
                torch::Tensor gsRefOriginCtrlPts;
                torch::Tensor gsRefCtrlPts;
                torch::Tensor gsRefDisplacements;
                torch::Tensor gsRefStresses;

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
    
    switch (DEGREE_CFG) {
    case 2: return run.template operator()<2>();
    case 3: return run.template operator()<3>();
    case 4: return run.template operator()<4>();
    case 5: return run.template operator()<5>();
    case 6: return run.template operator()<6>();
    default: std::cerr << "Error: Invalid degree " << DEGREE_CFG << " (2..6)\n" << std::endl;
        return 1;
    }

    iganet::finalize();
    return 0;
}
