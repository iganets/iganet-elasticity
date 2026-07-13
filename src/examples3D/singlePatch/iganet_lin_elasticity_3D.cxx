#include <iganet.h>
#include <iostream>
#include <fstream>
#include <utils/config.hpp>
#include <utils/paths.hpp>

using namespace iganet::literals;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using iganet_elasticity::utils::config::require;

/// @brief Specialization of the IgANet class for linear elasticity in 3D 
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

    // Collocation points
    typename Base::template collPts_t<0> collPts_;
    typename Base::template collPts_t<0> interiorCollPts_;

    // indices for fast jac/hess evaluation
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
    // derived / convenience
    int nrCollPts_; 
    // reference field for Dirichlet targets (same spline space as variable)
    typename std::tuple_element_t<0, Outputs> ref_;


    // simulation parameters
    int MAX_EPOCH_;
    double MIN_LOSS_;
    int64_t NR_CTRL_PTS_;

    //body force now 3D
    std::array<double, 3> BODY_FORCE_;

    // Dirichlet: (side, ux, uy, uz)
    // Neumann traction: (side, tx, ty, tz)  (optional)
    // traction-free sides: side list (optional)
    std::vector<std::tuple<int, double, double, double>> DIRI_SIDES_;
    std::vector<std::tuple<int, double, double, double>> FORCE_SIDES_;
    std::vector<int> TFBC_SIDES_;

    std::string JSON_PATH_;
    bool SUPERVISED_LEARNING_;

    bool isDirichletSide(int sideNr) const {
        return std::any_of(
            DIRI_SIDES_.begin(), DIRI_SIDES_.end(),
            [&](const auto& t) { return std::get<0>(t) == sideNr; }
        );
    }

    bool isNeumannSide(int sideNr) const {
        return std::any_of(
            FORCE_SIDES_.begin(), FORCE_SIDES_.end(),
            [&](const auto& t) { return std::get<0>(t) == sideNr; }
        );
    }

    // Priority:
    // Dirichlet   -> 3
    // Neumann     -> 2
    // TractionFree-> 1
    int bc_priority(int sideNr) const {
        if (isDirichletSide(sideNr)) {
            return 3;
        }
        if (isNeumannSide(sideNr)) {
            return 2;
        }
        return 1; // otherwise traction-free
    }

    // true if otherSide has priority over sideNr
    bool bc_other_wins(int otherSide, int sideNr) const {
        int otherPriority = bc_priority(otherSide);
        int thisPriority  = bc_priority(sideNr);

        if (otherPriority > thisPriority) {
            return true;
        }

        if (otherPriority == thisPriority && otherSide < sideNr) {
            return true;
        }

        return false;
    }

    // opposite faces do not intersect, all other distinct side pairs do
    bool sidesIntersect(int sideA, int sideB) const {
        if (sideA == sideB) {
            return false;
        }

        if ((sideA == 1 && sideB == 2) || (sideA == 2 && sideB == 1) ||
            (sideA == 3 && sideB == 4) || (sideA == 4 && sideB == 3) ||
            (sideA == 5 && sideB == 6) || (sideA == 6 && sideB == 5)) {
            return false;
        }

        return true;
    }

    // returns full 3D face coordinates of boundary collocation points
    std::array<torch::Tensor, 3> getFaceBoundaryPoints(int sideNr) const {
        switch (sideNr) {
            case 1: { // x = 0, local coords = (y,z)
                at::Tensor Y = std::get<0>(collPts_.boundary())[0];
                at::Tensor Z = std::get<0>(collPts_.boundary())[1];
                return {torch::zeros_like(Y), Y, Z};
            }
            case 2: { // x = 1, local coords = (y,z)
                at::Tensor Y = std::get<1>(collPts_.boundary())[0];
                at::Tensor Z = std::get<1>(collPts_.boundary())[1];
                return {torch::ones_like(Y), Y, Z};
            }
            case 3: { // y = 0, local coords = (x,z)
                at::Tensor X = std::get<2>(collPts_.boundary())[0];
                at::Tensor Z = std::get<2>(collPts_.boundary())[1];
                return {X, torch::zeros_like(X), Z};
            }
            case 4: { // y = 1, local coords = (x,z)
                at::Tensor X = std::get<3>(collPts_.boundary())[0];
                at::Tensor Z = std::get<3>(collPts_.boundary())[1];
                return {X, torch::ones_like(X), Z};
            }
            case 5: { // z = 0, local coords = (x,y)
                at::Tensor X = std::get<4>(collPts_.boundary())[0];
                at::Tensor Y = std::get<4>(collPts_.boundary())[1];
                return {X, Y, torch::zeros_like(X)};
            }
            case 6: { // z = 1, local coords = (x,y)
                at::Tensor X = std::get<5>(collPts_.boundary())[0];
                at::Tensor Y = std::get<5>(collPts_.boundary())[1];
                return {X, Y, torch::ones_like(X)};
            }
            default:
                throw std::invalid_argument("Boundary side must be 1..6.");
        }
    }

    // mask of points on pts that also lie on otherSide
    torch::Tensor maskPointsOnOtherSide(
        const std::array<torch::Tensor, 3>& pts,
        int otherSide) const
    {
        const auto& X = pts[0];
        const auto& Y = pts[1];
        const auto& Z = pts[2];

        switch (otherSide) {
            case 1: return torch::isclose(X, torch::zeros_like(X)); // x=0
            case 2: return torch::isclose(X, torch::ones_like(X));  // x=1
            case 3: return torch::isclose(Y, torch::zeros_like(Y)); // y=0
            case 4: return torch::isclose(Y, torch::ones_like(Y));  // y=1
            case 5: return torch::isclose(Z, torch::zeros_like(Z)); // z=0
            case 6: return torch::isclose(Z, torch::ones_like(Z));  // z=1
            default:
                throw std::invalid_argument("Boundary side must be 1..6.");
        }
    }

    // general keep-mask for any boundary side:
    // Dirichlet > Neumann > TractionFree
    // same priority -> smaller side number wins
    torch::Tensor buildKeepMaskForSide(int sideNr) const {
        auto pts = getFaceBoundaryPoints(sideNr);
        const auto& X = pts[0];

        torch::Tensor keepMask = torch::ones(
            {X.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(X.device())
        );

        for (int otherSide = 1; otherSide <= 6; ++otherSide) {
            if (!sidesIntersect(sideNr, otherSide)) {
                continue;
            }

            if (!bc_other_wins(otherSide, sideNr)) {
                continue;
            }

            torch::Tensor onOtherSide = maskPointsOnOtherSide(pts, otherSide);
            keepMask = torch::logical_and(keepMask, torch::logical_not(onOtherSide));
        }

        return keepMask;
    }
  
public:
    /// @brief Constructor
    template <typename... Args>
    linear_elasticity(double lambda, double mu, bool SUPERVISED_LEARNING, int MAX_EPOCH, 
                    double MIN_LOSS, std::array<double, 3> BODY_FORCE, std::vector<int> TFBC_SIDES,
                    std::vector<std::tuple<int, double, double, double>> FORCE_SIDES,
                    std::vector<std::tuple<int, double, double, double>> DIRI_SIDES, 
                    int64_t NR_CTRL_PTS, std::string JSON_PATH, std::vector<int64_t> &&layers, 
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
        : Base( std::forward<std::vector<int64_t>>(layers),
                std::forward<std::vector<std::vector<std::any>>>(activations),
                std::forward<Args>(args)...),
                lambda_(lambda), mu_(mu), SUPERVISED_LEARNING_(SUPERVISED_LEARNING), MAX_EPOCH_(MAX_EPOCH), 
                MIN_LOSS_(MIN_LOSS), BODY_FORCE_(BODY_FORCE), TFBC_SIDES_(TFBC_SIDES), FORCE_SIDES_(FORCE_SIDES), 
                DIRI_SIDES_(DIRI_SIDES), NR_CTRL_PTS_(NR_CTRL_PTS), JSON_PATH_(std::move(JSON_PATH)), 
                ref_(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS, NR_CTRL_PTS)) {}

    // /// @brief Returns a constant reference to the collocation point
    // auto const &collPts() const { return collPts_; }

    // /// @brief Returns a constant reference to the interior collocation points
    // auto const &interiorCollPts() const { return interiorCollPts_; }

    /// @brief Returns a constant reference to the reference solution
    auto const &ref() const { return ref_; }

    /// @brief Returns a non-constant reference to the reference solution
    auto &ref() { return ref_; }
    
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
        torch::Tensor stdCollDisplacement = torch::empty({nrStdCollCtrlPts, 3}, options);
    
        // fill the tensor with data from the JSON file
        for (int i = 0; i < nrStdCollCtrlPts; ++i) {
            stdCollDisplacement[i][0] = stdCollDisplacements_j[i][0].get<double>();
            stdCollDisplacement[i][1] = stdCollDisplacements_j[i][1].get<double>();
            stdCollDisplacement[i][2] = stdCollDisplacements_j[i][2].get<double>();
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


            // WARNING, only works for equal number of control points in x, y, and z direction
            nrCollPts_ = static_cast<int>(std::cbrt(collPts_.interior()[0].size(0)));
            torch::Tensor collPtsCoeffs = collPts_.interior()[0].slice(0, 0, nrCollPts_);
            nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
            for (int i = 0; i < collPtsCoeffs.size(0); ++i) {
                collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
            }
            appendToJsonFile("net_collPtsCoeffsRef1", collPtsCoeffs_j);
            appendToJsonFile("net_nrCollPtsRef1", {nrCollPts_});
            
            // precompute indices (output)
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

            // precompute indices (geometry/input)
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
        std::optional<torch::Tensor> supLoss;
        std::optional<torch::Tensor> forceLoss;

        // pre-allocation of the tensors for the traction boundary conditions
        std::optional<torch::Tensor> forceValues;
        std::optional<torch::Tensor> targetForce;
        std::optional<torch::Tensor> tractionFreeValues;
        std::optional<torch::Tensor> tractionZeros;


//-------- TRACTION / NEUMANN BOUNDARY CONDITIONS FOR 3D CUBE ----------------------------------------------
        if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty())
        {
            // collect sides of traction-free and force BCs
            std::vector<int> neumannSides;
            neumannSides.reserve(TFBC_SIDES_.size() + FORCE_SIDES_.size());
            neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
            for (const auto& force : FORCE_SIDES_) {
                neumannSides.push_back(std::get<0>(force));
            }
             // static storage of all traction collocation points
            static std::array<torch::Tensor, 3ul> tractionCollPts;
            static std::vector<int> nPtsPerSide;

            if (epoch == 0 && !tractionCollPts[0].defined())
            {
                std::vector<torch::Tensor> tractionCollPtsX;
                std::vector<torch::Tensor> tractionCollPtsY;
                std::vector<torch::Tensor> tractionCollPtsZ;
                nPtsPerSide.clear();

                auto make_face_points = [&](int side) -> std::array<torch::Tensor, 3ul>
{
                switch (side) {
                    case 1: { // x = 0, local coords = (y,z)
                        at::Tensor Y = std::get<0>(collPts_.boundary())[0];
                        at::Tensor Z = std::get<0>(collPts_.boundary())[1];
                        return {torch::zeros_like(Y), Y, Z};
                    }
                    case 2: { // x = 1, local coords = (y,z)
                        at::Tensor Y = std::get<1>(collPts_.boundary())[0];
                        at::Tensor Z = std::get<1>(collPts_.boundary())[1];
                        return {torch::ones_like(Y), Y, Z};
                    }
                    case 3: { // y = 0, local coords = (x,z)
                        at::Tensor X = std::get<2>(collPts_.boundary())[0];
                        at::Tensor Z = std::get<2>(collPts_.boundary())[1];
                        return {X, torch::zeros_like(X), Z};
                    }
                    case 4: { // y = 1, local coords = (x,z)
                        at::Tensor X = std::get<3>(collPts_.boundary())[0];
                        at::Tensor Z = std::get<3>(collPts_.boundary())[1];
                        return {X, torch::ones_like(X), Z};
                    }
                    case 5: { // z = 0, local coords = (x,y)
                        at::Tensor X = std::get<4>(collPts_.boundary())[0];
                        at::Tensor Y = std::get<4>(collPts_.boundary())[1];
                        return {X, Y, torch::zeros_like(X)};
                    }
                    case 6: { // z = 1, local coords = (x,y)
                        at::Tensor X = std::get<5>(collPts_.boundary())[0];
                        at::Tensor Y = std::get<5>(collPts_.boundary())[1];
                        return {X, Y, torch::ones_like(X)};
                    }
                    default:
                        throw std::invalid_argument("Side for 3D traction BC has to be 1..6.");
                }
            };

                for (int side : neumannSides)
                {
                    auto facePts = make_face_points(side);
                    at::Tensor X = facePts[0];
                    at::Tensor Y = facePts[1];
                    at::Tensor Z = facePts[2];

                    // unique ownership on edges/corners:
                    // Dirichlet > Force > TractionFree
                    // within same priority: smaller side number wins
                    at::Tensor keepMask = buildKeepMaskForSide(side);

                    at::Tensor idx = torch::nonzero(keepMask).reshape({-1});

                    at::Tensor Xf = X.index_select(0, idx);
                    at::Tensor Yf = Y.index_select(0, idx);
                    at::Tensor Zf = Z.index_select(0, idx);

                    nPtsPerSide.push_back(static_cast<int>(Xf.size(0)));

                    if (Xf.size(0) > 0) {
                        tractionCollPtsX.push_back(Xf);
                        tractionCollPtsY.push_back(Yf);
                        tractionCollPtsZ.push_back(Zf);
                    }
                }

                if (!tractionCollPtsX.empty()) {
                    tractionCollPts = {
                        torch::cat(tractionCollPtsX, 0),
                        torch::cat(tractionCollPtsY, 0),
                        torch::cat(tractionCollPtsZ, 0)
                    };
                } else {
                    tractionCollPts = {
                        torch::empty({0}, outputs.options()),
                        torch::empty({0}, outputs.options()),
                        torch::empty({0}, outputs.options())
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

            // if there are no remaining traction points after masking, skip safely
            if (tractionCollPts[0].numel() > 0) {
                auto jacobianBoundary = this->template output<0>().ijac(
                    this->template input<0>(),
                    tractionCollPts,
                    var_knot_indices_boundary_, var_coeff_indices_boundary_,
                    G_knot_indices_boundary_, G_coeff_indices_boundary_);

                // 3D displacement gradient entries
                auto ux_x = *jacobianBoundary[0];
                auto ux_y = *jacobianBoundary[1];
                auto ux_z = *jacobianBoundary[2];

                auto uy_x = *jacobianBoundary[3];
                auto uy_y = *jacobianBoundary[4];
                auto uy_z = *jacobianBoundary[5];

                auto uz_x = *jacobianBoundary[6];
                auto uz_y = *jacobianBoundary[7];
                auto uz_z = *jacobianBoundary[8];

                torch::Tensor tractionValuesX = torch::zeros({tractionCollPts[0].size(0)}, ux_x.options());
                torch::Tensor tractionValuesY = torch::zeros({tractionCollPts[0].size(0)}, ux_x.options());
                torch::Tensor tractionValuesZ = torch::zeros({tractionCollPts[0].size(0)}, ux_x.options());

                int pointCtr = 0;
                int sideCtr  = 0;

                for (int side : neumannSides)
                {
                    int n_vals = nPtsPerSide[sideCtr];

                    for (int i = 0; i < n_vals; ++i)
                    {
                        int idx = pointCtr + i;

                        if (side == 1) { // x = 0, n = (-1,0,0)
                            tractionValuesX[idx] = -((lambda_ + 2.0 * mu_) * ux_x[idx]
                                                + lambda_ * uy_y[idx]
                                                + lambda_ * uz_z[idx]);
                            tractionValuesY[idx] = -(mu_ * (ux_y[idx] + uy_x[idx]));
                            tractionValuesZ[idx] = -(mu_ * (ux_z[idx] + uz_x[idx]));
                        }
                        else if (side == 2) { // x = 1, n = (1,0,0)
                            tractionValuesX[idx] =  ((lambda_ + 2.0 * mu_) * ux_x[idx]
                                                + lambda_ * uy_y[idx]
                                                + lambda_ * uz_z[idx]);
                            tractionValuesY[idx] =  (mu_ * (ux_y[idx] + uy_x[idx]));
                            tractionValuesZ[idx] =  (mu_ * (ux_z[idx] + uz_x[idx]));
                        }
                        else if (side == 3) { // y = 0, n = (0,-1,0)
                            tractionValuesX[idx] = -(mu_ * (ux_y[idx] + uy_x[idx]));
                            tractionValuesY[idx] = -(lambda_ * ux_x[idx]
                                                + (lambda_ + 2.0 * mu_) * uy_y[idx]
                                                + lambda_ * uz_z[idx]);
                            tractionValuesZ[idx] = -(mu_ * (uy_z[idx] + uz_y[idx]));
                        }
                        else if (side == 4) { // y = 1, n = (0,1,0)
                            tractionValuesX[idx] =  (mu_ * (ux_y[idx] + uy_x[idx]));
                            tractionValuesY[idx] =  (lambda_ * ux_x[idx]
                                                + (lambda_ + 2.0 * mu_) * uy_y[idx]
                                                + lambda_ * uz_z[idx]);
                            tractionValuesZ[idx] =  (mu_ * (uy_z[idx] + uz_y[idx]));
                        }
                        else if (side == 5) { // z = 0, n = (0,0,-1)
                            tractionValuesX[idx] = -(mu_ * (ux_z[idx] + uz_x[idx]));
                            tractionValuesY[idx] = -(mu_ * (uy_z[idx] + uz_y[idx]));
                            tractionValuesZ[idx] = -(lambda_ * ux_x[idx]
                                                + lambda_ * uy_y[idx]
                                                + (lambda_ + 2.0 * mu_) * uz_z[idx]);
                        }
                        else if (side == 6) { // z = 1, n = (0,0,1)
                            tractionValuesX[idx] =  (mu_ * (ux_z[idx] + uz_x[idx]));
                            tractionValuesY[idx] =  (mu_ * (uy_z[idx] + uz_y[idx]));
                            tractionValuesZ[idx] =  (lambda_ * ux_x[idx]
                                                + lambda_ * uy_y[idx]
                                                + (lambda_ + 2.0 * mu_) * uz_z[idx]);
                        }
                        else {
                            throw std::invalid_argument("Side for 3D traction BC has to be 1..6.");
                        }
                    }

                    pointCtr += n_vals;
                    sideCtr++;
                }

                torch::Tensor tractionValues =
                    torch::stack({tractionValuesX, tractionValuesY, tractionValuesZ}, 1);

                if (!FORCE_SIDES_.empty())
                {
                    int cutlength = 0;
                    int forceSize = static_cast<int>(FORCE_SIDES_.size());

                    for (int i = static_cast<int>(nPtsPerSide.size()) - forceSize;
                        i < static_cast<int>(nPtsPerSide.size()); ++i) {
                        cutlength += nPtsPerSide[i];
                    }

                    tractionFreeValues.emplace(
                        tractionValues.slice(0, 0, tractionValues.size(0) - cutlength));
                    tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));

                    forceValues.emplace(
                        tractionValues.slice(0, tractionValues.size(0) - cutlength, tractionValues.size(0)));
                    targetForce.emplace(torch::zeros_like(*forceValues));

                    int offset = 0;
                    int startIdx = static_cast<int>(nPtsPerSide.size()) - forceSize;

                    for (size_t i = 0; i < FORCE_SIDES_.size(); ++i)
                    {
                        int reducedPts = nPtsPerSide[startIdx + static_cast<int>(i)];
                        auto rowSlice = (*targetForce).slice(0, offset, offset + reducedPts);

                        rowSlice.slice(1, 0, 1).fill_(std::get<1>(FORCE_SIDES_[i])); // x-force
                        rowSlice.slice(1, 1, 2).fill_(std::get<2>(FORCE_SIDES_[i])); // y-force
                        rowSlice.slice(1, 2, 3).fill_(std::get<3>(FORCE_SIDES_[i])); // z-force

                        offset += reducedPts;
                    }
                }
                else
                {
                    tractionFreeValues.emplace(tractionValues);
                    tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
                }
            }
        }

//----------- LINEAR ELASTICITY EQUATION------------------------------------------------------------------------

        // calculation of the second derivatives of the displacements (u)
        auto hessianColl = this->template output<0>().ihess(this->template input<0>(), interiorCollPts_.interior(), 
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_, G_coeff_indices_interior_);

        // partial derivatives of the displacements (u)
        auto& ux_xx = hessianColl(0,0,0);
        auto& ux_xy = hessianColl(0,1,0);
        auto& ux_xz = hessianColl(0,2,0);
        auto& ux_yx = hessianColl(1,0,0);
        auto& ux_yy = hessianColl(1,1,0);
        auto& ux_yz = hessianColl(1,2,0);
        auto& ux_zx = hessianColl(2,0,0);
        auto& ux_zy = hessianColl(2,1,0);
        auto& ux_zz = hessianColl(2,2,0);

        auto& uy_xx = hessianColl(0,0,1);
        auto& uy_xy = hessianColl(0,1,1);
        auto& uy_xz = hessianColl(0,2,1);
        auto& uy_yx = hessianColl(1,0,1);
        auto& uy_yy = hessianColl(1,1,1);
        auto& uy_yz = hessianColl(1,2,1);
        auto& uy_zx = hessianColl(2,0,1);
        auto& uy_zy = hessianColl(2,1,1);
        auto& uy_zz = hessianColl(2,2,1);

        auto& uz_xx = hessianColl(0,0,2);
        auto& uz_xy = hessianColl(0,1,2);
        auto& uz_xz = hessianColl(0,2,2);
        auto& uz_yx = hessianColl(1,0,2);
        auto& uz_yy = hessianColl(1,1,2);
        auto& uz_yz = hessianColl(1,2,2);
        auto& uz_zx = hessianColl(2,0,2);
        auto& uz_zy = hessianColl(2,1,2);
        auto& uz_zz = hessianColl(2,2,2);

        int64_t size = hessianColl(0,0,0).size(0);

        auto opts = hessianColl(0,0,0).options();

        // pre-allocation of the results
        torch::Tensor divStressX = torch::zeros({size}, opts);
        torch::Tensor divStressY = torch::zeros({size}, opts);
        torch::Tensor divStressZ = torch::zeros({size}, opts);

        torch::Tensor divZeros = torch::stack({divStressX, divStressY, divStressZ}, /*dim=*/1);
        
        for (int i = 0; i < size; ++i) {
            divStressX[i] =
                (lambda_ + 2.0 * mu_) * ux_xx[i]
            + mu_ * ux_yy[i]
            + mu_ * ux_zz[i]
            + (lambda_ + mu_) * (uy_xy[i] + uz_xz[i]);

            divStressY[i] =
                mu_ * uy_xx[i]
            + (lambda_ + 2.0 * mu_) * uy_yy[i]
            + mu_ * uy_zz[i]
            + (lambda_ + mu_) * (ux_yx[i] + uz_yz[i]);

            divStressZ[i] =
                mu_ * uz_xx[i]
            + mu_ * uz_yy[i]
            + (lambda_ + 2.0 * mu_) * uz_zz[i]
            + (lambda_ + mu_) * (ux_zx[i] + uy_zy[i]);
        }

    torch::Tensor divStress = torch::stack({divStressX, divStressY, divStressZ}, 1);
       //for (int i = 0; i < size; ++i) {

            // d/dx(div(u)) -> ux_xx + uy_yx + uz_zx
            // d/dy(div(u)) -> ux_xy + uy_yy + uz_zy
            // d/dz(div(u)) -> ux_xz + uy_yz + uz_zz

            // Laplacian terms
            //const auto lapUx = ux_xx[i] + ux_yy[i] + ux_zz[i];
            //const auto lapUy = uy_xx[i] + uy_yy[i] + uy_zz[i];
            //const auto lapUz = uz_xx[i] + uz_yy[i] + uz_zz[i];

            //const auto dDiv_dx = ux_xx[i] + uy_yx[i] + uz_zx[i];
            //const auto dDiv_dy = ux_xy[i] + uy_yy[i] + uz_zy[i];
            //const auto dDiv_dz = ux_xz[i] + uy_yz[i] + uz_zz[i];

            //divStressX[i] = mu_ * lapUx + (lambda_ + mu_) * dDiv_dx;
            //divStressY[i] = mu_ * lapUy + (lambda_ + mu_) * dDiv_dy;
            //divStressZ[i] = mu_ * lapUz + (lambda_ + mu_) * dDiv_dz;
        //}
         
        // create a tensor of the divergence of the stress tensor
        //torch::Tensor divStress = torch::stack({divStressX, divStressY, divStressZ}, /*dim=*/1);

        // BODY FORCE: constant vector (fx, fy)
        //auto opts = divStress.options();  // device + dtype passend zu divStress -> bereits davor definiert

        torch::Tensor bodyForce = torch::tensor(
            {BODY_FORCE_[0], BODY_FORCE_[1], BODY_FORCE_[2]},
            opts
        ).view({1, 3}).repeat({divStress.size(0), 1});   // (N,3)

        // UNSUPERVISED LEARNING (default)
        if (SUPERVISED_LEARNING_ == false) {

            // create command line output variable for all the different losses
            std::ostringstream singleLossOutput;

            // calculation of the loss function for double-sided constraint solid
            // div(sigma) + f = 0 --> div(sigma) = -f
            elastLoss = torch::mse_loss(divStress, -bodyForce);
            
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
                int bcWeight = 1e5;
                // initialize bcLoss variable
                bcLoss = torch::tensor(0.0,outputs.options());

                // evaluation of the displacements at the boundary points
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.boundary());
                // evaluation of the displacements at the reference boundary points
                auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.boundary());

                auto masked_side_loss = [&](const torch::Tensor& u0,
                            const torch::Tensor& u1,
                            const torch::Tensor& u2,
                            const torch::Tensor& b0,
                            const torch::Tensor& b1,
                            const torch::Tensor& b2,
                            int sideNr) -> torch::Tensor
            {
                torch::Tensor keepMask = buildKeepMaskForSide(sideNr);
                torch::Tensor keepIdx = torch::nonzero(keepMask).reshape({-1});

                if (keepIdx.numel() == 0) {
                    return torch::zeros({}, outputs.options());
                }

                return torch::mse_loss(u0.index_select(0, keepIdx), b0.index_select(0, keepIdx)) +
                    torch::mse_loss(u1.index_select(0, keepIdx), b1.index_select(0, keepIdx)) +
                    torch::mse_loss(u2.index_select(0, keepIdx), b2.index_select(0, keepIdx));
            };

            auto add_masked_side_loss = [&](const auto& u_side, const auto& b_side, int sideNr) {
                *bcLoss += bcWeight * masked_side_loss(
                    *u_side[0], *u_side[1], *u_side[2],
                    *b_side[0], *b_side[1], *b_side[2],
                    sideNr
                );
            };

                // loop through all dirichlet sides
                for (const auto& side : DIRI_SIDES_) {
                    int sideNr = std::get<0>(side);

                    switch (sideNr) {
                        case 1:
                            add_masked_side_loss(std::get<0>(u_bdr), std::get<0>(bdr), 1);
                            break;
                        case 2:
                            add_masked_side_loss(std::get<1>(u_bdr), std::get<1>(bdr), 2);
                            break;
                        case 3:
                            add_masked_side_loss(std::get<2>(u_bdr), std::get<2>(bdr), 3);
                            break;
                        case 4:
                            add_masked_side_loss(std::get<3>(u_bdr), std::get<3>(bdr), 4);
                            break;
                        case 5:
                            add_masked_side_loss(std::get<4>(u_bdr), std::get<4>(bdr), 5);
                            break;
                        case 6:
                            add_masked_side_loss(std::get<5>(u_bdr), std::get<5>(bdr), 6);
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
                modifiedOutputs.slice(0, 0, outputs.size(0) / 3),
                modifiedOutputs.slice(0, outputs.size(0) / 3, 2 * outputs.size(0) / 3),
                modifiedOutputs.slice(0, 2 * outputs.size(0) / 3, outputs.size(0)),
            }, 1);

            // load the displacements from the std collocation solution
            torch::Tensor stdCollDisplacements_ = loadDisplacements();

            stdCollDisplacements_ = stdCollDisplacements_.to(netDisplacements_.options()); //make sure stdCollDisplacements_ matches dtype/device of netDisplacements

            // supervised loss: MSE of net against standard collocation solution
            int supWeight = 1e7;
            supLoss = supWeight * torch::mse_loss(netDisplacements_, stdCollDisplacements_);

            // calculation of the loss function for double-sided constraint solid
            // div(sigma) + f = 0 --> div(sigma) = -f
            elastLoss = torch::mse_loss(divStress, -bodyForce);

            // add the elasticity loss and supervised loss to the total loss
            totalLoss = *supLoss + elastLoss;

            // add the elasticity and supervised losses to the cmd-output variable
            singleLossOutput << "SL " << std::setw(11) << (*supLoss).item<double>() / supWeight 
                             << " * 1e" << static_cast<int>(std::log10(supWeight))
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
                bcLoss = torch::tensor(0.0,outputs.options());

                // evaluation of the displacements at the boundary points
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.boundary());
                // evaluation of the displacements at the reference boundary points
                auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.boundary());
                auto masked_side_loss = [&](const torch::Tensor& u0,
                            const torch::Tensor& u1,
                            const torch::Tensor& u2,
                            const torch::Tensor& b0,
                            const torch::Tensor& b1,
                            const torch::Tensor& b2,
                            int sideNr) -> torch::Tensor
            {
                torch::Tensor keepMask = buildKeepMaskForSide(sideNr);
                torch::Tensor keepIdx = torch::nonzero(keepMask).reshape({-1});

                if (keepIdx.numel() == 0) {
                    return torch::zeros({}, outputs.options());
                }

                return torch::mse_loss(u0.index_select(0, keepIdx), b0.index_select(0, keepIdx)) +
                    torch::mse_loss(u1.index_select(0, keepIdx), b1.index_select(0, keepIdx)) +
                    torch::mse_loss(u2.index_select(0, keepIdx), b2.index_select(0, keepIdx));
            };

            auto add_masked_side_loss = [&](const auto& u_side, const auto& b_side, int sideNr) {
                *bcLoss += bcWeight * masked_side_loss(
                    *u_side[0], *u_side[1], *u_side[2],
                    *b_side[0], *b_side[1], *b_side[2],
                    sideNr
                );
            };

                // loop through all dirichlet sides
                for (const auto& side : DIRI_SIDES_) {
                    int sideNr = std::get<0>(side);

                    switch (sideNr) {
                        case 1:
                            add_masked_side_loss(std::get<0>(u_bdr), std::get<0>(bdr), 1);
                            break;
                        case 2:
                            add_masked_side_loss(std::get<1>(u_bdr), std::get<1>(bdr), 2);
                            break;
                        case 3:
                            add_masked_side_loss(std::get<2>(u_bdr), std::get<2>(bdr), 3);
                            break;
                        case 4:
                            add_masked_side_loss(std::get<3>(u_bdr), std::get<3>(bdr), 4);
                            break;
                        case 5:
                            add_masked_side_loss(std::get<4>(u_bdr), std::get<4>(bdr), 5);
                            break;
                        case 6:
                            add_masked_side_loss(std::get<5>(u_bdr), std::get<5>(bdr), 6);
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

            // calculate the jacobian of the displacements (u) at the collocation points
            auto jacobian = this->template output<0>().ijac(this->template input<0>(), collPts_.interior(), var_knot_indices_, 
                var_coeff_indices_, G_knot_indices_, G_coeff_indices_);
            
            auto ux_x = *jacobian[0];
            auto ux_y = *jacobian[1];
            auto ux_z = *jacobian[2];
            auto uy_x = *jacobian[3];
            auto uy_y = *jacobian[4];
            auto uy_z = *jacobian[5];
            auto uz_x = *jacobian[6];   
            auto uz_y = *jacobian[7];   
            auto uz_z = *jacobian[8];

            // allocate the stress tensor
            torch::Tensor sigma_xx = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor sigma_xy = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor sigma_xz = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor sigma_yy = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor sigma_yz = torch::zeros({jacobian[0]->size(0)}); 
            torch::Tensor sigma_zz = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor sigma_vm = torch::zeros({jacobian[0]->size(0)});   

            torch::Tensor epsilon_xx = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor epsilon_yy = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor epsilon_zz = torch::zeros({jacobian[0]->size(0)});
            torch::Tensor poisson_re = torch::zeros({jacobian[0]->size(0)});

            // create json object for the stresses
            nlohmann::json netVmStresses_j = nlohmann::json::array();
            nlohmann::json netXStresses_j = nlohmann::json::array();
            nlohmann::json netYStresses_j = nlohmann::json::array();
            nlohmann::json netZStresses_j = nlohmann::json::array();
            nlohmann::json netPoisson_j = nlohmann::json::array();

            // calculate the stress tensor
            for (int i = 0; i < jacobian[0]->size(0); ++i) {
                // calculate the stress values for all collocation points
                sigma_xx[i] = lambda_ * (ux_x[i] + uy_y[i]+ uz_z[i]) + 2 * mu_ * ux_x[i]; //-> normal stress in x-direction
                sigma_xy[i] = mu_ * (uy_x[i] + ux_y[i]); // -> shear stress in xy-direction
                sigma_xz[i] = mu_ * (uz_x[i] + ux_z[i]); // -> shear stress in xz-direction
                sigma_yy[i] = lambda_ * (ux_x[i] + uy_y[i]+ uz_z[i]) + 2 * mu_ * uy_y[i]; //-> normal stress in y-direction
                sigma_yz[i] = mu_ * (uz_y[i] + uy_z[i]); // -> shear stress in yz-direction
                sigma_zz[i] = lambda_ * (ux_x[i] + uy_y[i] + uz_z[i]) + 2 * mu_ * uz_z[i]; //-> normal stress in z-direction

                // calculate von mises stress at the collocation points
                sigma_vm[i] = sqrt(0.5 * (
                (sigma_xx[i] - sigma_yy[i]) * (sigma_xx[i] - sigma_yy[i]) +
                (sigma_yy[i] - sigma_zz[i]) * (sigma_yy[i] - sigma_zz[i]) +
                (sigma_zz[i] - sigma_xx[i]) * (sigma_zz[i] - sigma_xx[i]) +
                6.0 * (sigma_xy[i]*sigma_xy[i] + sigma_yz[i]*sigma_yz[i] + sigma_xz[i]*sigma_xz[i])
            ));
                
                // calculate the strains at the collocation points
                epsilon_xx[i] = ux_x[i];
                epsilon_yy[i] = uy_y[i];
                epsilon_zz[i] = uz_z[i];
                
                /*epsilon_xx[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) * 
                    (sigma_xx[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_yy[i]);
                epsilon_yy[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) * 
                    (sigma_yy[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_xx[i]);
                */
                // only valid for load in x-direction
                // poisson_re[i] = - epsilon_yy[i] / epsilon_xx[i];
                
                // add the stresses to the json objects
                netVmStresses_j.push_back({sigma_vm[i].item<double>()});
                netXStresses_j.push_back({sigma_xx[i].item<double>()});
                netYStresses_j.push_back({sigma_yy[i].item<double>()});
                netZStresses_j.push_back({sigma_zz[i].item<double>()});
                // add the poisson ratio to the json object
                //netPoisson_j.push_back({poisson_re[i].item<double>()});
            }

            // write the stresses and poisson ratios to the json file
            appendToJsonFile("net_VmStresses", netVmStresses_j);
            appendToJsonFile("net_XStresses", netXStresses_j);
            appendToJsonFile("net_YStresses", netYStresses_j);
            appendToJsonFile("net_ZStresses", netZStresses_j);
            //appendToJsonFile("net_Poisson", netPoisson_j);

            // CALCULATE THE NEW POSITION OF THE COLLPTS

            // create a tensor of the collocation points
            torch::Tensor collPtsFirstAsTensor = torch::stack(
                {std::get<0>(collPts_.interior()), std::get<1>(collPts_.interior()), std::get<2>(collPts_.interior())}, 1);
            auto displacementOfCollPts = this->template output<0>().eval(collPts_.interior());
            torch::Tensor displacementAsTensor = torch::stack(
                {*(displacementOfCollPts[0]), *(displacementOfCollPts[1]), *(displacementOfCollPts[2])}, 1);

            // create json objects for the collocation points' reference and displaced position
            nlohmann::json collPtsFirst_j = nlohmann::json::array();
            nlohmann::json collPtsFirstDispl_j = nlohmann::json::array();
            for (int i = 0; i < collPtsFirstAsTensor.size(0); ++i) {
                // reference position of the collocation points
                collPtsFirst_j.push_back({collPtsFirstAsTensor[i][0].item<double>(), 
                                        collPtsFirstAsTensor[i][1].item<double>(), 
                                        collPtsFirstAsTensor[i][2].item<double>()});
                // new position of the collocation points
                collPtsFirstDispl_j.push_back({collPtsFirstAsTensor[i][0].item<double>() + 
                                            displacementAsTensor[i][0].item<double>(), 
                                            collPtsFirstAsTensor[i][1].item<double>() + 
                                            displacementAsTensor[i][1].item<double>(), 
                                            collPtsFirstAsTensor[i][2].item<double>() + 
                                            displacementAsTensor[i][2].item<double>()});
            }
            // write the collocation points' original position to the json file
            appendToJsonFile("net_collPtsFirstAsTensor", collPtsFirst_j);
            // write the collocation points' new position to the json file
            appendToJsonFile("net_collPtsFirstAfterDisplacementAsTensor", collPtsFirstDispl_j);

            // WRITING DIVERGENCE OF THE STRESS TENSOR TO JSON FILE

            nlohmann::json netDivergenceX_j = nlohmann::json::array();
            nlohmann::json netDivergenceY_j = nlohmann::json::array();
            nlohmann::json netDivergenceZ_j = nlohmann::json::array();

            for (int i = 0; i < divStressX.size(0); ++i) {
                netDivergenceX_j.push_back({divStressX[i].item<double>()});
                netDivergenceY_j.push_back({divStressY[i].item<double>()});
                netDivergenceZ_j.push_back({divStressZ[i].item<double>()});
            }

            // write the divergence of the stress tensor to the json file
            appendToJsonFile("net_DivergenceX", netDivergenceX_j);
            appendToJsonFile("net_DivergenceY", netDivergenceY_j);
            appendToJsonFile("net_DivergenceZ", netDivergenceZ_j);
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
        repo_root / "src" / "examples3D" / "singlePatch" / "sim_config_3D_single_patch.json";
    const std::filesystem::path RESULT_JSON_PATH =
        repo_root / "results" / "result_iganet_lin_elasticity_3D.json";  // output file

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

    // run standard collocation simulation with the parameters from the config file 
    const std::string cmd =
        "cd \"" + repo_root.string() + "\" && python3 -m std_collocation_python.run_std_coll src/examples3D/singlePatch/sim_config_3D_single_patch.json";

    const int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "ERROR: python reference run (std_collocation_python/run_std_coll.py) failed. system() returned " << ret << "\n";
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

    // reference simulation parameters
    bool RUN_GS_REF_SIM = false;
    bool RUN_COLL_REF_SIM = false;
    int NR_CTRL_PTS_REF = 0;
    int DEGREE_REF = 0;

    // spline parameters
    int64_t NR_CTRL_PTS = 0;
    int DEGREE_CFG = 0;

    // boundary conditions
    std::vector<std::tuple<int, double, double, double>> FORCE_SIDES;
    std::vector<std::tuple<int, double, double, double>> DIRI_SIDES;
    std::vector<int> TFBC_SIDES;

    // body force
    std::array<double, 3> BODY_FORCE{0.0, 0.0, 0.0};

    try {
        // material
        YOUNG_MODULUS = require(j, "material.young_modulus").get<double>();
        POISSON_RATIO = require(j, "material.poisson_ratio").get<double>();

        // simulation
        MAX_EPOCH = require(j, "simulation.max_epoch").get<int>();
        MIN_LOSS = require(j, "simulation.min_loss").get<double>();
        SUPERVISED_LEARNING = require(j, "simulation.supervised_learning").get<bool>();

        // IMPORTANT: output json is fixed in results/
        JSON_PATH = RESULT_JSON_PATH.string();

        // spline
        const auto solutionSplineCfg =
            iganet_elasticity::utils::config::load_solution_spline_config(j);
        NR_CTRL_PTS = solutionSplineCfg.nr_ctrl_pts;
        DEGREE_CFG = solutionSplineCfg.degree;

        const auto& singlePatchCfg = j.contains("single_patch_3D") ? j["single_patch_3D"] : j;

        // BCs
        FORCE_SIDES.clear();
        for (const auto& fsj : require(singlePatchCfg, "boundary_conditions.force_sides")) {
            FORCE_SIDES.emplace_back(fsj.at(0).get<int>(), fsj.at(1).get<double>(), fsj.at(2).get<double>(), fsj.at(3).get<double>());
        }

        DIRI_SIDES.clear();
        for (const auto& dsj : require(singlePatchCfg, "boundary_conditions.diri_sides")) {
            DIRI_SIDES.emplace_back(dsj.at(0).get<int>(), dsj.at(1).get<double>(), dsj.at(2).get<double>(), dsj.at(3).get<double>());
        }

        TFBC_SIDES = require(singlePatchCfg, "boundary_conditions.tfbc_sides").get<std::vector<int>>();

        // body force
        {
            const auto& bf = require(singlePatchCfg, "body_force");
            BODY_FORCE[0] = bf.at(0).get<double>();
            BODY_FORCE[1] = bf.at(1).get<double>();
            BODY_FORCE[2] = bf.at(2).get<double>();
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
        
    // calculation of lame parameters
    double lambda = (YOUNG_MODULUS * POISSON_RATIO) / 
                    ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
    double mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO));

    auto run = [&]<int DEGREE>() -> int {
        using real_t = double;
        using namespace iganet::literals;
        using optimizer_t = torch::optim::LBFGS;
        using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 3, DEGREE, DEGREE, DEGREE>>;
        using variable_t = iganet::S<iganet::UniformBSpline<real_t, 3, DEGREE, DEGREE, DEGREE>>;
        using linear_elasticity_t = linear_elasticity<optimizer_t, geometry_t, variable_t>;

        linear_elasticity_t net(//simulation parameters 
            lambda, mu, SUPERVISED_LEARNING, MAX_EPOCH, MIN_LOSS, 
            BODY_FORCE, TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, NR_CTRL_PTS, JSON_PATH, 
            // Number of neurons per layer 
            {25, 25}, 
            // Activation functions 
            //{{iganet::activation::sigmoid}, {iganet::activation::sigmoid}, {iganet::activation::none}},
            {{iganet::activation::sigmoid},{iganet::activation::sigmoid}, {iganet::activation::none}}, 
            // Number of B-spline coefficients of the geometry 
            std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS, NR_CTRL_PTS)), 
            // Number of B-spline coefficients of the variable 
            std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS, NR_CTRL_PTS)) );

        // impose body force; this should be a volumetric load, not a boundary load
       /* net.template output<0>().transform([=](const std::array<real_t, 3> xi) {
            return std::array<real_t, 3>{BODY_FORCE[0], BODY_FORCE[1], BODY_FORCE[2]};
        });*/

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
            double zDispl = std::get<3>(side);

            switch (sideNr) {
                case 1:
                    net.ref().boundary().template side<1>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<1>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    net.ref().boundary().template side<1>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{zDispl};
                        },
                        std::array<iganet::short_t, 1>{2}
                    );
                    break;
                case 2:
                    net.ref().boundary().template side<2>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<2>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    net.ref().boundary().template side<2>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{zDispl};
                        },
                        std::array<iganet::short_t, 1>{2}
                    );
                    break;
                case 3:
                    net.ref().boundary().template side<3>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<3>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    net.ref().boundary().template side<3>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{zDispl};
                        },
                        std::array<iganet::short_t, 1>{2}
                    );
                    break;
                case 4:
                    net.ref().boundary().template side<4>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<4>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    net.ref().boundary().template side<4>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{zDispl};
                        },
                        std::array<iganet::short_t, 1>{2}
                    );
                    break;
                case 5:
                    net.ref().boundary().template side<5>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<5>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    net.ref().boundary().template side<5>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{zDispl};
                        },
                        std::array<iganet::short_t, 1>{2}
                    );
                    break;
                case 6:
                    net.ref().boundary().template side<6>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{xDispl};
                        },
                        std::array<iganet::short_t, 1>{0} 
                    );
                    net.ref().boundary().template side<6>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{yDispl};
                        },
                        std::array<iganet::short_t, 1>{1}
                    );
                    net.ref().boundary().template side<6>().template transform<1>(
                        [=](const std::array<real_t, 2> &xi) {
                            return std::array<real_t, 1>{zDispl};
                        },
                        std::array<iganet::short_t, 1>{2}
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
        
        int64_t N3 = NR_CTRL_PTS * NR_CTRL_PTS * NR_CTRL_PTS; // [CHANGE 2D->3D]
        torch::Tensor netCtrlPts     = torch::zeros({N3, 3});
        torch::Tensor netDisplacements = torch::zeros({N3, 3});
 
        for (int i = 0; i < N3; ++i) {
            netCtrlPts[i][0] = geometryAsTensor[i].item<double>();
            netCtrlPts[i][1] = geometryAsTensor[i + N3].item<double>();
            netCtrlPts[i][2] = geometryAsTensor[i + 2 * N3].item<double>(); // [CHANGE 2D->3D]
                
            netDisplacements[i][0] = displacementAsTensor[i].item<double>();
            netDisplacements[i][1] = displacementAsTensor[i + N3].item<double>();
            netDisplacements[i][2] = displacementAsTensor[i + 2 * N3].item<double>(); // [CHANGE 2D->3D]
        }
        
        // deformed position of the control points
        torch::Tensor displacedNetCtrlPts = netCtrlPts + netDisplacements;

        // json objects for deformed positions
        nlohmann::json displacedNetCtrlPts_j = nlohmann::json::array();
        for (int i = 0; i < displacedNetCtrlPts.size(0); ++i) {
            displacedNetCtrlPts_j.push_back({
                displacedNetCtrlPts[i][0].item<double>(),
                displacedNetCtrlPts[i][1].item<double>(),
                displacedNetCtrlPts[i][2].item<double>()

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
