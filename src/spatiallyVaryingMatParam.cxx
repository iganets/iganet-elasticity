#include <iganet.h>
#include <iostream>
#include <fstream>

using namespace iganet::literals;       // @ include\iganet.hpp
using namespace gismo;

/// @brief Specialization of  IgANet class for linear elasticity in 2D
template <typename Optimizer, typename Inputs, typename Outputs>
class linear_elasticity : public iganet::IgANet2<Optimizer, Inputs, Outputs>,
                          public iganet::IgANetCustomizable2<Inputs, Outputs> {

    private:
        using Base = iganet::IgANet2<Optimizer, Inputs, Outputs>;

        Base::template collPts_t<0> collPts_;
        Base::template collPts_t<0> interiorCollPts_;
        int nrCollPts_;
        typename std::tuple_element_t<0, Outputs> ref_;

        // naming convention for this file:
        // G_   = geometry  = Base::template input<0>();
        // f_   = rhs       = Base::template input<1>();
        // mat_ = material  = Base::template input<2>();
        // u_   = solution  = Base::template output<0>();
        // Nbdr ... Neumann BounDaRy | Dbdr ... Dirichlet BounDaRy 

        using Customizable = iganet::IgANetCustomizable2<Inputs, Outputs>;
        
        // Declaring  types of knot/coeff indices. for various input/output quantities. 
        // --- G_ --- INPUT. geometry 
        Customizable::template input_interior_knot_indices_t<0> G_knot_indices_;
        Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_;

        Customizable::template input_interior_knot_indices_t<0> G_knot_indices_interior_;
        Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_interior_;

        Customizable::template input_interior_knot_indices_t<0> G_knot_indices_Nbdr_;           // indices of knots where Neuman Boundary Conditions are applied
        Customizable::template input_interior_coeff_indices_t<0> G_coeff_indices_Nbdr_;

        // --- f_ --- INPUT. RHS
        Customizable::template input_interior_knot_indices_t<1> var_knot_indices_;
        Customizable::template input_interior_coeff_indices_t<1> var_coeff_indices_;

        Customizable::template input_interior_knot_indices_t<1> var_knot_indices_interior_;
        Customizable::template input_interior_coeff_indices_t<1> var_coeff_indices_interior_;

        // --- mat_ --- INPUT. material. (not in use yet)
        Customizable::template input_interior_knot_indices_t<2> mat_knot_indices_;
        Customizable::template input_interior_coeff_indices_t<2> mat_coeff_indices_;

        Customizable::template input_interior_knot_indices_t<2> mat_knot_indices_interior_;
        Customizable::template input_interior_coeff_indices_t<2> mat_coeff_indices_interior_;

        Customizable::template input_interior_knot_indices_t<2> mat_knot_indices_Nbdr_;
        Customizable::template input_interior_coeff_indices_t<2> mat_coeff_indices_Nbdr_;

        // // --- u_ --- OUTPUT. solution
        // Customizable::template output_interior_knot_indices_t<0> u_knot_indices_;
        // Customizable::template output_interior_coeff_indices_t<0> u_coeff_indices_;

        // Customizable::template output_interior_knot_indices_t<0> u_knot_indices_interior_;
        // Customizable::template output_interior_coeff_indices_t<0> u_coeff_indices_interior_;

        // Customizable::template output_interior_knot_indices_t<0> u_knot_indices_Nbdr_;
        // Customizable::template output_interior_coeff_indices_t<0> u_coeff_indices_Nbdr_;

        Customizable::template output_interior_knot_indices_t<0> u_knot_indices_Nbdr_;
        Customizable::template output_interior_coeff_indices_t<0> u_coeff_indices_Nbdr_;

        // simulation parameters
        int MAX_EPOCH_;
        double MIN_LOSS_;
        int64_t NR_CTRL_PTS_;
        std::vector<int> TFBC_SIDES_;
        std::string JSON_PATH_;
        std::vector<std::tuple<int, double, double>> FORCE_SIDES_;
        std::vector<std::tuple<int, double, double>> DIRI_SIDES_;
        bool SUPERVISED_LEARNING_;

    public:
        // Constructor
        template <typename... Args>
        linear_elasticity(bool SUPERVISED_LEARNING, int MAX_EPOCH, double MIN_LOSS, const torch::optim::LBFGSOptions& solver_opts,                                          // nn options
                          std::vector<int> TFBC_SIDES, std::vector<std::tuple<int, double, double>> FORCE_SIDES, std::vector<std::tuple<int, double, double>> DIRI_SIDES,   // boundary conditions
                          int64_t NR_CTRL_PTS, std::string JSON_PATH, std::vector<int64_t> &&layers,                                                                        // simulation/nn options
                          std::vector<std::vector<std::any>> &&activations, Args &&...args)
            : Base(std::forward<std::vector<int64_t>>(layers),
                   std::forward<std::vector<std::vector<std::any>>>(activations),
                   std::forward<Args>(args)...),
                   SUPERVISED_LEARNING_(SUPERVISED_LEARNING), MAX_EPOCH_(MAX_EPOCH), MIN_LOSS_(MIN_LOSS),
                   TFBC_SIDES_(TFBC_SIDES), FORCE_SIDES_(FORCE_SIDES), DIRI_SIDES_(DIRI_SIDES),
                   NR_CTRL_PTS_(NR_CTRL_PTS), JSON_PATH_(std::move(JSON_PATH)), 
                   ref_(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)) {}

        
        auto const &collPts() const { return collPts_; }                  // Returns a constant reference to  collocation points
        auto const &interiorCollPts() const { return interiorCollPts_; }  // Returns a constant reference to  interior collocation points
        auto const &ref() const { return ref_; }                          // Returns a constant reference to  reference solution
        auto &ref() { return ref_; }                                      // Returns a non-constant reference to  reference solution
    
        // Writes data to JSON file
        void appendToJsonFile(const std::string& key, const nlohmann::json& data) {             
            nlohmann::json jsonData;        // create json object

            // read JSON data from file
            try {
                std::ifstream json_file_in(JSON_PATH_);
                if (json_file_in.is_open()) {
                    json_file_in >> jsonData;
                    json_file_in.close();
                } else {
                    std::cerr << "Warning: Could not open file for reading: " << JSON_PATH_ << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Error reading JSON file: " << JSON_PATH_ << ". Exception: " << e.what() << "\n";
            }

            // add new data to JSON object
            try {
                jsonData[key] = data;
            } catch (const std::exception& e) {
                std::cerr << "Error adding key to JSON object: " << e.what() << "\n";
                return;
            }

            // write JSON data to file
            try {
                std::ofstream json_file_out(JSON_PATH_);
                if (json_file_out.is_open()) {
                    json_file_out << jsonData.dump(1);
                    json_file_out.close();
                } else {
                    std::cerr << "Error: Could not open file for writing: " << JSON_PATH_ << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Error writing JSON file: " << JSON_PATH_ << ". Exception: " << e.what() << "\n";
            }
        }

        // helper function to load matlab displacements from JSON file
        torch::Tensor loadDisplacements() {
            auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);    // create options for  tensor
        
            // open  JSON file
            std::ifstream file(JSON_PATH_);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + JSON_PATH_);
            }
        
            // parse  JSON file
            nlohmann::json jsonData;
            file >> jsonData;
            file.close();
        
            // extract matlabDisplacements array
            auto matlabDisplacements_j = jsonData["matlabDisplacements"];
            int numCtrlPts = matlabDisplacements_j.size();
            torch::Tensor matlabDisplacements = torch::empty({numCtrlPts, 2}, options); // create tensor for displacements
            for (int i = 0; i < numCtrlPts; ++i) {                                      // fill  tensor with data from  JSON file
                matlabDisplacements[i][0] = matlabDisplacements_j[i][0].get<double>();
                matlabDisplacements[i][1] = matlabDisplacements_j[i][1].get<double>();
            }
        
            return matlabDisplacements;
        }

        // Initializes  epoch, special behaviour for initial epoch
        bool epoch(int64_t epoch) override {
            std::cout << "Epoch: " << epoch << std::endl;               // print epoch number
            if (epoch == 0) {
                Base::inputs(epoch);
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
                
                // --- G --- geom
                G_knot_indices_ =
                    Base::template input<0>().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);
                G_coeff_indices_ =
                    Base::template input<0>().template find_coeff_indices<iganet::functionspace::interior>(G_knot_indices_);

                G_knot_indices_interior_ = 
                    Base::template input<0>().template find_knot_indices<iganet::functionspace::interior>(interiorCollPts_.first);
                G_coeff_indices_interior_ =
                    Base::template input<0>().template find_coeff_indices<iganet::functionspace::interior>(G_knot_indices_interior_);
                
                    // --- f --- rhs
                var_knot_indices_ =
                    Base::template input<1>().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);
                var_coeff_indices_ =
                    Base::template input<1>().template find_coeff_indices<iganet::functionspace::interior>(var_knot_indices_);

                var_knot_indices_interior_ =
                    Base::template input<1>().template find_knot_indices<iganet::functionspace::interior>(interiorCollPts_.first);
                var_coeff_indices_interior_ =
                    Base::template input<1>().template find_coeff_indices<iganet::functionspace::interior>(var_knot_indices_interior_);

                // ---- mat ----- mat_
                // mat_knot_indices_ =
                //     Base::template input<2>().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);
                // mat_coeff_indices_ =
                //     Base::template input<2>().template find_coeff_indices<iganet::functionspace::interior>(mat_knot_indices_);
                // mat_knot_indices_interior_ =
                //     Base::template input<2>().template find_knot_indices<iganet::functionspace::interior>(interiorCollPts_.first);
                // mat_coeff_indices_interior_ =
                //     Base::template input<2>().template find_coeff_indices<iganet::functionspace::interior>(mat_knot_indices_interior_);
                                
                // // --- u_ --- sol vorher bitte deklarieren
                // u_knot_indices_ =
                //     Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);
                // u_coeff_indices_ =
                //     Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(u_knot_indices_);

                // u_knot_indices_interior_ =
                //     Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(interiorCollPts_.first);
                // u_coeff_indices_interior_ =
                //     Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(u_knot_indices_interior_);

                return true;
            } else
                return false;
        }

        /// @brief Computes  loss function
        torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

            // create u_ from  training's outputs
            Base::template output<0>().from_tensor(outputs);

            // pre-allocation of  loss values
            torch::Tensor totalLoss; 
            torch::Tensor elastLoss;
            std::optional<torch::Tensor> bcLoss;
            std::optional<torch::Tensor> tfbcLoss;
            std::optional<torch::Tensor> gsLoss;
            std::optional<torch::Tensor> forceLoss;

            // pre-allocation of  tensors for  traction boundary conditions
            std::optional<torch::Tensor> forceValues;
            std::optional<torch::Tensor> targetForce;
            std::optional<torch::Tensor> tractionFreeValues;
            std::optional<torch::Tensor> tractionZeros;

            // TRACTION BOUNDARY CONDITIONS
        
            // only calculate  traction(-free) boundary conditions if re are any
            if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty()){   
                
                // intersecCtr is used to determine an intersection of dirichlet/force and trac.free sides
                static std::vector<int> intersecCtr(0);
                // allocate tensors for  traction-free boundary conditions
                static std::array<torch::Tensor, 2ul> tractionCollPts;
                // collect sides of traction-free and force BCs
                std::vector<int> neumannSides;

                // collect sides of Dirichlet or force BCs
                std::vector<int> diriOrForceSides;
                for (const auto& tuple : DIRI_SIDES_) {
                    // extract only  int-values from DIRI_SIDES_
                    diriOrForceSides.push_back(std::get<0>(tuple));
                }       
                
                // add  two vectors of force- and traction-free-BCs
                neumannSides.reserve(TFBC_SIDES_.size() + FORCE_SIDES_.size());
                neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
                // add  force sides to  neumannSides and diriOrForceSides
                for (const auto& force : FORCE_SIDES_) {
                    // add  force sides to  neumannSides
                    neumannSides.push_back(std::get<0>(force));
                    // add  force sides to  diriOrForceSides
                    diriOrForceSides.push_back(std::get<0>(force));
                }

                // calculate  tractionCollocationPoints once in  beginning of  simulation
                if (epoch == 0 && intersecCtr.empty()) {
                    // allocate tensors for  traction-free boundary conditions
                    std::vector<torch::Tensor> tractionCollPtsX;
                    std::vector<torch::Tensor> tractionCollPtsY;

                    // evaluate  boundary points depending on traction-free sides
                    for (int side : neumannSides) {
                        switch (side) {
                            case 1:
                                // check if diriOrForceSides has only side 3 as side
                                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                                    != diriOrForceSides.end() &&
                                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                                    == diriOrForceSides.end()) {     

                                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}));
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
                                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}));
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
                                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 2}));  
                                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                                    // 2 collPts have to be removed
                                    intersecCtr.push_back(2);
                                }
                                else {
                                    tractionCollPtsX.push_back(torch::zeros(nrCollPts_));
                                    tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                                    // no collPt has to be removed
                                    intersecCtr.push_back(0);
                                }
                                break;
                            case 2:
                                // check if diriOrForceSides has only side 3 as side
                                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                                    != diriOrForceSides.end() &&
                                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                                    == diriOrForceSides.end()) {    

                                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}));
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
                                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}));
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
                                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 2}));
                                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                                    // 2 collPts have to be removed
                                    intersecCtr.push_back(2);
                                }
                                else {
                                    tractionCollPtsX.push_back(torch::ones(nrCollPts_));
                                    tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                                    // no collPt has to be removed
                                    intersecCtr.push_back(0);
                                }
                                break;
                            case 3:
                                // check if diriOrForceSides has only side 1 as side
                                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                                    != diriOrForceSides.end() &&
                                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                                    == diriOrForceSides.end()) {   

                                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}));
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
                                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}));
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
                                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 2}));
                                    // 2 collPts have to be removed
                                    intersecCtr.push_back(2);
                                }
                                else {
                                    tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                                    tractionCollPtsY.push_back(torch::zeros(nrCollPts_));
                                    // no collPt has to be removed
                                    intersecCtr.push_back(0);
                                }
                                break;
                            case 4:
                                // check if diriOrForceSides has only side 1 as side
                                if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                                    != diriOrForceSides.end() &&
                                    std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                                    == diriOrForceSides.end()) {   

                                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}));
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
                                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}));
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
                                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 2}));
                                    // 2 collPts have to be removed
                                    intersecCtr.push_back(2);
                                }
                                else {
                                    tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                                    tractionCollPtsY.push_back(torch::ones(nrCollPts_));
                                    // no collPt has to be removed
                                    intersecCtr.push_back(0);
                                }
                                break;         
                            default:
                                throw std::invalid_argument("Side for traction BC has to be 1, 2, 3 or 4.");
                                break;
                        }
                    }
                    
                    // merge  tensors to get a (nrTractionCollPts, 2) tensor
                    if (!tractionCollPtsX.empty() && !tractionCollPtsY.empty()) {
                        tractionCollPts = {
                            torch::cat(tractionCollPtsX, 0), 
                            torch::cat(tractionCollPtsY, 0)
                        };
                    }

                    u_knot_indices_Nbdr_ =     // vormals _boundary (!)
                        Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(tractionCollPts);        // boundary vs interior.
                    u_coeff_indices_Nbdr_ =
                        Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(u_knot_indices_Nbdr_);
                    G_knot_indices_Nbdr_ =
                        Base::template input<0>().template find_knot_indices<iganet::functionspace::interior>(tractionCollPts);
                    G_coeff_indices_Nbdr_ =
                        Base::template input<0>().template find_coeff_indices<iganet::functionspace::interior>(G_knot_indices_Nbdr_);

                    mat_knot_indices_Nbdr_ =
                        Base::template input<2>().template find_knot_indices<iganet::functionspace::interior>(tractionCollPts);
                    mat_coeff_indices_Nbdr_ =
                        Base::template input<2>().template find_coeff_indices<iganet::functionspace::interior>(mat_knot_indices_Nbdr_);

                }  

                // calculate  Jacobian of  affected (Neumann) boundary points
                auto Jacobian_Nbdr = Base::template output<0>().ijac(Base::template input<0>(), tractionCollPts,         //G, xi, ?knot, ?coef, Gknot, Gcoef. ijac=J(?)*J(G)^T
                    u_knot_indices_Nbdr_, u_coeff_indices_Nbdr_,     //xi knot coef
                    G_knot_indices_Nbdr_, G_coeff_indices_Nbdr_);
                auto ux_x = *Jacobian_Nbdr[0];      // all sizes [12] so  tf boundaries
                auto ux_y = *Jacobian_Nbdr[1];
                auto uy_x = *Jacobian_Nbdr[2];
                auto uy_y = *Jacobian_Nbdr[3];

                // allocate tensors for  traction-free boundary conditions (tfbc)
                torch::Tensor tractionValuesX = torch::zeros({tractionCollPts[0].size(0)});
                torch::Tensor tractionValuesY = torch::zeros({tractionCollPts[0].size(0)});
                // calculate  traction values at  boundary points
                int pointCtr = 0;
                int sideCtr = 0; 
                auto mat = Base::template input<2>().eval(tractionCollPts);

                for (int side : neumannSides) {
                    int n_vals = nrCollPts_ - intersecCtr[sideCtr];

                    for (int i = 0; i < n_vals; ++i) {
                        int idx = pointCtr + i;     //11 oder 12 it. 0 bis 12?

                        double x_temp = tractionCollPts[0][i].item<double>();
                        double y_temp = tractionCollPts[1][i].item<double>();
                        double matLambda_temp = mat(0)[i].template item<double>();
                        double matMu_temp     = mat(1)[i].template item<double>();

                        switch (side) {
                            case 1:
                                tractionValuesX[idx] =  - matLambda_temp * (ux_x[idx] + uy_y[idx]) - 2 * matMu_temp * ux_x[idx];
                                tractionValuesY[idx] =  - matMu_temp * (uy_x[idx] + ux_y[idx]);
                                break;
                            case 2:
                                tractionValuesX[idx] = matLambda_temp * (ux_x[idx] + uy_y[idx]) + 2 * matMu_temp * ux_x[idx];
                                tractionValuesY[idx] = matMu_temp * (uy_x[idx] + ux_y[idx]);
                                break;
                            case 3: 
                                tractionValuesX[idx] =  - matMu_temp * (uy_x[idx] + ux_y[idx]);
                                tractionValuesY[idx] =  - matLambda_temp * (ux_x[idx] + uy_y[idx]) - 2 * matMu_temp * uy_y[idx];
                                break;
                            case 4:
                                tractionValuesX[idx] = matMu_temp * (uy_x[idx] + ux_y[idx]);
                                tractionValuesY[idx] = matLambda_temp * (ux_x[idx] + uy_y[idx]) + 2 * matMu_temp * uy_y[idx];
                                break;
                            default:
                                std::cerr << "Error: invalid side = " << side << std::endl;
                                break;
                        }
                    }

                    pointCtr += n_vals;
                    sideCtr++;
                }

                // merge  traction tensors of x- and y-directions
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
                    // fill in  known force values
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
                    // set  traction-free values
                    tractionFreeValues.emplace(tractionValues);
                    // set  target values to zero
                    tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
                }

            }

            // *** LINEAR ELASTICITY EQUATION

            // calculation of  second derivatives (Hessian Matrix) of  displacements (u)
            auto Hessian =  Base::template output<0>().ihess(Base::template input<0>(), interiorCollPts_.first, 
                            var_knot_indices_interior_, var_coeff_indices_interior_,
                            G_knot_indices_interior_, G_coeff_indices_interior_);


            auto mat = Base::template input<2>().eval(interiorCollPts_.first);      // matx = mat(0)

            // partial derivatives of  displacements (u) 
            auto& ux_xx = Hessian(0,0,0);
            auto& ux_xy = Hessian(0,1,0);
            auto& ux_yx = Hessian(1,0,0);
            auto& ux_yy = Hessian(1,1,0);

            auto& uy_xx = Hessian(0,0,1);
            auto& uy_xy = Hessian(0,1,1);
            auto& uy_yx = Hessian(1,0,1);
            auto& uy_yy = Hessian(1,1,1);

            // pre-allocation of  results
            torch::Tensor divStressX = torch::zeros({Hessian(0,0,0).size(0)});
            torch::Tensor divStressY = torch::zeros({Hessian(0,0,1).size(0)});

            // calculation of  divergence of  stress tensor, this is what we're trying to minimize
            for (int i = 0; i < Hessian(0,0,0).size(0); ++i) {      // 36 it über interior 
                at::Tensor tx = std::get<0>(interiorCollPts_.first)[i];
                double x_temp = tx.item<double>();  
                at::Tensor ty = std::get<1>(interiorCollPts_.first)[i];
                double y_temp = ty.item<double>();  
                double matLambda_temp = mat(0)[i].template item<double>();
                double matMu_temp     = mat(1)[i].template item<double>();

                // x-direction
                divStressX[i] = (matLambda_temp + 2 * matMu_temp) * ux_xx[i] + 
                                matMu_temp* ux_yy[i] + (matLambda_temp + matMu_temp) * uy_xy[i];

                // y-direction
                divStressY[i] = matMu_temp * uy_xx[i] + (matLambda_temp + 2 * matMu_temp) * uy_yy[i] + 
                                (matLambda_temp + matMu_temp) * ux_xy[i];
                
            }
            
            // create a tensor of  divergence of  stress tensor
            torch::Tensor divStress = torch::stack({divStressX, divStressY}, /*dim=*/1);

            // BODY FORCE

            // evaluate  reference body force f at all interior collocation points
            auto f = Base::template input<1>().eval(interiorCollPts_.first);        // shape 2x1x36. [0,0] is a data array of 36. same for [1,0].

            torch::Tensor bodyForce = torch::stack({*f[0], *f[1]}, /*dim=*/1).to(torch::kFloat32);

            // *** Loss values
            // UNSUPERVISED LEARNING (default)
            if (SUPERVISED_LEARNING_ == false) {

                // create command line output variable for all  different losses
                std::ostringstream singleLossOutput;

                // calculation of  loss function for double-sided constraint solid
                // div(sigma) + f = 0 --> div(sigma) = -f
                elastLoss = torch::mse_loss(divStress, bodyForce);
                
                // add  elasticity loss to  total loss
                totalLoss = elastLoss;

                // add  elasticity loss to  cmd-output variable
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
                    // add a BC weight for penalization of  training
                    int bcWeight = 1e7;
                    // initialize bcLoss variable
                    bcLoss = torch::tensor(0.0);

                    // evaluation of  displacements at  (Dirichlet) boundary points. Nbdr ... Neumann BounDaRy | Dbdr ... Dirichlet BounDaRy 
                    auto u_Dbdr = Base::template output<0>().template eval<iganet::functionspace::boundary>(collPts_.second);
                    // evaluation of  displacements at  reference boundary points.
                    auto ref_Dbdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

                    // loop through all dirichlet sides
                    for (const auto& side : DIRI_SIDES_) {
                        int sideNr = std::get<0>(side);
                        
                        switch (sideNr) {
                            case 1: 
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<0>(u_Dbdr)[0], *std::get<0>(ref_Dbdr)[0]) + 
                                    torch::mse_loss(*std::get<0>(u_Dbdr)[1], *std::get<0>(ref_Dbdr)[1]));
                                break;
                            case 2:
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<1>(u_Dbdr)[0], *std::get<1>(ref_Dbdr)[0]) + 
                                    torch::mse_loss(*std::get<1>(u_Dbdr)[1], *std::get<1>(ref_Dbdr)[1]));
                                break;
                            case 3:
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<2>(u_Dbdr)[0], *std::get<2>(ref_Dbdr)[0]) + 
                                    torch::mse_loss(*std::get<2>(u_Dbdr)[1], *std::get<2>(ref_Dbdr)[1]));
                                break;
                            case 4:
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<3>(u_Dbdr)[0], *std::get<3>(ref_Dbdr)[0]) + 
                                    torch::mse_loss(*std::get<3>(u_Dbdr)[1], *std::get<3>(ref_Dbdr)[1]));
                                break;
                            default:
                                std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                        }
                    }
                    totalLoss += *bcLoss;
                    singleLossOutput << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight 
                                    << " * 1e" << static_cast<int>(std::log10(bcWeight));
                }

                // print  loss values
                std::cout << std::setw(11) << 
                    totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
            }
            
            // SUPERVISED LEARNING, nachher bitte hier einfügen.
            else if (SUPERVISED_LEARNING_ == true) {

                // create command line output variable for all  different losses
                std::ostringstream singleLossOutput;
            
                // preprocess  outputs for comparison with  matlab solution
                torch::Tensor modifiedOutputs = outputs * 1.0;
            
                // create netDisplacements_ from slices of modifiedOutputs
                torch::Tensor netDisplacements_ = torch::stack({
                    modifiedOutputs.slice(0, 0, outputs.size(0) / 2),
                    modifiedOutputs.slice(0, outputs.size(0) / 2, outputs.size(0)),
                }, 1);

                // load  displacements from  matlab solution
                torch::Tensor matlabDisplacements_ = loadDisplacements();

                // supervised loss: MSE gegen matlab-Kontrollpunkte
                gsLoss = 1e9 * torch::mse_loss(netDisplacements_, matlabDisplacements_);

                // calculation of  loss function for double-sided constraint solid
                // div(sigma) + f = 0 --> div(sigma) = -f
                elastLoss = torch::mse_loss(divStress, bodyForce);

                // add  elasticity loss and supervised loss to  total loss
                totalLoss = *gsLoss + elastLoss;

                // add  elasticity and supervised losses to  cmd-output variable
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
                    // add a BC weight for penalization of  training
                    int bcWeight = 1e0;
                    // initialize bcLoss variable
                    bcLoss = torch::tensor(0.0);

                    // evaluation of  displacements at  boundary points
                    auto u_Dbdr = Base::template output<0>().template eval<iganet::functionspace::boundary>(collPts_.second);
                    // evaluation of  displacements at  reference boundary points
                    auto ref_Dbdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

                    // loop through all dirichlet sides
                    for (const auto& side : DIRI_SIDES_) {
                        int sideNr = std::get<0>(side);

                        switch (sideNr) {
                            case 1:
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<0>(u_Dbdr)[0], *std::get<0>(ref_Dbdr)[0]) +      // seite 1 x (links)  0
                                    torch::mse_loss(*std::get<0>(u_Dbdr)[1], *std::get<0>(ref_Dbdr)[1]));       // seite 1 y (rechts) 0
                                break;
                            case 2:
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<1>(u_Dbdr)[0], *std::get<1>(ref_Dbdr)[0]) +      // seite 2 x (links)  0.5
                                    torch::mse_loss(*std::get<1>(u_Dbdr)[1], *std::get<1>(ref_Dbdr)[1]));       // seite 2 y (rechts) 0
                                break;
                            case 3:
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<2>(u_Dbdr)[0], *std::get<2>(ref_Dbdr)[0]) + 
                                    torch::mse_loss(*std::get<2>(u_Dbdr)[1], *std::get<2>(ref_Dbdr)[1]));
                                break;
                            case 4:
                                *bcLoss += bcWeight * 
                                    (torch::mse_loss(*std::get<3>(u_Dbdr)[0], *std::get<3>(ref_Dbdr)[0]) + 
                                    torch::mse_loss(*std::get<3>(u_Dbdr)[1], *std::get<3>(ref_Dbdr)[1]));
                                break;
                            default:
                                std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                        }
                    }
                    totalLoss += *bcLoss;
                    singleLossOutput << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight 
                                    << " * 1e" << static_cast<int>(std::log10(bcWeight));
                }

                // print  loss values
                std::cout << std::setw(11) << 
                    totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
            }

            else {
                throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_ (should be true/false)");
            }

            // *** POSTPROCESSING PREPARATION - WRITING DATA TO JSON FILE

            // only calculate this at  end of  simulation
            if ((epoch == MAX_EPOCH_ - 1) || (totalLoss.item<double>() <= MIN_LOSS_)) {
                
                // STRESS CALCULATION

                // calculate  Jacobian of  displacements (u) at  collocation points
                auto Jacobian = Base::template output<0>().ijac(Base::template input<0>(), collPts_.first,
                                var_knot_indices_, var_coeff_indices_,
                                G_knot_indices_, G_coeff_indices_);
                
                auto ux_x = *Jacobian[0];
                auto ux_y = *Jacobian[1];
                auto uy_x = *Jacobian[2];
                auto uy_y = *Jacobian[3];

                // allocate  stress tensor
                torch::Tensor sigma_xx = torch::zeros({Jacobian[0]->size(0)});
                torch::Tensor sigma_xy = torch::zeros({Jacobian[0]->size(0)});
                torch::Tensor sigma_yy = torch::zeros({Jacobian[0]->size(0)}); 
                torch::Tensor sigma_vm = torch::zeros({Jacobian[0]->size(0)});   

                torch::Tensor epsilon_xx = torch::zeros({Jacobian[0]->size(0)});
                torch::Tensor epsilon_yy = torch::zeros({Jacobian[0]->size(0)});
                torch::Tensor poisson_re = torch::zeros({Jacobian[0]->size(0)});

                // create json object for  stresses
                nlohmann::json netVmStresses_j = nlohmann::json::array();
                nlohmann::json netXStresses_j = nlohmann::json::array();
                nlohmann::json netYStresses_j = nlohmann::json::array();
                nlohmann::json netPoisson_j = nlohmann::json::array();

                auto mat = Base::template input<2>().eval(collPts_.first);

                // calculate  stress tensor
                for (int i = 0; i < Jacobian[0]->size(0); ++i) {        // 64 it über gesamte domain

                    at::Tensor tx = std::get<0>(collPts_.first)[i];
                    double x_temp = tx.item<double>();  
                    at::Tensor ty = std::get<1>(collPts_.first)[i];
                    double y_temp = ty.item<double>();  
                    double matLambda_temp = mat(0)[i].template item<double>();
                    double matMu_temp     = mat(1)[i].template item<double>();

                    // calculate  stress values for all collocation points
                    sigma_xx[i] = matLambda_temp * (ux_x[i] + uy_y[i]) + 2 * matMu_temp * ux_x[i];
                    sigma_xy[i] = matMu_temp * (uy_x[i] + ux_y[i]);
                    sigma_yy[i] = matLambda_temp * (ux_x[i] + uy_y[i]) + 2 * matMu_temp * uy_y[i];
                    
                    // calculate von mises stress at  collocation points
                    sigma_vm[i] = sqrt(sigma_xx[i] * sigma_xx[i] + sigma_yy[i] * sigma_yy[i] 
                                    - sigma_xx[i] * sigma_yy[i] + sigma_xy[i] * sigma_xy[i] * 3);
                    
                    // calculate  strains at  collocation points
                    epsilon_xx[i] = (matLambda_temp + matMu_temp) / (matMu_temp * (3 * matLambda_temp + 2 * matMu_temp)) * 
                        (sigma_xx[i] - matLambda_temp / (2 * (matLambda_temp + matMu_temp)) * sigma_yy[i]);
                    epsilon_yy[i] = (matLambda_temp + matMu_temp) / (matMu_temp * (3 * matLambda_temp + 2 * matMu_temp)) * 
                        (sigma_yy[i] - matLambda_temp / (2 * (matLambda_temp + matMu_temp)) * sigma_xx[i]);

                    // only valid for load in x-direction
                    poisson_re[i] = - epsilon_yy[i] / epsilon_xx[i];
                    
                    // add  stresses to  json objects
                    netVmStresses_j.push_back({sigma_vm[i].item<double>()});
                    netXStresses_j.push_back({sigma_xx[i].item<double>()});
                    netYStresses_j.push_back({sigma_yy[i].item<double>()});
                    // add  poisson ratio to  json object
                    netPoisson_j.push_back({poisson_re[i].item<double>()});
                }

                // write  stresses and poisson ratios to  json file
                appendToJsonFile("net_VmStresses", netVmStresses_j);
                appendToJsonFile("net_XStresses", netXStresses_j);
                appendToJsonFile("net_YStresses", netYStresses_j);
                appendToJsonFile("net_Poisson", netPoisson_j);

                // CALCULATE THE NEW POSITION OF THE COLLPTS

                // create a tensor of  collocation points
                torch::Tensor collPtsFirstAsTensor = torch::stack(
                    {std::get<0>(collPts_.first), std::get<1>(collPts_.first)}, 1);
                auto displacementOfCollPts = Base::template output<0>().eval(collPts_.first);
                torch::Tensor displacementAsTensor = torch::stack(
                    {*(displacementOfCollPts[0]), *(displacementOfCollPts[1]) }, 1);

                // create json objects for  collocation points' reference and displaced position
                nlohmann::json collPtsFirst_j = nlohmann::json::array();
                nlohmann::json collPtsFirstDispl_j = nlohmann::json::array();
                for (int i = 0; i < collPtsFirstAsTensor.size(0); ++i) {
                    // reference position of  collocation points
                    collPtsFirst_j.push_back({collPtsFirstAsTensor[i][0].item<double>(), 
                                            collPtsFirstAsTensor[i][1].item<double>()});
                    // new position of  collocation points
                    collPtsFirstDispl_j.push_back({collPtsFirstAsTensor[i][0].item<double>() + 
                                                displacementAsTensor[i][0].item<double>(), 
                                                collPtsFirstAsTensor[i][1].item<double>() + 
                                                displacementAsTensor[i][1].item<double>()});
                }
                // write  collocation points' original position to  json file
                appendToJsonFile("net_collPtsFirstAsTensor", collPtsFirst_j);
                // write  collocation points' new position to  json file
                appendToJsonFile("net_collPtsFirstAfterDisplacementAsTensor", collPtsFirstDispl_j);

                // WRITING DIVERGENCE OF THE STRESS TENSOR TO JSON FILE

                nlohmann::json netDivergenceX_j = nlohmann::json::array();
                nlohmann::json netDivergenceY_j = nlohmann::json::array();

                for (int i = 0; i < divStressX.size(0); ++i) {
                    netDivergenceX_j.push_back({divStressX[i].item<double>()});
                    netDivergenceY_j.push_back({divStressY[i].item<double>()});
                }

                // write  divergence of  stress tensor to  json file
                appendToJsonFile("net_DivergenceX", netDivergenceX_j);
                appendToJsonFile("net_DivergenceY", netDivergenceY_j);
            }
            return totalLoss;
        }
};



int main() {
    iganet::init();
    iganet::verbose(std::cout);

    // ------- USER INPUTS ------- //
    // simulation parameters
    int MAX_EPOCH = 100;
    double MIN_LOSS = 1e-12;
    bool SUPERVISED_LEARNING = false;
    std::string JSON_PATH = "/home/isabellaunix/DevelDA/singerDA/ConfigResult/result.json";       

    // spline parameters
    int64_t NR_CTRL_PTS = 8;  // in each direction 
    constexpr int DEGREE = 4; // for geometry and variable og.: constexpr

    // boundary conditions
    std::vector<std::tuple<int, double, double>> FORCE_SIDES = {
        //   {2, 50.0,  0.0},   // {side, x-traction, y-traction}
        };
    std::vector<std::tuple<int, double, double>> DIRI_SIDES = {
        {1, 0.0,  0.0},       // {side, x-displ, y-displ}
        {2, 0.05,  0.0},
        };
    std::vector<int> TFBC_SIDES = {3,4}; // {sides}

    // body force (constant over  whole domain)
    std::pair<double, double> BODY_FORCE = {0.0, 0.0}; // {fx, fy}
    
    auto solver_options = torch::optim::LBFGSOptions(1.0).
                                        max_iter(50).
                                        max_eval(75).
                                        history_size(200).
                                        tolerance_grad(1e-12).
                                        tolerance_change(1e-12).
                                        line_search_fn("strong_wolfe");

    // OPTIONAL .json input for easy change of params. no need to rebuild :) auszukommentieren, wenn nicht gebraucht.
    bool USERINPUT = true;
    std::ifstream file("/home/isabellaunix/DevelDA/singerDA/ConfigResult/config.json");
    if (!file) {
        std::cerr << "Could not open config.json\n";
        return 1;
    }
    nlohmann::json j;
    file >> j;
    USERINPUT = j["simulation"]["USERINPUT"];
    if (USERINPUT==false) {

        // simulation parameters
        MAX_EPOCH = j["simulation"]["max_epoch"];
        MIN_LOSS = j["simulation"]["min_loss"];
        SUPERVISED_LEARNING = j["simulation"]["supervised_learning"];
        std::string JSON_PATH = j["simulation"]["json_path"];

        // spline parameters
        NR_CTRL_PTS = j["spline"]["nr_ctrl_pts"];
        // DEGREE = 4; // could be set dynamically too

        // boundary conditions
        FORCE_SIDES.clear();    //inhomo neumann
        for (const auto& fs : j["boundary_conditions"]["force_sides"]) {
            FORCE_SIDES.emplace_back(fs[0], fs[1], fs[2]);
        }

        DIRI_SIDES.clear();     //inhomo homo dirichlet
        for (const auto& ds : j["boundary_conditions"]["diri_sides"]) {
            DIRI_SIDES.emplace_back(ds[0], ds[1], ds[2]);
        }

        TFBC_SIDES = j["boundary_conditions"]["tfbc_sides"].get<std::vector<int>>();    //homo neumann

        // body force
        BODY_FORCE.first = j["body_force"][0];      //vorce
        BODY_FORCE.second = j["body_force"][1];

        // just to verify
        std::cout << "TFBC sides: ";
        for (auto side : TFBC_SIDES) std::cout << side << " ";
        std::cout << "\n";
    } 
    // OPTIONAL END
    // --------------------------- //

    using real_t = double;
    using namespace iganet::literals;
    using optimizer_t = torch::optim::LBFGS;
    
    using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, 3, 3>>;   
    using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, 4, 4>>;
    using material_t = iganet::S<iganet::UniformBSpline<real_t, 2, 4, 4>>;

    using inputs_t = std::tuple<geometry_t, variable_t, material_t>;     
    using outputs_t = std::tuple<variable_t>;     
    using linear_elasticity_t = linear_elasticity<optimizer_t, inputs_t, outputs_t>;
        
    linear_elasticity_t net( // simulation parameters
        SUPERVISED_LEARNING, MAX_EPOCH, MIN_LOSS, solver_options, 
        TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, NR_CTRL_PTS, JSON_PATH,
        // Number of neurons per layer
        {25, 25},
        // Activation functions
        {{iganet::activation::sigmoid},
            {iganet::activation::sigmoid},
            {iganet::activation::none}},
        // Number of B-spline coefficients of  geometry
        std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)),
        // Number of B-spline coefficients of  variable
        std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS))
    );

    // xml in net.template input<1>().eval(collPts.first)
    pugi::xml_document xml;
    xml.load_file("/home/isabellaunix/DevelDA/singerDA/ConfigResult/mat.xml");
    net.template input<2>().from_xml(xml);

    // imposing body force f
    net.template input<1>().transform([=](const std::array<real_t, 2> xi) {
        return std::array<real_t, 2>{BODY_FORCE.first, BODY_FORCE.second};
    });

    // get  coefficients of  control points
    torch::Tensor ctrlPtsCoeffs = net.template output<0>().as_tensor().slice(0, 0, NR_CTRL_PTS);
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
                net.ref().boundary().side<1>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net.ref().boundary().side<1>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 2:
                net.ref().boundary().side<2>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net.ref().boundary().side<2>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 3:
                net.ref().boundary().side<3>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net.ref().boundary().side<3>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 4:
                net.ref().boundary().side<4>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net.ref().boundary().side<4>().transform<1>(
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

    // Set tolerance for  loss functions
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

    // *** POSTPROCESSING
    //  get  geometry and displacement as tensors
    torch::Tensor geometryAsTensor = net.template input<0>().as_tensor();
    torch::Tensor displacementAsTensor = net.template output<0>().as_tensor();
    
    // creating collection matrix for all  control points / displacements (iganet)
    gsMatrix<real_t> netCtrlPts(NR_CTRL_PTS * NR_CTRL_PTS, 2);
    gsMatrix<real_t> netDisplacements(NR_CTRL_PTS * NR_CTRL_PTS, 2);

    // filling  collection matrices with  values from  tensors
    for (int i = 0; i < NR_CTRL_PTS * NR_CTRL_PTS; ++i) {
        double x = geometryAsTensor[i].item<double>();          
        double y = geometryAsTensor[i + NR_CTRL_PTS * NR_CTRL_PTS].item<double>();
        netCtrlPts(i, 0) = x;
        netCtrlPts(i, 1) = y;
            
        double ux = displacementAsTensor[i].item<double>();
        double uy = displacementAsTensor[i + NR_CTRL_PTS * NR_CTRL_PTS].item<double>();
        netDisplacements(i, 0) = ux;
        netDisplacements(i, 1) = uy;
    }

    // deformed position of CtrlPts
    gsMatrix<double> displacedNetCtrlPts = netCtrlPts + netDisplacements;
    nlohmann::json displacedNetCtrlPts_j = nlohmann::json::array();  // json objects for deformed positions of CtrlPts

    // write net data from matrices to json objects
    for (int i = 0; i < displacedNetCtrlPts.rows(); ++i) {
        displacedNetCtrlPts_j.push_back({displacedNetCtrlPts(i, 0), displacedNetCtrlPts(i, 1)});        // new control points IgANet
    }

    // write data to  json file
    net.appendToJsonFile("net_CtrlPts", displacedNetCtrlPts_j);
    net.appendToJsonFile("net_Degree", DEGREE);
    
    iganet::finalize();
    return 0;
}
// int main() { return 0; }