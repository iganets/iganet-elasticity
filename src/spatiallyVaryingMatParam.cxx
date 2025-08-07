#include <iganet.h>
#include <iostream>
#include <fstream>

using namespace iganet::literals;
using namespace gismo;

template <typename Optimizer, typename GeometryMap, typename Variable>
class spatiallyVaryingMatParam: public iganet::IgANet<Optimizer, GeometryMap, Variable>,
                                public iganet::IgANetCustomizable<GeometryMap, Variable> {
  private:
    using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

    typename Base::variable_collPts_type collPts_;
    typename Base::variable_collPts_type interiorCollPts_;
    
    int nrCollPts_;
    Variable ref_;

    using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

    typename Customizable::variable_interior_knot_indices_type var_knot_indices_;
    typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_;

    typename Customizable::variable_interior_knot_indices_type var_knot_indices_interior_;
    typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_interior_;

    typename Customizable::variable_interior_knot_indices_type var_knot_indices_boundary_;
    typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_boundary_;

    typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;
    typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_;

    typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_interior_;
    typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_interior_;

    typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_boundary_;
    typename Customizable::geometryMap_interior_c*11oeff_indices_type G_coeff_indices_boundary_;

    // placeholder material properties               
    
    // placeholder simulation parameter

    // placeholder

  public:
    template <typename... Args>
    PLACEHOLDER(
        // placholder - args
        std::vector<int64_t> &&layers,
        std::vector<std::vector<std::any>> &&activations,
        Args &&...args)
        :Base(std::forward<std::vector<int64_t>>(layers),
            std::forward<std::vector<std::vector<std::any>>>(activations),
            std::forward<Args>(args)...),
        // placeholder VALUE(value),
        ref_(iganet::utils::to_array(placeholder, placeholder)) {}      // placeholder {} == constructor body if needed
    
    auto const &collPts() const { return collPts_; }                        // Returns a constant reference to the collocation points
    auto const &interiorCollPts() const { return interiorCollPts_; }        // Returns a constant reference to the interior collocation points
    auto const &ref() const { return ref_; }                                // Returns a constant reference to the reference solution
    auto &ref() { return ref_; }                                            // Returns a non-constant reference to the reference solution

    void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
        nlohmann::json jsonData;                                            // create json object
        try {                                                               // try to read the JSON data from the file
            std::ifstream json_file_in(JSON_PATH);
            if (json_file_in.is_open()) {
                json_file_in >> jsonData;
                json_file_in.close();
            } else {
                std::cerr << "Warning: Could not open file for reading: " << JSON_PATH << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading JSON file: " << JSON_PATH << ". Exception: " << e.what() << "\n";
        }
        try {                                                               // add new data to the JSON object
            jsonData[key] = data;
        } catch (const std::exception& e) {
            std::cerr << "Error adding key to JSON object: " << e.what() << "\n";
            return;
        }
        try {                                                               // write the JSON data to the file
            std::ofstream json_file_out(JSON_PATH);
            if (json_file_out.is_open()) {
                json_file_out << jsonData.dump(1);
                json_file_out.close();
            } else {
                std::cerr << "Error: Could not open file for writing: " << JSON_PATH << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error writing JSON file: " << JSON_PATH << ". Exception: " << e.what() << "\n";
        }
    }   // END void appendToJsonFile

    // helper function to calculate the Greville abscissae
    static std::vector<double> computeGrevilleAbscissae(const gsKnotVector<double>& knotVector, int degree, int numCtrlPts) {
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

    void write_result() {                // write geometry and solution spline data to file
        appendToJsonFile("G", Base::G_.to_json());
        appendToJsonFile("u", Base::u_.to_json());
    }

    // Initializes epoch
    bool epoch(int64_t epoch) override {
        std::cout << "Epoch: " << epoch << std::endl;
        if (epoch == 0) {
            // placeholder
            return true;
        } else {
            return false;
        }
    }

    // Computes loss function
    torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {      
        
        // create u_ from the training's outputs
        Base::u_.from_tensor(outputs);

        // pre-allocation of the loss values
        torch::Tensor totalLoss;                            // = \sum \limits _i Loss_i
        torch::Tensor EqLoss;                               // og elastLoss
        std::optional<torch::Tensor> bcDirichletLoss;       // og bcLoss
        std::optional<torch::Tensor> bcNeumannLoss;         // og tfbcLoss and forceLoss
        std::optional<torch::Tensor> archLoss;              // eg for supervised learning og gsLoss (!not for now)

        // placeholder Eq Loss
        totalLoss = EqLoss;

        // placeholder bc Dirichlet Loss
        totalLoss += bcDirichletLoss;

        // placeholder bc Neumann Loss
        totalLoss += bcNeumannLoss;

        // placeholder

        std::cout << std::setw(11) << totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
        return totalLoss; 
    }

};   // END class spatiallyVaryingMatParam

int main(int argc, char* argv[]) {
    iganet::init();
    iganet::verbose(std::cout);

    // placeholder user input

    // placeholder calculation of lame parameters

    using real_t = double;
    using namespace iganet::literals;
    using optimizer_t = torch::optim::LBFGS;
    using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
    using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
    using spatiallyVaryingMatParam_t = spatiallyVaryingMatParam<optimizer_t, geometry_t, variable_t>;

    spatiallyVaryingMatParam_t
        net(      // simulation parameters
          PLACEHOLDER
          // Number of neurons per layer
          {PLACEHOLDER, PLACEHOLDER},
          // Activation functions
          {{iganet::activation::PLACEHOLDER},
           {iganet::activation::PLACEHOLDER},
           {iganet::activation::PLACEHOLDER}},
          // Number of B-spline coefficients of the geometry
          std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS))
        );

    // imposing body force
    net.f().transform([=](const std::array<real_t, 2> xi) {
        return std::array<real_t, 2>{PLACEHOLDER, PLACEHOLDER};         //BODY_FORCE.first, BODY_FORCE.second or 0, 0
    });

    net.options().max_epoch(MAX_EPOCH);                         // Set maximum number of epochs
    net.options().min_loss(MIN_LOSS);                           // Set tolerance for the loss functions
    auto t1 = std::chrono::high_resolution_clock::now();        // Start time measurement
    net.train();                                                // Train network
    auto t2 = std::chrono::high_resolution_clock::now();         // Stop time measurement
    iganet::Log(iganet::log::info)
        << "Training took " << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() << " seconds\n";      // maybe as auto var runtime

    net.write_result();

    // placeholder

    iganet::finalize();
    return 0
}