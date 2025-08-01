#include <iganet.h>
#include <iostream>
#include <fstream>

using namespace iganet::literals;
using namespace gismo;

// Defining displacement function
using DispFunc = std::function<std::array<double,1>(const std::array<double,1>&)>;

/// @brief Specialization of the IgANet class for non linear neo-Hookean material. 
// This class is defined to work with a square where Dirichlet conditions are applied to left and right boundary 
// and top and bottom boundary are left traction free. 
// If you want to have other sides traction free, adapt the loss function manually.
// This method first minimizes strain energy and then divergence according to the pde.
// Therefore this setup works well only for no specified tractions and no body forces.
// If you want to integrate those, the loss function for energy minimization needs adaptation.
template <typename Optimizer, typename GeometryMap, typename Variable>
class neo_Hook : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
                          public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  typename Base::variable_collPts_type collPts_;
  typename Base::variable_collPts_type interiorCollPts_;
  std::array<torch::Tensor, 2ul> tractionCollPts_;

  int nrCollPts_;
  Variable ref_;

  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_tf_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_tf_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_interior_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_interior_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_boundary_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_boundary_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_tf_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_tf_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_interior_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_interior_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_boundary_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_boundary_;

  // material properties - lame's parameters
  double lambda_;
  double mu_;

  // simulation parameter
  double MAX_EPOCH_;
  double MIN_LOSS_;
  std::vector<std::tuple<int,DispFunc,DispFunc>> DIRI_SIDES_;

  // solver options
  const torch::optim::LBFGSOptions& SOLVER_OPTS;

  // json output path
  const char* JSON_PATH;

public:
  /// @brief Constructor
  template <typename... Args>
  neo_Hook(double lambda, double mu, double MAX_EPOCH, 
                    double MIN_LOSS, const char* json_path, const torch::optim::LBFGSOptions& solver_opts,
                    std::vector<std::tuple<int,DispFunc,DispFunc>> DIRI_SIDES, 
                    std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        lambda_(lambda), mu_(mu), MAX_EPOCH_(MAX_EPOCH), 
        MIN_LOSS_(MIN_LOSS), JSON_PATH(json_path), SOLVER_OPTS(solver_opts), DIRI_SIDES_(std::move(DIRI_SIDES)), 
        ref_(iganet::utils::to_array(8_i64, 8_i64)) {
            this->initialize_dirichlet_boundaries();
        }

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the interior collocation points
  auto const &interiorCollPts() const { return interiorCollPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  void initialize_dirichlet_boundaries() {
    // run through all DIRI_SIDES and register respective lambdas
    for (const auto& side : this->DIRI_SIDES_) {
        int sideNr = std::get<0>(side);
        DispFunc xDispFun = std::get<1>(side);
        DispFunc yDispFun = std::get<2>(side);

        switch (sideNr) {
            case 1:
                this->ref_.boundary().template side<1>().template transform<1>(
                    xDispFun, std::array<iganet::short_t,1>{0});
                this->ref_.boundary().template side<1>().template transform<1>(
                    yDispFun, std::array<iganet::short_t,1>{1});
                break;
            case 2:
                this->ref_.boundary().template side<2>().template transform<1>(
                    xDispFun, std::array<iganet::short_t,1>{0});
                this->ref_.boundary().template side<2>().template transform<1>(
                    yDispFun, std::array<iganet::short_t,1>{1});
                break;
            case 3:
                this->ref_.boundary().template side<3>().template transform<1>(
                    xDispFun, std::array<iganet::short_t,1>{0});
                this->ref_.boundary().template side<3>().template transform<1>(
                    yDispFun, std::array<iganet::short_t,1>{1});
                break;
            case 4:
                this->ref_.boundary().template side<4>().template transform<1>(
                    xDispFun, std::array<iganet::short_t,1>{0});
                this->ref_.boundary().template side<4>().template transform<1>(
                    yDispFun, std::array<iganet::short_t,1>{1});
                break;
            default:
                std::cerr << "Error: Invalid side number " << sideNr << std::endl;
        }
    }
  }
  
  void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
    
    // create json object
    nlohmann::json jsonData;

    // try to read the JSON data from the file
    try {
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


    // add new data to the JSON object
    try {
        jsonData[key] = data;
    } catch (const std::exception& e) {
        std::cerr << "Error adding key to JSON object: " << e.what() << "\n";
        return;
    }

    // write the JSON data to the file
    try {
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
  }

  /// @brief helper function to calculate the Greville abscissae
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

  void write_result() {
    // write geometry and solution spline data to file
    appendToJsonFile("G", Base::G_.to_json());
    appendToJsonFile("u", Base::u_.to_json());
  }


  /// @brief Initializes the epoch
  bool epoch(int64_t epoch) override {
    // print epoch number
    std::cout << "Epoch: " << epoch << std::endl;

    if (epoch == 0) {
      // set the solver options
      this->optimizerOptionsReset(SOLVER_OPTS);

      Base::inputs(epoch);

      collPts_ = Base::variable_collPts(iganet::collPts::greville);
      interiorCollPts_ = Base::variable_collPts(iganet::collPts::greville_interior);
      // WARNING, only works for equal number of control points in x and y direction
      nrCollPts_ = static_cast<int>(std::sqrt(std::get<0>(collPts_)[0].size(0)));

    // WARNING! Neumann Loss is hardcoded right now for top and bottom boundary
      std::vector<torch::Tensor> tractionCollPtsX;
      std::vector<torch::Tensor> tractionCollPtsY;
      at::Tensor collPts_temp = std::get<0>(collPts_.second)[0];
      tractionCollPtsX.push_back(collPts_temp.slice(0, 1, -1));
      tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 2}));
      tractionCollPtsX.push_back(collPts_temp.slice(0, 1, -1));
      tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 2}));
      tractionCollPts_ = {
        torch::cat(tractionCollPtsX, 0), 
        torch::cat(tractionCollPtsY, 0)};


      torch::Tensor collPtsCoeffs = std::get<0>(collPts_)[0].slice(0, 0, nrCollPts_);
      nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
      for (int i = 0; i < collPtsCoeffs.size(0); ++i) {
          collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
      }

      var_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      var_knot_indices_interior_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
                interiorCollPts_.first);
      var_coeff_indices_interior_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_interior_);

      G_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      G_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_);

      G_knot_indices_interior_ = 
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              interiorCollPts_.first);
      G_coeff_indices_interior_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_interior_);

    var_knot_indices_tf_ =
        Base::f_.template find_knot_indices<iganet::functionspace::interior>(
        tractionCollPts_);
    var_coeff_indices_tf_ =
        Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
        var_knot_indices_tf_);
    G_knot_indices_tf_ =
        Base::G_.template find_knot_indices<iganet::functionspace::interior>(
            tractionCollPts_);
    G_coeff_indices_tf_ =
    Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
        G_knot_indices_tf_);

      return true;
    } 
    else if (epoch == 2) {
        this->optimizerReset();
        this->optimizerOptionsReset(SOLVER_OPTS);
    }
    else if (epoch == MAX_EPOCH_-1) {
        // write geometry and solution spline data to file
        write_result();
    }
      return false;
  }

  /// @brief Computes the loss function
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    // create u_ from the training's outputs
    Base::u_.from_tensor(outputs);

    // pre-allocation of the loss values
    torch::Tensor totalLoss; 
    torch::Tensor elastLoss;
    std::optional<torch::Tensor> bcLoss;
  
    // create command line output variable for all the different losses
    std::ostringstream singleLossOutput;
        
    // first we minimize strain energy, after 20 epochs we change to minimizing divergence according to pde
    if (epoch >= 2) {

        // Elasticity Loss

        // calculate the jacobian of the displacements (u) at the interior collocation points
        auto jacobian = Base::u_.ijac(Base::G_, interiorCollPts_.first, 
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_, G_coeff_indices_interior_);
        
        auto& u1_x = jacobian(0);
        auto& u1_y = jacobian(1);
        auto& u2_x = jacobian(2);
        auto& u2_y = jacobian(3);

        // calculation of the second derivatives of the interior displacements (u)
        auto hessianColl = Base::u_.ihess(Base::G_, interiorCollPts_.first, 
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_, G_coeff_indices_interior_);

        // partial derivatives of the displacements (u)
        auto& u1_xx = hessianColl(0,0,0);
        auto& u1_xy = hessianColl(0,1,0);
        auto& u1_yx = hessianColl(1,0,0);
        auto& u1_yy = hessianColl(1,1,0);

        auto& u2_xx = hessianColl(0,0,1);
        auto& u2_xy = hessianColl(0,1,1);
        auto& u2_yx = hessianColl(1,0,1);
        auto& u2_yy = hessianColl(1,1,1);

        double beta = 10.0;
        // Divergence of first Kirchhoff stress tensor
        auto J = 1.0 + u1_x + u2_y + u1_x*u2_y - u1_y*u2_x;
        auto J_sp = torch::nn::functional::softplus(J, torch::nn::functional::SoftplusFuncOptions()
            .beta(beta)
            .threshold(20.0));
        auto J_x = u1_xx + u2_yx + u1_xx*u2_y +u1_x*u2_yx - u1_yx*u2_x - u1_y*u2_xx;
        auto J_y = u1_xy + u2_yy + u1_xy*u2_y +u1_x*u2_yy - u1_yy*u2_x - u1_y*u2_xy;
        auto A = (lambda_ * torch::log1p(J_sp - 1) - mu_) / J_sp;
        auto A_x = (lambda_ + mu_ - lambda_*torch::log1p(J_sp - 1)) * J_x / (J_sp*J_sp);
        auto A_y = (lambda_ + mu_ - lambda_*torch::log1p(J_sp - 1)) * J_y / (J_sp*J_sp);

        // First derivatives of first Piola Kirchhoff stress tensor
        auto P11_x = mu_*u1_xx + A_x + A_x*u2_y + A*u2_yx;
        auto P12_y = mu_*u1_yy - A_y*u2_x - A*u2_xy;
        auto P21_x = mu_*u2_xx - A_x*u1_y - A*u1_yx;
        auto P22_y = mu_*u2_yy + A_y + A_y*u1_x + A*u1_xy;

        // Divergence of first PK tensor
        auto divPX = P11_x + P12_y;
        auto divPY = P21_x + P22_y;
        auto divP  = torch::stack({divPX, divPY}, 1);

        // BODY FORCE

        // evaluate the reference body force f at all interior collocation points
        auto f = Base::f_.eval(interiorCollPts_.first);

        auto bodyForce = torch::stack({*f[0], *f[1]}, 1).to(divP.dtype());

        // calculation of the loss function for double-sided constraint solid
        // div(sigma) + f = 0 --> div(sigma) = -f
        elastLoss = torch::mse_loss(divP, bodyForce);
        
        // add the elasticity loss to the total loss
        totalLoss = elastLoss;

        // add the elasticity loss to the cmd-output variable
        singleLossOutput << "EL " << std::setw(11) << elastLoss.item<double>();


        // ------------------------------ Neumann Loss ------------------------------------
        // WARNING! Neumann Loss is hardcoded right now for top and bottom boundary

        // calculate the jacobian of the displacements (u) at the collocation points
        auto jacobian_tf = Base::u_.ijac(Base::G_, tractionCollPts_, 
            var_knot_indices_tf_, var_coeff_indices_tf_,
            G_knot_indices_tf_, G_coeff_indices_tf_);
        
        auto& u1_x_tf = jacobian_tf(0);
        auto& u1_y_tf = jacobian_tf(1);
        auto& u2_x_tf = jacobian_tf(2);
        auto& u2_y_tf = jacobian_tf(3);

        // calculation of the second derivatives of the displacements (u)
        auto hessianColl_tf = Base::u_.ihess(Base::G_, tractionCollPts_, 
            var_knot_indices_tf_, var_coeff_indices_tf_,
            G_knot_indices_tf_, G_coeff_indices_tf_);

        // partial derivatives of the displacements (u)
        auto& u1_xx_tf = hessianColl_tf(0,0,0);
        auto& u1_xy_tf = hessianColl_tf(0,1,0);
        auto& u1_yx_tf = hessianColl_tf(1,0,0);
        auto& u1_yy_tf = hessianColl_tf(1,1,0);

        auto& u2_xx_tf = hessianColl_tf(0,0,1);
        auto& u2_xy_tf = hessianColl_tf(0,1,1);
        auto& u2_yx_tf = hessianColl_tf(1,0,1);
        auto& u2_yy_tf = hessianColl_tf(1,1,1);

        // Divergence of first Kirchhoff stress tensor
        auto J_tf = 1.0 + u1_x_tf + u2_y_tf + u1_x_tf*u2_y_tf - u1_y_tf*u2_x_tf;
        auto J_sp_tf = torch::nn::functional::softplus(J_tf, torch::nn::functional::SoftplusFuncOptions()
            .beta(beta)
            .threshold(20.0));
        auto J_x_tf = u1_xx_tf + u2_yx_tf + u1_xx_tf*u2_y_tf +u1_x_tf*u2_yx_tf - u1_yx_tf*u2_x_tf - u1_y_tf*u2_xx_tf;
        auto J_y_tf = u1_xy_tf + u2_yy_tf + u1_xy_tf*u2_y_tf +u1_x_tf*u2_yy_tf - u1_yy_tf*u2_x_tf - u1_y_tf*u2_xy_tf;
        auto A_tf = (lambda_ * torch::log1p(J_sp_tf - 1) - mu_) / J_sp_tf;
        auto A_x_tf = (lambda_ + mu_ - lambda_*torch::log1p(J_sp_tf - 1)) * J_x_tf / (J_sp_tf*J_sp_tf);
        auto A_y_tf = (lambda_ + mu_ - lambda_*torch::log1p(J_sp_tf - 1)) * J_y_tf / (J_sp_tf*J_sp_tf);

        // Components of first Piola Kirchhoff stress tensor
        auto P11 = mu_ * (1 + u1_x_tf) + A_tf * (1 + u2_y_tf);
        auto P12 = mu_ * u1_y_tf - A_tf * u2_x_tf; 
        auto P21 = mu_ * u2_x_tf - A_tf * u1_y_tf;
        auto P22 = mu_ * (1 + u2_y_tf) + A_tf * (1 + u1_x_tf);

        totalLoss += torch::mse_loss(P12, torch::zeros_like(P12));
        totalLoss += torch::mse_loss(P22, torch::zeros_like(P22));

    } else {
        // energy minimization first
        // calculate the jacobian of the displacements (u) at the collocation points
        auto jacobian = Base::u_.ijac(Base::G_, collPts_.first, 
            var_knot_indices_, var_coeff_indices_,
            G_knot_indices_, G_coeff_indices_);
        
        auto& u1_x = jacobian(0);
        auto& u1_y = jacobian(1);
        auto& u2_x = jacobian(2);
        auto& u2_y = jacobian(3);

        double beta = 2.0;
        // Divergence of first Kirchhoff stress tensor
        auto J = 1.0 + u1_x + u2_y + u1_x*u2_y - u1_y*u2_x;
        auto J_sp = torch::nn::functional::softplus(J, torch::nn::functional::SoftplusFuncOptions()
            .beta(beta)
            .threshold(20.0));

        auto W = lambda_/2 * torch::square(torch::log(J_sp)) + mu_/2 * (torch::square(1+u1_x)+torch::square(u1_y)+torch::square(u2_x)+torch::square(1+u2_y)-2-2*torch::log(J_sp));
        totalLoss = torch::mse_loss(W, torch::zeros_like(W));

        // add the elasticity loss to the cmd-output variable
        singleLossOutput << "EL " << std::setw(11) << totalLoss.item<double>();
    }




    // only consider BC loss if dirichlet BCs are applied
    if (!DIRI_SIDES_.empty()) {
        // add a BC weight for penalization of the training
        int bcWeight = 1e7;
        // initialize bcLoss variable
        bcLoss = torch::tensor(0.0);

        // evaluation of the displacements at the boundary points
        auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
        // evaluation of the displacements at the reference boundary points
        auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

        // loop through all dirichlet sides
        for (const auto& side : DIRI_SIDES_) {
            int sideNr = std::get<0>(side);
            
            switch (sideNr) {
                case 1: 
                    *bcLoss += bcWeight * (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                    break;
                case 2:
                    *bcLoss += bcWeight * (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                    break;
                case 3:
                    *bcLoss += bcWeight * (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                    break;
                case 4:
                    *bcLoss += bcWeight * (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
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
    std::cout << std::setw(11) << totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;

    return totalLoss;
  }
};

int main(int argc, char* argv[]) {
  iganet::init();
  iganet::verbose(std::cout);

  // ------- USER INPUTS ------- //

  // material parameters
  double YOUNG_MODULUS = 10;
  double POISSON_RATIO = 0.3;

  // simulation parameters
  int MAX_EPOCH = 300;
  double MIN_LOSS = 1e-9;
  bool SUPERVISED_LEARNING = false;
  bool RUN_REF_SIM = false;

  // spline parameters
  int64_t NR_CTRL_PTS;  // in each direction 
  int NR_CTRL_PTS_temp = 5;
  constexpr int DEGREE = 4;
  double nodes_factor = 1.0;
  std::string json_path_temp = "/home/chg/Programming/PythonNet_IGA/Network_V1/NeuralNet/";

  gsCmdLine cmd("Square being stretched with nonlinear elasticity solver.");
  cmd.addInt("n","numbercontrolpoints","number control points",NR_CTRL_PTS_temp);
  cmd.addReal("f","nodes_factor","nodes factor",nodes_factor);
  cmd.addString("p","pathname", "name of output path", json_path_temp);
  try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

  NR_CTRL_PTS = static_cast<int64_t>(NR_CTRL_PTS_temp);
  std::string full_json_path_temp = json_path_temp + "iganet_result.json";
  const char* json_path = full_json_path_temp.c_str();
  std::string csv_path = json_path_temp + "runtimes.csv";


  // solver options
  auto solver_options = torch::optim::LBFGSOptions(1.0).
                                        max_iter(150).
                                        max_eval(100).
                                        history_size(200).
                                        tolerance_grad(1e-12).
                                        tolerance_change(1e-12).
                                        line_search_fn("strong_wolfe");


  // Dirichlet boundary conditions
  DispFunc zeroDisp = [](auto const& xi) {return std::array<double,1>{ 0.0 };};
  DispFunc Disp2 = [](auto const& xi) {
    double s = xi[0];
    return std::array<double,1>{2.0};};


  std::vector<std::tuple<int,DispFunc,DispFunc>> DIRI_SIDES = {
      {1, zeroDisp, zeroDisp},
      {2, Disp2, zeroDisp}};
    
  // --------------------------- //


  // calculation of lame parameters
  double lambda = (YOUNG_MODULUS * POISSON_RATIO) / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  double mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO));

  using real_t = double;
  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
  using neo_Hook_t = neo_Hook<optimizer_t, geometry_t, variable_t>;
  
  auto number_nodes = std::max(1, static_cast<int>(std::round(NR_CTRL_PTS*NR_CTRL_PTS*2*nodes_factor)));

    neo_Hook_t
      net(// simulation parameters
          lambda, mu, MAX_EPOCH, MIN_LOSS, json_path, solver_options, std::move(DIRI_SIDES),
          // Number of neurons per layer
          {number_nodes, number_nodes},
          // Activation functions
          {{iganet::activation::tanh},
           {iganet::activation::tanh},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry
          std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)));

  // imposing body force
  net.f().transform([=](const std::array<real_t, 2> xi) {
    // body force {f_x, f_y}
    return std::array<real_t, 2>{0, 0};
  });


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
  auto runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
  iganet::Log(iganet::log::info)
      << "Training took "
      << runtime
      << " seconds\n";

  net.write_result();

  // write the runtimes
  std::ofstream outFile(csv_path, std::ios_base::app);
  outFile << std::log2(NR_CTRL_PTS-DEGREE) << ", " << DEGREE << ", " << std::fixed << std::setprecision(1) << nodes_factor << std::fixed << std::setprecision(5) << ", " << runtime << std::endl;

  iganet::finalize();
  return 0; 
}
