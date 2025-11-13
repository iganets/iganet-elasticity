#include <iganet.h>
#include <iostream>
#include <fstream>
#include <ElasticityForLocalizedMaterial.hpp>

int main() {
    iganet::init();
    iganet::verbose(std::cout);

    // USER INPUTS ----------------------------------------------------------------------------------------------------------
    //    There are two options for the user input. Either you modify 
    //    the default parameters where they are initalized or you set
    //    choice below to 'json' to have a quick access userinput.
    // simulation parameters
    int MAX_EPOCH = 100;
    double MIN_LOSS = 1e-12;
    bool SUPERVISED_LEARNING = false;
    std::string JSON_PATH = "/home/isabellaunix/DevelDA/singerDA/ConfigResult/result.json";     // SetBeforeSim    

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

    
    std::string choice = "json";     // 'json' for json input. OPTIONAL .json input for easy change of params. no need to rebuild :) auszukommentieren, wenn nicht gebraucht.
    
    
    if (choice == "json") {
        std::ifstream file("/home/isabellaunix/DevelDA/singerDA/ConfigResult/config.json");     // SetBeforeSim
        if (!file) {std::cerr << "Could not open config.json\n";
                    return 1;}
        nlohmann::json j;
        file >> j;

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
        BODY_FORCE.first = j["body_force"][0];
        BODY_FORCE.second = j["body_force"][1];

        // just to verify
        std::cout << "TFBC sides: ";
        for (auto side : TFBC_SIDES) std::cout << side << " ";
        std::cout << "\n";
    }
    // --------------------------- //

    using real_t = double;
    using namespace iganet::literals;
    using optimizer_t = torch::optim::LBFGS;
    
    using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, 3, 3>>;   
    using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, 4, 4>>;
    using material_t = iganet::S<iganet::UniformBSpline<real_t, 2, 4, 4>>;

    using inputs_t = std::tuple<geometry_t, variable_t, material_t>;     
    using outputs_t = std::tuple<variable_t>;     
    using linear_elasticity_t = ElasticityForLocalizedMaterial<optimizer_t, inputs_t, outputs_t>;

    linear_elasticity_t net2(TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, NR_CTRL_PTS, JSON_PATH);

    linear_elasticity_t net( // simulation parameters
        SUPERVISED_LEARNING, MAX_EPOCH, MIN_LOSS, solver_options, 
        TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, NR_CTRL_PTS, JSON_PATH,
        // Number of neurons per layer
        {25, 25},
        // Activation functions
        {{iganet::activation::sigmoid},
            {iganet::activation::sigmoid},
            {iganet::activation::none}},
        // Number of B-spline coefficients of input in inputs_t
        std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS),
                   iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS),
                   iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)),
        // Number of B-spline coefficients of output in outputs_t
        std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS))
    );

    // xml in net.template input<1>().eval(collPts.first)
    pugi::xml_document xml;
    xml.load_file("/home/isabellaunix/DevelDA/singerDA/ConfigResult/mat.xml");      // SetBeforeSim
    net2.template input<2>().from_xml(xml);

    // imposing body force f
    net2.template input<1>().transform([=](const std::array<real_t, 2> xi) {
        return std::array<real_t, 2>{BODY_FORCE.first, BODY_FORCE.second};
    });

    // get  coefficients of  control points
    torch::Tensor ctrlPtsCoeffs = net.template input<0>().as_tensor().slice(0, 0, NR_CTRL_PTS);
    nlohmann::json ctrlPtsCoeffs_j = nlohmann::json::array();
    for (int i = 0; i < NR_CTRL_PTS; ++i) {
        ctrlPtsCoeffs_j.push_back({ctrlPtsCoeffs[i].item<double>()});
    }
    net2.appendToJsonFile("net_ctrlPtsCoeffs", ctrlPtsCoeffs_j);

    // run through all DIRI_SIDES
    for (const auto& side : DIRI_SIDES) {
        int sideNr = std::get<0>(side);
        double xDispl = std::get<1>(side);
        double yDispl = std::get<2>(side);

        switch (sideNr) {
            case 1:
                net2.ref().boundary().side<1>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net2.ref().boundary().side<1>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 2:
                net2.ref().boundary().side<2>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net2.ref().boundary().side<2>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 3:
                net2.ref().boundary().side<3>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net2.ref().boundary().side<3>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 4:
                net2.ref().boundary().side<4>().transform<1>(
                    [=](const std::array<real_t, 1> &xi) {
                        return std::array<real_t, 1>{xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                net2.ref().boundary().side<4>().transform<1>(
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

    // net.options().max_epoch(MAX_EPOCH);     // Set maximum number of epochs
    // net.options().min_loss(MIN_LOSS);       // Set tolerance for  loss functions
    // net.options().min_loss_change(0);       // overwrite. to only have max epoch and min loss as stopping criteria
    // net.options().min_loss_rel_change(0);   // overwrite.
    
    // Train network ----------------------------------------------------------------------------------------------------------
    net2.load("/home/isabellaunix/DevelDA/singerDA/ConfigResult/trained_iganet.pt");
    net2.eval();
    // std::cout << net2 << std::endl;

    // POSTPROCESSING ----------------------------------------------------------------------------------------------------------
    //  get  geometry and displacement as tensors
    torch::Tensor geometryAsTensor = net2.template input<0>().as_tensor();
    torch::Tensor displacementAsTensor = net2.template output<0>().as_tensor();
    
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
    net2.appendToJsonFile("net_CtrlPts", displacedNetCtrlPts_j);
    net2.appendToJsonFile("net_Degree", DEGREE);
    
    iganet::finalize();
    return 0;
}