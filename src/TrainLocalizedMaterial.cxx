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
    int NONEURONS = 25;
    int WEIGHT1 = 1;

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

    // OPTIONAL .json input for easy change of params.
    std::string choice = "json";       // 'json' for json input.

    if (choice == "json") {
        std::ifstream file("/home/isabellaunix/DevelDA/singerDA/ConfigResult/config.json");     // SetBeforeSim
        if (!file) {
            std::cerr << "Could not open config.json\n";
            return 1;
        }
        nlohmann::json j;
        file >> j;

        // simulation parameters
        MAX_EPOCH = j["simulation"]["max_epoch"];
        MIN_LOSS = j["simulation"]["min_loss"];
        SUPERVISED_LEARNING = j["simulation"]["supervised_learning"];
        std::string JSON_PATH = j["simulation"]["json_path"];
        NONEURONS = j["simulation"]["noNeurons"];
        WEIGHT1 = j["simulation"]["weight1"];


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
    // --------------------------- //

    using real_t = double;
    using namespace iganet::literals;
    using optimizer_t = torch::optim::LBFGS;
    
    using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;   
    using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
    using material_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;

    using inputs_t = std::tuple<geometry_t, variable_t, material_t>;     
    using outputs_t = std::tuple<variable_t>;     
    using linear_elasticity_t = ElasticityForLocalizedMaterial<optimizer_t, inputs_t, outputs_t>;
        
    linear_elasticity_t net( // simulation parameters
        SUPERVISED_LEARNING, MAX_EPOCH, MIN_LOSS, solver_options, 
        TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, NR_CTRL_PTS, JSON_PATH, WEIGHT1,
        // Number of neurons per layer
        {NONEURONS, NONEURONS},
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
    net.template input<2>().from_xml(xml);

    // imposing body force f
    net.template input<1>().transform([=](const std::array<real_t, 2> xi) {
        return std::array<real_t, 2>{BODY_FORCE.first, BODY_FORCE.second};
    });

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

    net.options().max_epoch(MAX_EPOCH);     // Set maximum number of epochs
    net.options().min_loss(MIN_LOSS);       // Set tolerance for  loss functions
    net.options().min_loss_change(0);       // overwrite. to only have max epoch and min loss as stopping criteria
    net.options().min_loss_rel_change(0);   // overwrite.

    // Start time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    // Train network ----------------------------------------------------------------------------------------------------------
    net.train();

    // std::cout << typeid(net).name() << std::endl;
    // Stop time measurement
    auto t2 = std::chrono::high_resolution_clock::now();
    iganet::Log(iganet::log::info)
        << "Training took "
        << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                .count()
        << " seconds\n";

    // PostProcessing ----------------------------------------------------------------------------------------------------------
    net.PostProc();     // rausschreiben der postprocessing größen
    // net.save("/home/isabellaunix/DevelDA/singerDA/ConfigResult/trained_iganet.pt");     // save the state of the trained network to evaluate later

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
    net.appendToJsonFile("net_CtrlPts", displacedNetCtrlPts_j);     // deformed
    net.appendToJsonFile("net_Degree", DEGREE);
    
    iganet::finalize();
    return 0;
}