
#include "lin_elasticity_net.hpp"

#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace iganet::literals;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using iganet_elasticity::utils::config::require;

template <int DEGREE>
int run(
    const std::filesystem::path& repoRoot,
    const std::filesystem::path& xmlPath,
    const std::filesystem::path& resultJsonPath,
    GeometryMode                 geoMode,
    const nlohmann::json&        j,
    double lambda, double mu,
    bool   supervisedLearning,
    int    maxEpoch, double minLoss,
    std::array<double,3>                              bodyForce,
    std::vector<int>                                  tfbcSides,
    std::vector<std::tuple<int,double,double,double>> forceSides,
    std::vector<std::tuple<int,double,double,double>> diriSides,
    int64_t                                           nrCtrlPts,
    int                                               degreeRef,
    bool                                              runGsRefSim,
    double youngModulus, double poissonRatio)
{
    using real_t      = double;
    using optimizer_t = torch::optim::LBFGS;
    using geometry_t  = iganet::S<iganet::NonUniformBSpline<real_t, 3, DEGREE, DEGREE, DEGREE>>;
    using variable_t  = iganet::S<iganet::NonUniformBSpline<real_t, 3, DEGREE, DEGREE, DEGREE>>;
    using net_t       = linear_elasticity<optimizer_t, geometry_t, variable_t>;

    const std::string jsonPath = resultJsonPath.string();

    std::array<std::vector<real_t>, 3> knotVectors;
    XmlGeometryData xmlData;   // leer wenn parametrisch

    if (geoMode == GeometryMode::Xml) {

        // --- XML-Modus: Knotenvektoren + Kontrollpunkte aus Datei ---
        xmlData = loadXmlKnotVectors(xmlPath, "100", DEGREE);
        knotVectors = xmlData.knotVectors;
        nrCtrlPts   = xmlData.nCtrlPts;  // Überschreibt den Config-Wert

    } else {

        // --- Parametrischer Modus: uniformer Einheitswürfel ---
        // nr_ctrl_pts aus geometry-Block (falls vorhanden), sonst aus spline-Block
        if (j.contains("geometry") && j["geometry"].contains("nr_ctrl_pts"))
            nrCtrlPts = j["geometry"]["nr_ctrl_pts"].get<int64_t>();

        for (int d = 0; d < 3; ++d)
            knotVectors[d] = makeUniformKnotVector(static_cast<int>(nrCtrlPts), DEGREE);

        std::cout << "[Parametric] nrCtrlPts=" << nrCtrlPts
                  << "  KV-Länge=" << knotVectors[0].size() << "\n";
    }

    net_t net(
        lambda, mu, supervisedLearning, maxEpoch, minLoss,
        bodyForce, tfbcSides, forceSides, diriSides,
        nrCtrlPts, jsonPath,
        {25, 30},
        {{iganet::activation::sigmoid},
         {iganet::activation::sigmoid},
         {iganet::activation::none}},
        std::make_tuple(std::make_tuple(knotVectors)),  // Geometrie
        std::make_tuple(std::make_tuple(knotVectors))); // Variable

    if (geoMode == GeometryMode::Xml) {

        // Kontrollpunkte und KVs aus XML laden
        net.template input<0>().from_xml(xmlData.doc, 0);

    } else {

        std::array<double,3> origin = {0.0, 0.0, 0.0};
        std::array<double,3> scale  = {1.0, 1.0, 1.0};

        if (j.contains("geometry")) {
            const auto& gj = j["geometry"];
            if (gj.contains("origin"))
                origin = {gj["origin"][0].get<double>(),
                          gj["origin"][1].get<double>(),
                          gj["origin"][2].get<double>()};
            if (gj.contains("scale"))
                scale  = {gj["scale"][0].get<double>(),
                          gj["scale"][1].get<double>(),
                          gj["scale"][2].get<double>()};
        }

        net.template input<0>().transform(
            [origin, scale](const std::array<real_t,3>& xi) -> std::array<real_t,3> {
                return {origin[0] + scale[0]*xi[0],
                        origin[1] + scale[1]*xi[1],
                        origin[2] + scale[2]*xi[2]};
            });

        std::cout << "[Parametric] Geometrie gesetzt:"
                  << "  origin=(" << origin[0] << "," << origin[1] << "," << origin[2] << ")"
                  << "  scale=(" << scale[0] << "," << scale[1] << "," << scale[2] << ")\n";
    }

  
    auto setComponent = [&](auto& boundary, double value, iganet::short_t comp) {
        boundary.template transform<1>(
            [value](const std::array<real_t,2>&) -> std::array<real_t,1> {
                return {value};
            },
            std::array<iganet::short_t,1>{comp});
    };

    for (const auto& side : diriSides) {
        const int    sNr   = std::get<0>(side);
        const double xDisp = std::get<1>(side);
        const double yDisp = std::get<2>(side);
        const double zDisp = std::get<3>(side);

        switch (sNr) {
            case 1:
                setComponent(net.ref().boundary().template side<1>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<1>(), yDisp, 1);
                setComponent(net.ref().boundary().template side<1>(), zDisp, 2);
                break;
            case 2:
                setComponent(net.ref().boundary().template side<2>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<2>(), yDisp, 1);
                setComponent(net.ref().boundary().template side<2>(), zDisp, 2);
                break;
            case 3:
                setComponent(net.ref().boundary().template side<3>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<3>(), yDisp, 1);
                setComponent(net.ref().boundary().template side<3>(), zDisp, 2);
                break;
            case 4:
                setComponent(net.ref().boundary().template side<4>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<4>(), yDisp, 1);
                setComponent(net.ref().boundary().template side<4>(), zDisp, 2);
                break;
            case 5:
                setComponent(net.ref().boundary().template side<5>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<5>(), yDisp, 1);
                setComponent(net.ref().boundary().template side<5>(), zDisp, 2);
                break;
            case 6:
                setComponent(net.ref().boundary().template side<6>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<6>(), yDisp, 1);
                setComponent(net.ref().boundary().template side<6>(), zDisp, 2);
                break;
            default:
                std::cerr << "Error: Invalid side number " << sNr << "\n";
        }
    }

    net.options().max_epoch(maxEpoch);
    net.options().min_loss(minLoss);
    net.initialize_problem_data();

    auto t1 = std::chrono::high_resolution_clock::now();
    // net.train();
    auto t2 = std::chrono::high_resolution_clock::now();
    iganet::Log(iganet::log::info)
        << "Training took "
        << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
        << " seconds\n";

 
    net.PostProc();

    torch::Tensor geomTensor = net.template input<0>().as_tensor();
    int64_t totalSize = geomTensor.size(0);

    if (totalSize % 3 != 0) {
        std::cerr << "Error: Geometry tensor size not divisible by 3 (size="
                  << totalSize << ")\n";
        return 1;
    }
    int64_t nCtrlPtsXml = totalSize / 3;

    // Debug-Ausgabe
    std::cout << "\n=== CONTROL POINT DEBUG ===\n";
    std::cout << "Total tensor size: " << totalSize << "\n";
    std::cout << "Number of control points: " << nCtrlPtsXml << "\n";
    std::cout << "First CP: x=" << geomTensor[0].item<double>()
              << "  y=" << geomTensor[nCtrlPtsXml].item<double>()
              << "  z=" << geomTensor[2*nCtrlPtsXml].item<double>() << "\n";
    std::cout << "Last CP:  x=" << geomTensor[nCtrlPtsXml-1].item<double>()
              << "  y=" << geomTensor[2*nCtrlPtsXml-1].item<double>()
              << "  z=" << geomTensor[3*nCtrlPtsXml-1].item<double>() << "\n";
    std::cout << "===========================\n";

    torch::Tensor netCtrlPts = torch::zeros({nCtrlPtsXml, 3}, geomTensor.options());
    for (int64_t i = 0; i < nCtrlPtsXml; ++i) {
        netCtrlPts[i][0] = geomTensor[i].item<double>();
        netCtrlPts[i][1] = geomTensor[i + nCtrlPtsXml].item<double>();
        netCtrlPts[i][2] = geomTensor[i + 2*nCtrlPtsXml].item<double>();
    }

    nlohmann::json netCtrlPts_j = nlohmann::json::array();
    for (int64_t i = 0; i < netCtrlPts.size(0); ++i)
        netCtrlPts_j.push_back({netCtrlPts[i][0].item<double>(),
                                 netCtrlPts[i][1].item<double>(),
                                 netCtrlPts[i][2].item<double>()});

    net.appendToJsonFile("net_CtrlPts",          netCtrlPts_j);
    net.appendToJsonFile("net_nrCtrlPtsFromXml",  nCtrlPtsXml);
    net.appendToJsonFile("net_Degree",            DEGREE);

    //  Optional: GISMO-Referenzsimulation

#ifdef IGANET_WITH_GISMO
    if (runGsRefSim) {
        // GISMO erwartet 2D-Tupel für DIRI_SIDES / FORCE_SIDES
        // Konvertierung: (side,ux,uy,uz) → (side,ux,uy)  [nur x/y verwendet]
        std::vector<std::tuple<int,double,double>> diriGs, forceGs;
        for (const auto& d : diriSides)
            diriGs.emplace_back(std::get<0>(d), std::get<1>(d), std::get<2>(d));
        for (const auto& f : forceSides)
            forceGs.emplace_back(std::get<0>(f), std::get<1>(f), std::get<2>(f));

        std::pair<double,double> bodyForce2D = {bodyForce[0], bodyForce[1]};

        auto [gsOrigin, gsDispl, gsStress] = net_t::RunGismoSimulation(
            nrCtrlPts, DEGREE, youngModulus, poissonRatio,
            diriGs, forceGs, bodyForce2D);

        auto gsCtrlPts = gsOrigin + gsDispl;
        nlohmann::json gsOrigin_j=nlohmann::json::array(), gsDispl_j=nlohmann::json::array();
        nlohmann::json gsCtrl_j=nlohmann::json::array(),   gsStress_j=nlohmann::json::array();
        for (int i = 0; i < gsCtrlPts.size(0); ++i) {
            gsOrigin_j.push_back({gsOrigin[i][0].item<double>(), gsOrigin[i][1].item<double>()});
            gsDispl_j .push_back({gsDispl[i][0].item<double>(),  gsDispl[i][1].item<double>()});
            gsCtrl_j  .push_back({gsCtrlPts[i][0].item<double>(),gsCtrlPts[i][1].item<double>()});
            gsStress_j.push_back({gsStress[i][0].item<double>()});
        }
        net.appendToJsonFile("gsOriginCtrlPts", gsOrigin_j);
        net.appendToJsonFile("gsDisplacements", gsDispl_j);
        net.appendToJsonFile("gsCtrlPts",       gsCtrl_j);
        net.appendToJsonFile("gsStresses",      gsStress_j);
    }
#endif

    return 0;
}

int main() {
    iganet::init();
    iganet::verbose(std::cout);

    // Repo-Wurzel bestimmen
    std::filesystem::path repoRoot;
    try {
        repoRoot = repo_root_from_build_exe();
    } catch (const std::exception& e) {
        std::cerr << "Could not determine repo root: " << e.what() << "\n";
        return 1;
    }

    const auto CONFIG_PATH  = repoRoot / "sim_config.json";
    const auto RESULT_PATH  = repoRoot / "result.json";
    const auto XML_PATH     = repoRoot / "bone_simplified.xml";

    std::ifstream cfgFile(CONFIG_PATH);
    if (!cfgFile) {
        std::cerr << "Could not open config file: " << CONFIG_PATH << "\n";
        return 1;
    }
    nlohmann::json j;
    try {
        cfgFile >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse config JSON: " << e.what() << "\n";
        return 1;
    }

    const std::string cmd =
        "cd \"" + repoRoot.string() + "\" && python3 run_std_coll.py";
    const int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "ERROR: python reference run failed. system() returned "
                  << ret << "\n";
        return 1;
    }

  
    double youngModulus = 0., poissonRatio = 0.;
    int    maxEpoch     = 0;
    double minLoss      = 0.;
    bool   supervisedLearning = false;
    int64_t nrCtrlPts   = 0;
    int     degreeCfg   = 0;
    bool    runGsRefSim = false;
    int     degreeRef   = 0;

    std::vector<std::tuple<int,double,double,double>> forceSides, diriSides;
    std::vector<int>     tfbcSides;
    std::array<double,3> bodyForce{0.,0.,0.};

    try {
        youngModulus      = require(j, "material.young_modulus").get<double>();
        poissonRatio      = require(j, "material.poisson_ratio").get<double>();
        maxEpoch          = require(j, "simulation.max_epoch").get<int>();
        minLoss           = require(j, "simulation.min_loss").get<double>();
        supervisedLearning= require(j, "simulation.supervised_learning").get<bool>();
        nrCtrlPts         = require(j, "spline.nr_ctrl_pts").get<int64_t>();
        degreeCfg         = require(j, "spline.degree").get<int>();

        for (const auto& fs : require(j, "boundary_conditions.force_sides"))
            forceSides.emplace_back(fs[0].get<int>(), fs[1].get<double>(),
                                    fs[2].get<double>(), fs[3].get<double>());
        for (const auto& ds : require(j, "boundary_conditions.diri_sides"))
            diriSides.emplace_back(ds[0].get<int>(), ds[1].get<double>(),
                                   ds[2].get<double>(), ds[3].get<double>());
        tfbcSides = require(j, "boundary_conditions.tfbc_sides").get<std::vector<int>>();

        const auto& bf = require(j, "body_force");
        bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};

        if (j.contains("reference_simulation")) {
            runGsRefSim = require(j,"reference_simulation.run_gs_ref_sim").get<bool>();
            degreeRef   = require(j,"reference_simulation.degree_ref").get<int>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    // Lamé-Parameter
    const double lambda = (youngModulus * poissonRatio) /
                          ((1. + poissonRatio) * (1. - 2.*poissonRatio));
    const double mu     = youngModulus / (2. * (1. + poissonRatio));

    // Geometriemodus aus Config bestimmen
    GeometryMode geoMode = parseGeometryMode(j);
    std::cout << "[main] geometry.mode = "
              << (geoMode == GeometryMode::Xml ? "xml" : "parametric") << "\n";


    switch (degreeCfg) {
        case 2: return run<2>(repoRoot, XML_PATH, RESULT_PATH, geoMode, j,
                              lambda, mu, supervisedLearning, maxEpoch, minLoss,
                              bodyForce, tfbcSides, forceSides, diriSides,
                              nrCtrlPts, degreeRef, runGsRefSim,
                              youngModulus, poissonRatio);
        case 3: return run<3>(repoRoot, XML_PATH, RESULT_PATH, geoMode, j,
                              lambda, mu, supervisedLearning, maxEpoch, minLoss,
                              bodyForce, tfbcSides, forceSides, diriSides,
                              nrCtrlPts, degreeRef, runGsRefSim,
                              youngModulus, poissonRatio);
        case 4: return run<4>(repoRoot, XML_PATH, RESULT_PATH, geoMode, j,
                              lambda, mu, supervisedLearning, maxEpoch, minLoss,
                              bodyForce, tfbcSides, forceSides, diriSides,
                              nrCtrlPts, degreeRef, runGsRefSim,
                              youngModulus, poissonRatio);
        case 5: return run<5>(repoRoot, XML_PATH, RESULT_PATH, geoMode, j,
                              lambda, mu, supervisedLearning, maxEpoch, minLoss,
                              bodyForce, tfbcSides, forceSides, diriSides,
                              nrCtrlPts, degreeRef, runGsRefSim,
                              youngModulus, poissonRatio);
        case 6: return run<6>(repoRoot, XML_PATH, RESULT_PATH, geoMode, j,
                              lambda, mu, supervisedLearning, maxEpoch, minLoss,
                              bodyForce, tfbcSides, forceSides, diriSides,
                              nrCtrlPts, degreeRef, runGsRefSim,
                              youngModulus, poissonRatio);
        default:
            std::cerr << "Error: Invalid degree " << degreeCfg << " (2..6)\n";
            return 1;
    }

    iganet::finalize();
    return 0;
}
