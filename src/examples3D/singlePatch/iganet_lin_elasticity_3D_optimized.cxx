/*
 * Example: optimized 3D single-patch linear elasticity.
 *
 * This file mirrors the structure of the optimized 2D example:
 *   - keep the example control flow here,
 *   - delegate the PDE-specific implementation details to the shared header,
 *   - support both parametric and XML-based geometry setup.
 */

#include "headers/lin_elasticity_net.hpp"

#include <utils/config.hpp>
#include <utils/paths.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace iganet::literals;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;
using iganet_elasticity::utils::config::require;

// run() executes one degree-specialized simulation instance.
template <int DEGREE, typename GeometrySpline, typename VariableSpline,
          typename GeometrySpaceSpec, typename VariableSpaceSpec>
int run(
    const std::filesystem::path&                        repoRoot,
    const std::filesystem::path&                        xmlPath,
    const std::filesystem::path&                        resultJsonPath,
    torch::Device                                       computeDevice,
    GeometryMode                                        geoMode,
    const nlohmann::json&                               j,
    double                                              lambda, 
    double                                              mu,
    bool                                                supervisedLearning,
    int                                                 maxEpoch, 
    double                                              minLoss,
    std::array<double,3>                                bodyForce,
    std::vector<int>                                    tfbcSides,
    std::vector<std::tuple<int,double,double,double>>   forceSides,
    std::vector<std::tuple<int,double,double,double>>   diriSides,
    int64_t                                             nrCtrlPts,
    int                                                 degreeRef,
    bool                                                runGsRefSim,
    double                                              youngModulus,
    double                                              poissonRatio,
    GeometrySpaceSpec&&                                 geometrySpaceSpec,
    VariableSpaceSpec&&                                 variableSpaceSpec,
    const XmlGeometryData*                              xmlDataPtr = nullptr)
{
    // The degree is compile-time because the spline type is compile-time.
    using real_t      = double;
    using optimizer_t = torch::optim::LBFGS;
    using geometry_t  = iganet::S<GeometrySpline>;
    using variable_t  = iganet::S<VariableSpline>;
    using net_t       = linear_elasticity<optimizer_t, geometry_t, variable_t>;

    const std::string jsonPath = resultJsonPath.string();

    const auto netOptions = iganet::Options<real_t>{}.device(computeDevice);

    net_t net(
        lambda, mu, supervisedLearning, maxEpoch, minLoss,
        bodyForce, tfbcSides, forceSides, diriSides,
        nrCtrlPts, jsonPath,
        {25, 30},
        {{iganet::activation::sigmoid},
         {iganet::activation::sigmoid},
         {iganet::activation::none}},
        std::forward<GeometrySpaceSpec>(geometrySpaceSpec),
        std::forward<VariableSpaceSpec>(variableSpaceSpec),
        iganet::init::greville,
        iganet::IgANetOptions{},
        netOptions);
    // The shared linear_elasticity header performs the actual mechanics,
    // collocation handling, and training. This file mainly prepares geometry,
    // space types, and result export.

    if (geoMode == GeometryMode::Xml) {

        // XML mode imports the geometry map from the spline file.
        if (!xmlDataPtr) {
            std::cerr << "Error: XML geometry requested without loaded XML data.\n";
            return 1;
        }
        net.template input<0>().from_xml(xmlDataPtr->doc, 0);

    } else {
        // Parametric mode creates a box-like spline geometry directly from
        // Greville points. This is convenient for controlled test problems.

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

        const auto knotVector = makeUniformKnotVector(static_cast<int>(nrCtrlPts), DEGREE);
        const auto greville   = computeGrevilleAbscissae(knotVector, DEGREE, static_cast<int>(nrCtrlPts));
        const int64_t nPts    = nrCtrlPts * nrCtrlPts * nrCtrlPts;

        // Internal spline tensors use component-major storage:
        // [all x coordinates | all y coordinates | all z coordinates].
        auto geomTensor = torch::zeros({3 * nPts}, net.template input<0>().as_tensor().options());

        int64_t idx = 0;
        for (int64_t k = 0; k < nrCtrlPts; ++k) {
            for (int64_t jv = 0; jv < nrCtrlPts; ++jv) {
                for (int64_t i = 0; i < nrCtrlPts; ++i, ++idx) {
                    geomTensor[idx]          = origin[0] + scale[0] * greville[static_cast<std::size_t>(i)];
                    geomTensor[idx + nPts]   = origin[1] + scale[1] * greville[static_cast<std::size_t>(jv)];
                    geomTensor[idx + 2*nPts] = origin[2] + scale[2] * greville[static_cast<std::size_t>(k)];
                }
            }
        }

        net.template input<0>().from_tensor(geomTensor);

    }

  
    auto setComponent = [&](auto& boundary, double value, iganet::short_t comp) {
        // Assign one scalar component on one reference boundary side.
        boundary.template transform<1>(
            [value](const std::array<real_t,2>&) -> std::array<real_t,1> {
                return {value};
            },
            std::array<iganet::short_t,1>{comp});
    };

    for (const auto& side : diriSides) {
        // The reference field stores the prescribed boundary displacement and
        // is later used by the shared loss implementation.
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
    // In the 3D shared header, all collocation-point caches are prepared
    // explicitly before the training loop starts.
    net.initialize_problem_data();

    auto t1 = std::chrono::high_resolution_clock::now();
    net.train();
    auto t2 = std::chrono::high_resolution_clock::now();
    iganet::Log(iganet::log::info)
        << "Training took "
        << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
        << " seconds\n";

 
    net.PostProc();
    // PostProc() writes stresses, residuals, and displaced collocation points.
    // We additionally export deformed control points below because the final
    // geometry viewer reconstructs the spline from them.

    torch::Tensor geomTensor = net.template input<0>().as_tensor();
    torch::Tensor dispTensor = net.template output<0>().as_tensor();
    int64_t totalSize = geomTensor.size(0);

    if (totalSize % 3 != 0) {
        std::cerr << "Error: Geometry tensor size not divisible by 3 (size="
                  << totalSize << ")\n";
        return 1;
    }
    if (dispTensor.size(0) != totalSize) {
        std::cerr << "Error: Displacement tensor size does not match geometry tensor size.\n";
        return 1;
    }
    int64_t nCtrlPtsXml = totalSize / 3;

    torch::Tensor netCtrlPts = torch::zeros({nCtrlPtsXml, 3}, geomTensor.options());
    torch::Tensor netDisplacements = torch::zeros({nCtrlPtsXml, 3}, geomTensor.options());
    // Convert flat component-major tensors back into one row per control point.
    for (int64_t i = 0; i < nCtrlPtsXml; ++i) {
        netDisplacements[i][0] = dispTensor[i].item<double>();
        netDisplacements[i][1] = dispTensor[i + nCtrlPtsXml].item<double>();
        netDisplacements[i][2] = dispTensor[i + 2*nCtrlPtsXml].item<double>();

        netCtrlPts[i][0] = geomTensor[i].item<double>() + netDisplacements[i][0].item<double>();
        netCtrlPts[i][1] = geomTensor[i + nCtrlPtsXml].item<double>() + netDisplacements[i][1].item<double>();
        netCtrlPts[i][2] = geomTensor[i + 2*nCtrlPtsXml].item<double>() + netDisplacements[i][2].item<double>();
    }

    nlohmann::json netCtrlPts_j = nlohmann::json::array();
    nlohmann::json netDispl_j   = nlohmann::json::array();
    for (int64_t i = 0; i < netCtrlPts.size(0); ++i)
        netCtrlPts_j.push_back({netCtrlPts[i][0].item<double>(),
                                 netCtrlPts[i][1].item<double>(),
                                 netCtrlPts[i][2].item<double>()});
    for (int64_t i = 0; i < netDisplacements.size(0); ++i)
        netDispl_j.push_back({netDisplacements[i][0].item<double>(),
                              netDisplacements[i][1].item<double>(),
                              netDisplacements[i][2].item<double>()});

    net.appendToJsonFile("net_CtrlPts",          netCtrlPts_j);
    net.appendToJsonFile("net_Displacements",    netDispl_j);
    net.appendToJsonFile("net_nrCtrlPts",        nCtrlPtsXml);
    net.appendToJsonFile("net_Degree",           DEGREE);

    // Optional: GISMO reference simulation
    //
    // This branch is only a comparison/debugging aid. The IgANet solve itself
    // does not depend on it.

#ifdef IGANET_WITH_GISMO
    if (runGsRefSim) {
        // GISMO expects 2D tuples for DIRI_SIDES / FORCE_SIDES
        // Conversion: (side,ux,uy,uz) -> (side,ux,uy)  [only x/y are used]
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
    // Keep the high-level flow easy to follow: configuration, geometry mode,
    // degree dispatch, training, export.
    iganet::init();
    iganet::verbose(std::cout);

    // get repo root from build exe path
    std::filesystem::path repoRoot;
    try {
        repoRoot = repo_root_from_build_exe();
    } catch (const std::exception& e) {
        std::cerr << "Could not determine repo root: " << e.what() << "\n";
        return 1;
    }

    const auto CONFIG_PATH  = repoRoot / "src" / "examples3D" / "singlePatch" /
                              "sim_config_3D_single_patch.json";
    const auto RESULT_PATH  = repoRoot / "results" / "result_iganet_lin_elasticity_3D_optimized.json";

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


    double  youngModulus        = 0.;
    double  poissonRatio        = 0.;
    int     maxEpoch            = 0;
    double  minLoss             = 0.;
    bool    supervisedLearning  = false;
    int64_t nrCtrlPts           = 0;
    int     degreeCfg           = 0;
    bool    runGsRefSim         = false;
    bool    runCollRefSim       = false;
    int     degreeRef           = 0;

    std::vector<std::tuple<int,double,double,double>> forceSides, diriSides;
    std::vector<int>     tfbcSides;
    std::array<double,3> bodyForce{0.,0.,0.};

    try {
        youngModulus      = require(j, "material.young_modulus").get<double>();
        poissonRatio      = require(j, "material.poisson_ratio").get<double>();
        maxEpoch          = require(j, "simulation.max_epoch").get<int>();
        minLoss           = require(j, "simulation.min_loss").get<double>();
        supervisedLearning= require(j, "simulation.supervised_learning").get<bool>();
        const auto solutionSplineCfg =
            iganet_elasticity::utils::config::load_solution_spline_config(j);
        nrCtrlPts         = solutionSplineCfg.nr_ctrl_pts;
        degreeCfg         = solutionSplineCfg.degree;

        const auto& singlePatchCfg = require(j, "single_patch_3D");

        for (const auto& fs : require(singlePatchCfg, "boundary_conditions.force_sides"))
            forceSides.emplace_back(fs[0].get<int>(), fs[1].get<double>(),
                                    fs[2].get<double>(), fs[3].get<double>());
        for (const auto& ds : require(singlePatchCfg, "boundary_conditions.diri_sides"))
            diriSides.emplace_back(ds[0].get<int>(), ds[1].get<double>(),
                                   ds[2].get<double>(), ds[3].get<double>());
        tfbcSides = require(singlePatchCfg, "boundary_conditions.tfbc_sides").get<std::vector<int>>();

        const auto& bf = require(singlePatchCfg, "body_force");
        bodyForce = {bf[0].get<double>(), bf[1].get<double>(), bf[2].get<double>()};

        if (j.contains("reference_simulation")) {
            runGsRefSim   = require(j,"reference_simulation.run_gs_ref_sim").get<bool>();
            runCollRefSim = require(j,"reference_simulation.run_coll_ref_sim").get<bool>();
            degreeRef     = require(j,"reference_simulation.degree_ref").get<int>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    if (runCollRefSim) {
        const std::string cmd =
            "cd \"" + repoRoot.string() +
            "\" && python3 -m std_collocation_python.run_std_coll \"" +
            CONFIG_PATH.string() + "\" \"" + RESULT_PATH.string() + "\"";
        // As in 2D, the external Python run creates a standard-collocation
        // baseline in the same result JSON as the IgANet solve.
        const int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "ERROR: python reference run failed. system() returned "
                      << ret << "\n";
            return 1;
        }
    }

    const GeometryMode geoMode = parseGeometryMode(j);
    const torch::Device computeDevice =
        torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                    : torch::Device(torch::kCPU);

    std::filesystem::path xmlPath;
    std::string xmlGeometryId;
    if (geoMode == GeometryMode::Xml) {
        try {
            const auto& gj = require(j, "geometry");
            std::string xmlPathCfg;

            if (gj.contains("xml_path")) {
                xmlPathCfg = gj["xml_path"].get<std::string>();
            } else if (gj.contains("multipatch_xml_path")) {
                xmlPathCfg = gj["multipatch_xml_path"].get<std::string>();
            } else {
                throw std::runtime_error(
                    "Missing required config key: geometry.xml_path or geometry.multipatch_xml_path");
            }

            if (gj.contains("xml_id")) {
                xmlGeometryId = gj["xml_id"].get<std::string>();
            } else if (gj.contains("multipatch_id")) {
                xmlGeometryId = std::to_string(gj["multipatch_id"].get<int>());
            } else {
                throw std::runtime_error(
                    "Missing required config key: geometry.xml_id or geometry.multipatch_id");
            }

            xmlPath = std::filesystem::path(xmlPathCfg);
            if (xmlPath.is_relative()) {
                xmlPath = repoRoot / xmlPath;
            }
        } catch (const std::exception& e) {
            std::cerr << "Config error: " << e.what() << "\n";
            return 1;
        }
    }

    // Lamé parameters
    const double lambda = (youngModulus * poissonRatio) /
                          ((1. + poissonRatio) * (1. - 2.*poissonRatio));
    const double mu     = youngModulus / (2. * (1. + poissonRatio));

    std::optional<XmlGeometryData> xmlData;
    if (geoMode == GeometryMode::Xml) {
        xmlData.emplace(loadXmlKnotVectors(xmlPath, xmlGeometryId.c_str(), degreeCfg));
        nrCtrlPts = xmlData->nCtrlPts;
    } else if (j.contains("geometry") && j["geometry"].contains("nr_ctrl_pts")) {
        nrCtrlPts = j["geometry"]["nr_ctrl_pts"].get<int64_t>();
    }


    switch (degreeCfg) {
        case 2:
            if (geoMode == GeometryMode::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 3, 2, 2, 2>;
                return run<2, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    &*xmlData);
            } else {
                using spline_t = iganet::UniformBSpline<double, 3, 2, 2, 2>;
                return run<2, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)),
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)));
            }
        case 3:
            if (geoMode == GeometryMode::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 3, 3, 3, 3>;
                return run<3, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    &*xmlData);
            } else {
                using spline_t = iganet::UniformBSpline<double, 3, 3, 3, 3>;
                return run<3, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)),
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)));
            }
        case 4:
            if (geoMode == GeometryMode::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 3, 4, 4, 4>;
                return run<4, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    &*xmlData);
            } else {
                using spline_t = iganet::UniformBSpline<double, 3, 4, 4, 4>;
                return run<4, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)),
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)));
            }
        case 5:
            if (geoMode == GeometryMode::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 3, 5, 5, 5>;
                return run<5, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    &*xmlData);
            } else {
                using spline_t = iganet::UniformBSpline<double, 3, 5, 5, 5>;
                return run<5, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)),
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)));
            }
        case 6:
            if (geoMode == GeometryMode::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 3, 6, 6, 6>;
                return run<6, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    std::make_tuple(std::make_tuple(xmlData->knotVectors)),
                    &*xmlData);
            } else {
                using spline_t = iganet::UniformBSpline<double, 3, 6, 6, 6>;
                return run<6, spline_t, spline_t>(
                    repoRoot, xmlPath, RESULT_PATH, computeDevice, geoMode, j,
                    lambda, mu, supervisedLearning, maxEpoch, minLoss,
                    bodyForce, tfbcSides, forceSides, diriSides,
                    nrCtrlPts, degreeRef, runGsRefSim, youngModulus, poissonRatio,
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)),
                    std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts, nrCtrlPts)));
            }
        default:
            std::cerr << "Error: Invalid degree " << degreeCfg << " (2..6)\n";
            return 1;
    }

    iganet::finalize();
    return 0;
}
