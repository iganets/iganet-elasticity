/*
 * Example: optimized 2D single-patch linear elasticity.
 *
 * This version keeps the high-level example logic relatively sequential while
 * delegating the heavy PDE and post-processing work to the shared
 * linear_elasticity header. The intention is:
 *   - the control flow in this file should remain easy to follow,
 *   - the expensive spline evaluation details should stay centralized.
 */

#include "headers/lin_elasticity_net.hpp"
#include "headers/lin_elasticity_utils.hpp"

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
using iganet_elasticity::utils::config::require;
using iganet_elasticity::utils::paths::repo_root_from_build_exe;

namespace {

// The example currently supports either a purely parametric rectangular
// geometry or a spline geometry loaded from an XML file.
enum class GeometryMode2D {
    Xml,
    Parametric
};

GeometryMode2D parse_geometry_mode_2d(const nlohmann::json& j) {
    // If no geometry mode is specified, use the most lightweight setup:
    // a parametric rectangle generated inside the example.
    if (!j.contains("geometry") || !j["geometry"].contains("mode")) {
        return GeometryMode2D::Parametric;
    }

    const auto mode = j["geometry"]["mode"].get<std::string>();
    if (mode == "xml") {
        return GeometryMode2D::Xml;
    }
    if (mode == "parametric") {
        return GeometryMode2D::Parametric;
    }

    throw std::runtime_error(
        "Unknown geometry.mode '" + mode + "'. Allowed: 'xml', 'parametric'.");
}

struct XmlGeometryData2D {
    pugi::xml_document doc;
    std::array<std::vector<double>, 2> knotVectors;
    int64_t nCtrlPts = 0;
};

// Read just enough XML data to build the spline space locally:
// the full control point geometry is loaded later via from_xml().
XmlGeometryData2D load_xml_knot_vectors_2d(const std::filesystem::path& xmlPath,
                                           const char* geomId,
                                           int degree) {
    XmlGeometryData2D result;

    if (!result.doc.load_file(xmlPath.c_str())) {
        throw std::runtime_error("Could not load geometry from XML: " + xmlPath.string());
    }

    auto geomNode = result.doc.child("xml")
                        .find_child_by_attribute("Geometry", "id", geomId);
    if (!geomNode) {
        throw std::runtime_error(
            std::string("Geometry id=") + geomId + " not found in XML: " + xmlPath.string());
    }

    int kvIdx = 0;
    for (auto& basisNode : geomNode.child("Basis").children("Basis")) {
        if (kvIdx >= 2) {
            break;
        }

        std::istringstream iss(basisNode.child("KnotVector").child_value());
        double value = 0.0;
        while (iss >> value) {
            result.knotVectors[kvIdx].push_back(value);
        }

        if (result.knotVectors[kvIdx].empty()) {
            throw std::runtime_error(
                "Empty knot vector " + std::to_string(kvIdx) + " in geometry id=" +
                std::string(geomId));
        }

        ++kvIdx;
    }

    if (kvIdx != 2) {
        throw std::runtime_error(
            "Expected 2 knot vectors, got " + std::to_string(kvIdx) + " in XML geometry.");
    }

    result.nCtrlPts = static_cast<int64_t>(result.knotVectors[0].size()) - degree - 1;
    return result;
}

template <int DEGREE, typename GeometrySpline, typename VariableSpline,
          typename GeometrySpaceSpec, typename VariableSpaceSpec>
int run(const std::filesystem::path& repoRoot,
        const std::filesystem::path& resultJsonPath,
        GeometryMode2D geoMode,
        const nlohmann::json& j,
        double lambda,
        double mu,
        bool supervisedLearning,
        int maxEpoch,
        double minLoss,
        std::pair<double, double> bodyForce,
        std::vector<int> tfbcSides,
        std::vector<std::tuple<int, double, double>> forceSides,
        std::vector<std::tuple<int, double, double>> diriSides,
        int64_t nrCtrlPts,
        GeometrySpaceSpec&& geometrySpaceSpec,
        VariableSpaceSpec&& variableSpaceSpec,
        const XmlGeometryData2D* xmlDataPtr = nullptr) {
    // This helper performs the actual simulation for one fixed compile-time
    // spline degree. main() only selects the correct instantiation.
    using real_t = double;
    using geometry_t = iganet::S<GeometrySpline>;
    using variable_t = iganet::S<VariableSpline>;
    using net_t = linear_elasticity<torch::optim::LBFGS, geometry_t, variable_t>;

    const torch::Device computeDevice =
        torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                    : torch::Device(torch::kCPU);
    const auto options = iganet::Options<real_t>{}.device(computeDevice);

    net_t net(
        lambda, mu, supervisedLearning, maxEpoch, minLoss,
        bodyForce, tfbcSides, forceSides, diriSides,
        nrCtrlPts, resultJsonPath.string(),
        {25, 25},
        {{iganet::activation::sigmoid},
         {iganet::activation::sigmoid},
         {iganet::activation::none}},
        std::forward<GeometrySpaceSpec>(geometrySpaceSpec),
        std::forward<VariableSpaceSpec>(variableSpaceSpec),
        iganet::init::greville,
        iganet::IgANetOptions{},
        options);
    // The shared linear_elasticity header handles the actual PDE residual,
    // boundary losses, optimizer interaction, and post-processing. This file
    // stays focused on assembling the right spaces and inputs.

    if (geoMode == GeometryMode2D::Xml) {
        if (!xmlDataPtr) {
            std::cerr << "XML geometry requested without loaded XML data.\n";
            return 1;
        }
        // XML mode imports the geometry map exactly from file.
        net.template input<0>().from_xml(xmlDataPtr->doc, 0);
    } else {
        // Parametric mode creates a simple tensor-product geometry directly in
        // code by placing control points on Greville abscissae.
        std::array<double, 2> origin{0.0, 0.0};
        std::array<double, 2> scale{1.0, 1.0};
        if (j.contains("geometry")) {
            const auto& gj = j["geometry"];
            if (gj.contains("origin")) {
                origin = {gj["origin"][0].get<double>(), gj["origin"][1].get<double>()};
            }
            if (gj.contains("scale")) {
                scale = {gj["scale"][0].get<double>(), gj["scale"][1].get<double>()};
            }
        }

        const auto knotVector = makeUniformKnotVector(static_cast<int>(nrCtrlPts), DEGREE);
        const auto greville = computeGrevilleAbscissae(
            knotVector, DEGREE, static_cast<int>(nrCtrlPts));
        const int64_t nPts = nrCtrlPts * nrCtrlPts;

        // Internal spline tensors use component-major storage:
        // [all x coordinates | all y coordinates].
        auto geomTensor = torch::zeros({2 * nPts}, net.template input<0>().as_tensor().options());
        int64_t idx = 0;
        for (int64_t jv = 0; jv < nrCtrlPts; ++jv) {
            for (int64_t i = 0; i < nrCtrlPts; ++i, ++idx) {
                geomTensor[idx] = origin[0] + scale[0] * greville[static_cast<std::size_t>(i)];
                geomTensor[idx + nPts] =
                    origin[1] + scale[1] * greville[static_cast<std::size_t>(jv)];
            }
        }
        net.template input<0>().from_tensor(geomTensor);
    }

    auto setComponent = [&](auto& boundary, double value, iganet::short_t comp) {
        // Assign one scalar component on one boundary side of the reference
        // field. The shared header later uses that field for Dirichlet terms.
        boundary.template transform<1>(
            [value](const std::array<real_t, 1>&) -> std::array<real_t, 1> {
                return {value};
            },
            std::array<iganet::short_t, 1>{comp});
    };

    for (const auto& side : diriSides) {
        // Store the prescribed displacement values in the reference field. The
        // shared linear_elasticity implementation later compares against this
        // field when building the Dirichlet loss term.
        const int sideNr = std::get<0>(side);
        const double xDisp = std::get<1>(side);
        const double yDisp = std::get<2>(side);

        switch (sideNr) {
            case 1:
                setComponent(net.ref().boundary().template side<1>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<1>(), yDisp, 1);
                break;
            case 2:
                setComponent(net.ref().boundary().template side<2>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<2>(), yDisp, 1);
                break;
            case 3:
                setComponent(net.ref().boundary().template side<3>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<3>(), yDisp, 1);
                break;
            case 4:
                setComponent(net.ref().boundary().template side<4>(), xDisp, 0);
                setComponent(net.ref().boundary().template side<4>(), yDisp, 1);
                break;
            default:
                std::cerr << "Invalid Dirichlet side " << sideNr << "\n";
                return 1;
        }
    }

    torch::Tensor ctrlPtsCoeffs = net.template input<0>().as_tensor().slice(0, 0, nrCtrlPts);
    // Export one representative coefficient line so the result JSON records the
    // actual spline discretization used by this run.
    nlohmann::json ctrlPtsCoeffs_j = nlohmann::json::array();
    for (int64_t i = 0; i < nrCtrlPts; ++i) {
        ctrlPtsCoeffs_j.push_back({ctrlPtsCoeffs[i].item<double>()});
    }
    net.appendToJsonFile("net_ctrlPtsCoeffs", ctrlPtsCoeffs_j);

    net.options().max_epoch(maxEpoch);
    net.options().min_loss(minLoss);

    auto t1 = std::chrono::high_resolution_clock::now();
    net.train();
    auto t2 = std::chrono::high_resolution_clock::now();
    iganet::Log(iganet::log::info)
        << "Training took "
        << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
        << " seconds\n";

    const auto geometryTensor = net.template input<0>().as_tensor();
    const auto displacementTensor = net.template output<0>().as_tensor();
    // Reassemble one row per control point from the internal component-major
    // tensor storage used by the spline objects.
    torch::Tensor netCtrlPts = torch::zeros({nrCtrlPts * nrCtrlPts, 2}, geometryTensor.options());
    torch::Tensor netDisplacements =
        torch::zeros({nrCtrlPts * nrCtrlPts, 2}, displacementTensor.options());

    for (int64_t i = 0; i < nrCtrlPts * nrCtrlPts; ++i) {
        netCtrlPts[i][0] = geometryTensor[i].template item<double>();
        netCtrlPts[i][1] =
            geometryTensor[i + nrCtrlPts * nrCtrlPts].template item<double>();
        netDisplacements[i][0] = displacementTensor[i].template item<double>();
        netDisplacements[i][1] =
            displacementTensor[i + nrCtrlPts * nrCtrlPts].template item<double>();
    }

    const auto displacedNetCtrlPts = netCtrlPts + netDisplacements;
    // The show script rebuilds the final spline from deformed control points
    // plus the spline degree, so those are the key export quantities here.
    nlohmann::json displacedNetCtrlPts_j = nlohmann::json::array();
    for (int64_t i = 0; i < displacedNetCtrlPts.size(0); ++i) {
        displacedNetCtrlPts_j.push_back(
            {displacedNetCtrlPts[i][0].item<double>(), displacedNetCtrlPts[i][1].item<double>()});
    }

    net.appendToJsonFile("net_CtrlPts", displacedNetCtrlPts_j);
    net.appendToJsonFile("net_Degree", DEGREE);

    return 0;
}

} // namespace

int main() {
    // main() is intentionally written as a sequence of steps:
    // configuration -> geometry selection -> degree dispatch -> run().
    iganet::init();
    iganet::verbose(std::cout);

    std::filesystem::path repoRoot;
    try {
        repoRoot = repo_root_from_build_exe();
    } catch (const std::exception& e) {
        std::cerr << "Could not determine repo root: " << e.what() << "\n";
        return 1;
    }

    const auto CONFIG_PATH = repoRoot / "src" / "examples2D" / "singlePatch" /
                             "sim_config_2D_single_patch.json";
    const auto RESULT_PATH =
        repoRoot / "results" / "result_iganet_lin_elasticity_2D_optimized.json";

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

    double youngModulus = 0.0;
    double poissonRatio = 0.0;
    int maxEpoch = 0;
    double minLoss = 0.0;
    bool supervisedLearning = false;
    int64_t nrCtrlPts = 0;
    int degreeCfg = 0;
    optimizer_config_t optimizerCfg;
    bool runCollRefSim = false;

    std::vector<std::tuple<int, double, double>> forceSides;
    std::vector<std::tuple<int, double, double>> diriSides;
    std::vector<int> tfbcSides;
    std::pair<double, double> bodyForce{0.0, 0.0};

    try {
        // Read the config in the same conceptual order as the simulation:
        // material -> optimization -> spline space -> boundary conditions.
        youngModulus = require(j, "material.young_modulus").get<double>();
        poissonRatio = require(j, "material.poisson_ratio").get<double>();
        maxEpoch = require(j, "simulation.max_epoch").get<int>();
        minLoss = require(j, "simulation.min_loss").get<double>();
        supervisedLearning = require(j, "simulation.supervised_learning").get<bool>();
        optimizerCfg = iganet_elasticity::utils::config::load_optimizer_config(j);

        const auto solutionSplineCfg =
            iganet_elasticity::utils::config::load_solution_spline_config(j);
        nrCtrlPts = solutionSplineCfg.nr_ctrl_pts;
        degreeCfg = solutionSplineCfg.degree;

        const auto patchCfg =
            iganet_elasticity::utils::config::load_single_patch_config_2d(j);
        for (const auto& bc : patchCfg.force_sides) {
            forceSides.emplace_back(bc.side, bc.x, bc.y);
        }
        for (const auto& bc : patchCfg.diri_sides) {
            diriSides.emplace_back(bc.side, bc.x, bc.y);
        }
        tfbcSides = patchCfg.tfbc_sides;
        bodyForce = {patchCfg.body_force[0], patchCfg.body_force[1]};

        if (j.contains("reference_simulation")) {
            runCollRefSim = require(j, "reference_simulation.run_coll_ref_sim").get<bool>();
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
        // The Python reference run writes a standard-collocation baseline into
        // the same result JSON as the IgANet solve.
        const int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "ERROR: python reference run failed. system() returned " << ret << "\n";
            return 1;
        }
    }

    if (optimizerCfg.type != optimizer_type_t::lbfgs) {
        std::cerr << "2D optimized example currently supports only LBFGS.\n";
        return 1;
    }

    const auto geoMode = parse_geometry_mode_2d(j);

    std::filesystem::path xmlPath;
    std::string xmlGeometryId = "100";
    std::optional<XmlGeometryData2D> xmlData;
    if (geoMode == GeometryMode2D::Xml) {
        try {
            // Imported knot vectors define how many control points the geometry
            // actually has, so nrCtrlPts is updated from the XML data here.
            const auto& gj = require(j, "geometry");
            xmlPath = std::filesystem::path(require(gj, "xml_path").get<std::string>());
            if (xmlPath.is_relative()) {
                xmlPath = repoRoot / xmlPath;
            }
            if (gj.contains("xml_id")) {
                xmlGeometryId = gj["xml_id"].get<std::string>();
            }
            xmlData.emplace(load_xml_knot_vectors_2d(xmlPath, xmlGeometryId.c_str(), degreeCfg));
            nrCtrlPts = xmlData->nCtrlPts;
        } catch (const std::exception& e) {
            std::cerr << "XML config error: " << e.what() << "\n";
            return 1;
        }
    }

    const double lambda =
        (youngModulus * poissonRatio) / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));
    const double mu = youngModulus / (2.0 * (1.0 + poissonRatio));

    const auto run_parametric_dispatch = [&]<int DEGREE, typename SplineType>() -> int {
        // Uniform splines are sufficient when the geometry is generated
        // internally from degree and number of control points.
        return run<DEGREE, SplineType, SplineType>(
            repoRoot, RESULT_PATH, geoMode, j,
            lambda, mu, supervisedLearning, maxEpoch, minLoss,
            bodyForce, tfbcSides, forceSides, diriSides, nrCtrlPts,
            std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts)),
            std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts)));
    };

    const auto run_xml_dispatch = [&]<int DEGREE, typename SplineType>() -> int {
        // Imported XML geometries bring their own knot vectors, so we dispatch
        // to non-uniform spline types in that case.
        return run<DEGREE, SplineType, SplineType>(
            repoRoot, RESULT_PATH, geoMode, j,
            lambda, mu, supervisedLearning, maxEpoch, minLoss,
            bodyForce, tfbcSides, forceSides, diriSides, nrCtrlPts,
            std::make_tuple(std::make_tuple(xmlData->knotVectors)),
            std::make_tuple(std::make_tuple(xmlData->knotVectors)),
            &*xmlData);
    };

    switch (degreeCfg) {
        // This is a type dispatch: each case selects a distinct compile-time
        // spline type with the requested degree.
        case 2:
            if (geoMode == GeometryMode2D::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 2, 2, 2>;
                return run_xml_dispatch.template operator()<2, spline_t>();
            } else {
                using spline_t = iganet::UniformBSpline<double, 2, 2, 2>;
                return run_parametric_dispatch.template operator()<2, spline_t>();
            }
        case 3:
            if (geoMode == GeometryMode2D::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 2, 3, 3>;
                return run_xml_dispatch.template operator()<3, spline_t>();
            } else {
                using spline_t = iganet::UniformBSpline<double, 2, 3, 3>;
                return run_parametric_dispatch.template operator()<3, spline_t>();
            }
        case 4:
            if (geoMode == GeometryMode2D::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 2, 4, 4>;
                return run_xml_dispatch.template operator()<4, spline_t>();
            } else {
                using spline_t = iganet::UniformBSpline<double, 2, 4, 4>;
                return run_parametric_dispatch.template operator()<4, spline_t>();
            }
        case 5:
            if (geoMode == GeometryMode2D::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 2, 5, 5>;
                return run_xml_dispatch.template operator()<5, spline_t>();
            } else {
                using spline_t = iganet::UniformBSpline<double, 2, 5, 5>;
                return run_parametric_dispatch.template operator()<5, spline_t>();
            }
        case 6:
            if (geoMode == GeometryMode2D::Xml) {
                using spline_t = iganet::NonUniformBSpline<double, 2, 6, 6>;
                return run_xml_dispatch.template operator()<6, spline_t>();
            } else {
                using spline_t = iganet::UniformBSpline<double, 2, 6, 6>;
                return run_parametric_dispatch.template operator()<6, spline_t>();
            }
        default:
            std::cerr << "Error: Invalid degree " << degreeCfg << " (2..6)\n";
            return 1;
    }
}
