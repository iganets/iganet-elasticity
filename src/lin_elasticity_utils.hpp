#pragma once

// =============================================================================
//  Hilfsfunktionen und Datentypen ohne iganet-Abhängigkeit
//  Wird von lin_elasticity_net.hpp und iganet_lin_elasticity_3D.cxx eingebunden
//
//  Wichtigste Funktionen:
//    1. GeometryMode          – wählt XML- oder parametrische Geometrie
//    2. appendToJsonFile()    – Schreibt einen Key sofort in eine JSON-Datei
//    3. loadDisplacements()   – Liest "stdCollDisplacement" als (N,3)-Tensor
//    4. loadXmlKnotVectors()  – Liest Knotenvektoren aus gismo-XML
//    5. makeUniformKnotVector()– Erzeugt uniformen, offenen B-Spline-KV
//    6. computeGrevilleAbscissae() – Greville-Abszissen eines B-Spline-Raums
// =============================================================================

#include <pugixml.hpp>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class GeometryMode {
    Xml,        ///< Knotenvektoren + Kontrollpunkte aus gismo-XML-Datei
    Parametric  ///< Uniformer Einheitswürfel, definiert über sim_config.json
};

/// Liest "geometry.mode" aus einem JSON-Objekt.
/// Wirft std::runtime_error wenn der Wert unbekannt ist.
inline GeometryMode parseGeometryMode(const nlohmann::json& j) {
    // Rückwärtskompatibel: fehlt das Feld, wird XML angenommen
    if (!j.contains("geometry") || !j["geometry"].contains("mode")) {
        return GeometryMode::Xml;
    }
    const std::string mode = j["geometry"]["mode"].get<std::string>();
    if (mode == "xml")        return GeometryMode::Xml;
    if (mode == "parametric") return GeometryMode::Parametric;
    throw std::runtime_error(
        "Unbekannter geometry.mode: '" + mode + "'. Erlaubt: 'xml', 'parametric'.");
}


inline void appendToJsonFile(const std::string&    jsonPath,
                              const std::string&    key,
                              const nlohmann::json& data)
{
    
    nlohmann::json jsonData;
    try {
        std::ifstream in(jsonPath);
        if (in.is_open()) {
            in >> jsonData;
        } else {
            std::cerr << "Warning: Could not open file for reading: "
                      << jsonPath << "\n";
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
        std::ofstream out(jsonPath);
        if (out.is_open()) {
            out << jsonData.dump(1);
        } else {
            std::cerr << "Error: Could not open file for writing: "
                      << jsonPath << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing JSON file: " << jsonPath
                  << ". Exception: " << e.what() << "\n";
    }
}

//  loadDisplacements
//
//  Liest "stdCollDisplacement" aus einer JSON-Datei und gibt einen
//  CPU-Double-Tensor der Form (N, 3) zurück.
//
//  Erwartet JSON-Format:
//    { "stdCollDisplacement": [[ux0,uy0,uz0], [ux1,uy1,uz1], ...] }

inline torch::Tensor loadDisplacements(const std::string& jsonPath) {
    auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + jsonPath);
    }

    nlohmann::json jsonData;
    file >> jsonData;
    file.close();

    auto& arr             = jsonData["stdCollDisplacement"];
    int   nrStdCollCtrlPts = static_cast<int>(arr.size());

    torch::Tensor result = torch::empty({nrStdCollCtrlPts, 3}, options);
    for (int i = 0; i < nrStdCollCtrlPts; ++i) {
        result[i][0] = arr[i][0].get<double>();
        result[i][1] = arr[i][1].get<double>();
        result[i][2] = arr[i][2].get<double>();
    }
    return result;
}


// loadXmlKnotVectors
//
//  Liest die drei B-Spline-Knotenvektoren (u/v/w) aus einer XML-Datei.

struct XmlGeometryData {
    pugi::xml_document            doc;          ///< geladenes XML-Dokument (für from_xml)
    std::array<std::vector<double>, 3> knotVectors; ///< KV[0]=u, KV[1]=v, KV[2]=w
    int64_t                       nCtrlPts = 0; ///< Kontrollpunkte pro Richtung
};

inline XmlGeometryData loadXmlKnotVectors(const std::filesystem::path& xmlPath,
                                           const char* geomId = "100",
                                           int         degree = 2)
{
    XmlGeometryData result;

    // XML-Datei laden
    if (!result.doc.load_file(xmlPath.c_str())) {
        throw std::runtime_error(
            "Could not load geometry from XML: " + xmlPath.string());
    }

    auto geomNode = result.doc.child("xml")
                              .find_child_by_attribute("Geometry", "id", geomId);
    if (!geomNode) {
        throw std::runtime_error(
            std::string("Geometry id=") + geomId + " not found in XML: "
            + xmlPath.string());
    }

    int kvIdx = 0;
    for (auto& basisNode : geomNode.child("Basis").children("Basis")) {
        if (kvIdx >= 3) break;

        std::istringstream iss(basisNode.child("KnotVector").child_value());
        double v;
        while (iss >> v) result.knotVectors[kvIdx].push_back(v);

        if (result.knotVectors[kvIdx].empty()) {
            throw std::runtime_error(
                "Empty knot vector " + std::to_string(kvIdx)
                + " in geometry id=" + geomId);
        }
        ++kvIdx;
    }

    if (kvIdx != 3) {
        throw std::runtime_error(
            "Expected 3 knot vectors, got " + std::to_string(kvIdx)
            + " (geometry id=" + std::string(geomId) + ")");
    }

    for (int d = 0; d < 3; ++d) {
        const int minLen = 2 * (degree + 1);
        if (static_cast<int>(result.knotVectors[d].size()) < minLen) {
            throw std::runtime_error(
                "Knot vector " + std::to_string(d) + " too short for degree "
                + std::to_string(degree) + " (size="
                + std::to_string(result.knotVectors[d].size())
                + ", minimum=" + std::to_string(minLen) + ")");
        }
    }

    result.nCtrlPts =
        static_cast<int64_t>(result.knotVectors[0].size()) - degree - 1;

    // Debug-Ausgabe
    std::cout << "\n=== KNOT VECTOR DEBUG ===\n";
    std::cout << "Geometry id=" << geomId << "\n";
    for (int d = 0; d < 3; ++d) {
        std::cout << "KV[" << d << "] size=" << result.knotVectors[d].size() << ": ";
        for (double x : result.knotVectors[d]) std::cout << x << " ";
        std::cout << "\n";
    }
    std::cout << "nCtrlPts/direction = " << result.nCtrlPts << "\n";
    std::cout << "=========================\n";

    return result;
}


inline std::vector<double> makeUniformKnotVector(int nCtrlPts, int degree) {
    const int nKnots = nCtrlPts + degree + 1;
    const int nInner = nKnots - 2 * (degree + 1);

    std::vector<double> kv;
    kv.reserve(nKnots);

    for (int i = 0; i <= degree; ++i) kv.push_back(0.0);
    for (int i = 1; i <= nInner; ++i)
        kv.push_back(static_cast<double>(i) / static_cast<double>(nInner + 1));
    for (int i = 0; i <= degree; ++i) kv.push_back(1.0);

    return kv;
}


//computeGrevilleAbscissae
//
//  Berechnet die Greville-Abszissen für einen B-Spline-Raum:
//    g_i = (t_{i+1} + ... + t_{i+degree}) / degree

inline std::vector<double> computeGrevilleAbscissae(
    const std::vector<double>& knotVector, int degree, int numCtrlPts)
{
    std::vector<double> greville(numCtrlPts, 0.0);
    for (int i = 0; i < numCtrlPts; ++i) {
        double sum = 0.0;
        for (int j = i + 1; j <= i + degree; ++j)
            sum += knotVector.at(j);
        greville[i] = sum / static_cast<double>(degree);
    }
    return greville;
}
