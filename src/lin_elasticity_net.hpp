#pragma once

// Structural changes compared to the original:
//
//  1. initialize_problem_data()  [new, public]
//       Precomputes collocation points and all index tensors.
//
//  2. PostProc()  [new, public]
//       Contains the full post-processing block that originally lived
//       at the end of loss() under if(epoch == MAX_EPOCH_-1).

#include "lin_elasticity_utils.hpp"
#include <iganet.h>

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

/// @brief IgANet specialization for 3D linear elasticity.
template <typename Optimizer, typename GeometryMap, typename Variable>
class linear_elasticity
    : public iganet::IgANet<Optimizer, std::tuple<GeometryMap>, std::tuple<Variable>>,
      public iganet::IgANetCustomizable<std::tuple<GeometryMap>, std::tuple<Variable>>
{
private:
    using Inputs       = std::tuple<GeometryMap>;
    using Outputs      = std::tuple<Variable>;
    using Base         = iganet::IgANet<Optimizer, Inputs, Outputs>;
    using Customizable = iganet::IgANetCustomizable<Inputs, Outputs>;

    typename Base::template collPts_t<0> collPts_;
    typename Base::template collPts_t<0> interiorCollPts_;

    // Precomputed index tensors for fast Jacobian/Hessian evaluation.
 
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

    
    // Material parameters.
   
    double lambda_; ///< Lamé parameter lambda.
    double mu_;     ///< Lamé parameter mu (shear modulus).

    typename std::tuple_element_t<0, Outputs> ref_;

    // Simulation parameters.
 
    int    MAX_EPOCH_;
    double MIN_LOSS_;
    int64_t NR_CTRL_PTS_;

    std::array<double, 3> BODY_FORCE_; ///< Body force [fx, fy, fz].

    /// Dirichlet boundary conditions: (side, ux, uy, uz)
    std::vector<std::tuple<int, double, double, double>> DIRI_SIDES_;
    /// Neumann boundary conditions / prescribed tractions: (side, tx, ty, tz)
    std::vector<std::tuple<int, double, double, double>> FORCE_SIDES_;
    /// Traction-free sides as side numbers.
    std::vector<int> TFBC_SIDES_;

    std::string JSON_PATH_;
    bool        SUPERVISED_LEARNING_;

    bool                                  tractionPtsInitialized_ = false;
    std::array<torch::Tensor, 3>          tractionCollPts_;
    std::vector<int>                      nPtsPerSide_;

    int nrCollPts_ = 0;

    /// @brief Returns true if the given side has a Dirichlet condition.
    bool isDirichletSide(int sideNr) const {
        return std::any_of(DIRI_SIDES_.begin(), DIRI_SIDES_.end(),
            [&](const auto& t) { return std::get<0>(t) == sideNr; });
    }

    /// @brief Returns true if the given side has a Neumann traction condition.
    bool isNeumannSide(int sideNr) const {
        return std::any_of(FORCE_SIDES_.begin(), FORCE_SIDES_.end(),
            [&](const auto& t) { return std::get<0>(t) == sideNr; });
    }

    /// @brief Returns boundary-condition priority for conflict resolution at corners.
    /// @details Priority order: Dirichlet = 3, Neumann = 2, traction-free = 1.
    int bc_priority(int sideNr) const {
        if (isDirichletSide(sideNr)) return 3;
        if (isNeumannSide(sideNr))   return 2;
        return 1;
    }

    /// @brief Returns true if another side wins a shared corner against sideNr.
    bool bc_other_wins(int otherSide, int sideNr) const {
        int op = bc_priority(otherSide);
        int tp = bc_priority(sideNr);
        if (op > tp)                    return true;
        if (op == tp && otherSide < sideNr) return true;
        return false;
    }

    /// @brief Returns true if two cube faces intersect geometrically.
    bool sidesIntersect(int a, int b) const {
        if (a == b) return false;
        return !((a==1&&b==2)||(a==2&&b==1)||
                 (a==3&&b==4)||(a==4&&b==3)||
                 (a==5&&b==6)||(a==6&&b==5));
    }

    /// @brief Builds physical face coordinates for one boundary side.
    std::array<torch::Tensor, 3> getFaceBoundaryPoints(int sideNr) const {
        switch (sideNr) {
            case 1: { auto Y=std::get<0>(collPts_.second)[0];
                      auto Z=std::get<0>(collPts_.second)[1];
                      return {torch::zeros_like(Y), Y, Z}; }
            case 2: { auto Y=std::get<1>(collPts_.second)[0];
                      auto Z=std::get<1>(collPts_.second)[1];
                      return {torch::ones_like(Y), Y, Z}; }
            case 3: { auto X=std::get<2>(collPts_.second)[0];
                      auto Z=std::get<2>(collPts_.second)[1];
                      return {X, torch::zeros_like(X), Z}; }
            case 4: { auto X=std::get<3>(collPts_.second)[0];
                      auto Z=std::get<3>(collPts_.second)[1];
                      return {X, torch::ones_like(X), Z}; }
            case 5: { auto X=std::get<4>(collPts_.second)[0];
                      auto Y=std::get<4>(collPts_.second)[1];
                      return {X, Y, torch::zeros_like(X)}; }
            case 6: { auto X=std::get<5>(collPts_.second)[0];
                      auto Y=std::get<5>(collPts_.second)[1];
                      return {X, Y, torch::ones_like(X)}; }
            default:
                throw std::invalid_argument("Boundary side must be 1..6.");
        }
    }

    /// @brief Masks points that lie on a second boundary side.
    torch::Tensor maskPointsOnOtherSide(const std::array<torch::Tensor,3>& pts,
                                         int otherSide) const
    {
        const auto& X=pts[0]; const auto& Y=pts[1]; const auto& Z=pts[2];
        switch (otherSide) {
            case 1: return torch::isclose(X, torch::zeros_like(X));
            case 2: return torch::isclose(X, torch::ones_like(X));
            case 3: return torch::isclose(Y, torch::zeros_like(Y));
            case 4: return torch::isclose(Y, torch::ones_like(Y));
            case 5: return torch::isclose(Z, torch::zeros_like(Z));
            case 6: return torch::isclose(Z, torch::ones_like(Z));
            default:
                throw std::invalid_argument("Boundary side must be 1..6.");
        }
    }

    /// @brief Keeps only boundary points owned by the given side after corner arbitration.
    torch::Tensor buildKeepMaskForSide(int sideNr) const {
        auto pts = getFaceBoundaryPoints(sideNr);
        torch::Tensor keepMask = torch::ones(
            {pts[0].size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(pts[0].device()));

        for (int other = 1; other <= 6; ++other) {
            if (!sidesIntersect(sideNr, other)) continue;
            if (!bc_other_wins(other, sideNr))  continue;
            keepMask = torch::logical_and(
                keepMask, torch::logical_not(maskPointsOnOtherSide(pts, other)));
        }
        return keepMask;
    }

    /// @brief Initializes cached traction collocation points for Neumann-type boundaries.
    void initTractionCollPts(const std::vector<int>& neumannSides,
                              const torch::TensorOptions& opts)
    {
        std::vector<torch::Tensor> xV, yV, zV;
        nPtsPerSide_.clear();

        auto make_face_points = [&](int side) -> std::array<torch::Tensor, 3> {
            return getFaceBoundaryPoints(side);
        };

        for (int side : neumannSides) {
            auto facePts = make_face_points(side);
            auto keepMask = buildKeepMaskForSide(side);
            auto idx = torch::nonzero(keepMask).reshape({-1});

            auto Xf = facePts[0].index_select(0, idx);
            auto Yf = facePts[1].index_select(0, idx);
            auto Zf = facePts[2].index_select(0, idx);

            nPtsPerSide_.push_back(static_cast<int>(Xf.size(0)));
            if (Xf.size(0) > 0) {
                xV.push_back(Xf); yV.push_back(Yf); zV.push_back(Zf);
            }
        }

        if (!xV.empty()) {
            tractionCollPts_ = {
                torch::cat(xV, 0),
                torch::cat(yV, 0),
                torch::cat(zV, 0)};
        } else {
            tractionCollPts_ = {
                torch::empty({0}, opts),
                torch::empty({0}, opts),
                torch::empty({0}, opts)};
        }

        // Precompute indices for boundary evaluations.
        var_knot_indices_boundary_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                tractionCollPts_);
        var_coeff_indices_boundary_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_boundary_);
        G_knot_indices_boundary_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                tractionCollPts_);
        G_coeff_indices_boundary_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_boundary_);

        tractionPtsInitialized_ = true;
    }


public:
    /// @brief Constructs the 3D linear-elasticity network wrapper.
    template <typename... Args>
    linear_elasticity(double lambda, double mu, bool SUPERVISED_LEARNING,
                      int MAX_EPOCH, double MIN_LOSS,
                      std::array<double, 3> BODY_FORCE,
                      std::vector<int> TFBC_SIDES,
                      std::vector<std::tuple<int,double,double,double>> FORCE_SIDES,
                      std::vector<std::tuple<int,double,double,double>> DIRI_SIDES,
                      int64_t NR_CTRL_PTS, std::string JSON_PATH,
                      std::vector<int64_t>&& layers,
                      std::vector<std::vector<std::any>>&& activations,
                      Args&&... args)
        : Base(std::forward<std::vector<int64_t>>(layers),
               std::forward<std::vector<std::vector<std::any>>>(activations),
               std::forward<Args>(args)...)
        , lambda_(lambda), mu_(mu)
        , ref_(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS, NR_CTRL_PTS))
        , MAX_EPOCH_(MAX_EPOCH), MIN_LOSS_(MIN_LOSS), NR_CTRL_PTS_(NR_CTRL_PTS)
        , BODY_FORCE_(BODY_FORCE)
        , DIRI_SIDES_(DIRI_SIDES), FORCE_SIDES_(FORCE_SIDES), TFBC_SIDES_(TFBC_SIDES)
        , JSON_PATH_(std::move(JSON_PATH))
        , SUPERVISED_LEARNING_(SUPERVISED_LEARNING)
    {}

    /// @brief Returns the reference displacement field.
    auto const& ref() const { return ref_; }
    /// @brief Returns the mutable reference displacement field.
    auto&       ref()       { return ref_; }

    /// @brief Writes one result entry into the configured JSON output file.
    void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
        ::appendToJsonFile(JSON_PATH_, key, data);
    }

    /// @brief Precomputes collocation points and knot/coefficient index caches.
    void initialize_problem_data() {
        Base::inputs(0);
        collPts_         = Base::template collPts<0>(iganet::collPts::greville);
        interiorCollPts_ = Base::template collPts<0>(iganet::collPts::greville_interior);

        nrCollPts_ = static_cast<int>(
            std::cbrt(static_cast<double>(std::get<0>(collPts_)[0].size(0))));

        torch::Tensor collPtsCoeffs =
            std::get<0>(collPts_)[0].slice(0, 0, nrCollPts_);
        nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
        for (int i = 0; i < collPtsCoeffs.size(0); ++i)
            collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
        appendToJsonFile("net_collPtsCoeffsRef1", collPtsCoeffs_j);
        appendToJsonFile("net_nrCollPtsRef1", {nrCollPts_});

        var_knot_indices_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                collPts_.first);
        var_coeff_indices_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_);

        var_knot_indices_interior_ =
            Base::template output<0>().template find_knot_indices<iganet::functionspace::interior>(
                interiorCollPts_.first);
        var_coeff_indices_interior_ =
            Base::template output<0>().template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_interior_);

        G_knot_indices_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                collPts_.first);
        G_coeff_indices_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_);

        G_knot_indices_interior_ =
            this->template input<0>().template find_knot_indices<iganet::functionspace::interior>(
                interiorCollPts_.first);
        G_coeff_indices_interior_ =
            this->template input<0>().template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_interior_);

        if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty()) {
            std::vector<int> neumannSides;
            neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
            for (const auto& f : FORCE_SIDES_)
                neumannSides.push_back(std::get<0>(f));

            auto dummyOpts = torch::TensorOptions().dtype(torch::kDouble);
            initTractionCollPts(neumannSides, dummyOpts);
        }
    }

    /// @brief Epoch callback used for lightweight logging.
    bool epoch(int64_t epoch) override {
        std::cout << "Epoch: " << epoch << std::endl;
        return epoch == 0;
    }

    /// @brief Computes the training loss for either supervised or unsupervised mode.
    torch::Tensor loss(const torch::Tensor& outputs, int64_t epoch) override {

        this->template output<0>().from_tensor(outputs);

        torch::Tensor totalLoss;
        torch::Tensor elastLoss;
        std::optional<torch::Tensor> bcLoss, tfbcLoss, supLoss, forceLoss;
        std::optional<torch::Tensor> forceValues, targetForce;
        std::optional<torch::Tensor> tractionFreeValues, tractionZeros;

        // Traction / Neumann boundary conditions.
      
        if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty()) {

            std::vector<int> neumannSides;
            neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
            for (const auto& f : FORCE_SIDES_)
                neumannSides.push_back(std::get<0>(f));

            if (!tractionPtsInitialized_)
                initTractionCollPts(neumannSides, outputs.options());

            if (tractionCollPts_[0].numel() > 0) {
                auto jacobianBoundary = this->template output<0>().ijac(
                    this->template input<0>(), tractionCollPts_,
                    var_knot_indices_boundary_, var_coeff_indices_boundary_,
                    G_knot_indices_boundary_,   G_coeff_indices_boundary_);

                auto ux_x = *jacobianBoundary[0]; auto ux_y = *jacobianBoundary[1];
                auto ux_z = *jacobianBoundary[2]; auto uy_x = *jacobianBoundary[3];
                auto uy_y = *jacobianBoundary[4]; auto uy_z = *jacobianBoundary[5];
                auto uz_x = *jacobianBoundary[6]; auto uz_y = *jacobianBoundary[7];
                auto uz_z = *jacobianBoundary[8];

                const int64_t nTrac = tractionCollPts_[0].size(0);
                torch::Tensor tvX = torch::zeros({nTrac}, ux_x.options());
                torch::Tensor tvY = torch::zeros({nTrac}, ux_x.options());
                torch::Tensor tvZ = torch::zeros({nTrac}, ux_x.options());

                int pointCtr = 0;
                int sideCtr  = 0;
                for (int side : neumannSides) {
                    int n = nPtsPerSide_[sideCtr];
                    for (int i = 0; i < n; ++i) {
                        int idx = pointCtr + i;
                        if (side == 1) {
                            tvX[idx] = -((lambda_+2.*mu_)*ux_x[idx]+lambda_*uy_y[idx]+lambda_*uz_z[idx]);
                            tvY[idx] = -(mu_*(ux_y[idx]+uy_x[idx]));
                            tvZ[idx] = -(mu_*(ux_z[idx]+uz_x[idx]));
                        } else if (side == 2) {
                            tvX[idx] =  (lambda_+2.*mu_)*ux_x[idx]+lambda_*uy_y[idx]+lambda_*uz_z[idx];
                            tvY[idx] =   mu_*(ux_y[idx]+uy_x[idx]);
                            tvZ[idx] =   mu_*(ux_z[idx]+uz_x[idx]);
                        } else if (side == 3) {
                            tvX[idx] = -(mu_*(ux_y[idx]+uy_x[idx]));
                            tvY[idx] = -(lambda_*ux_x[idx]+(lambda_+2.*mu_)*uy_y[idx]+lambda_*uz_z[idx]);
                            tvZ[idx] = -(mu_*(uy_z[idx]+uz_y[idx]));
                        } else if (side == 4) {
                            tvX[idx] =   mu_*(ux_y[idx]+uy_x[idx]);
                            tvY[idx] =   lambda_*ux_x[idx]+(lambda_+2.*mu_)*uy_y[idx]+lambda_*uz_z[idx];
                            tvZ[idx] =   mu_*(uy_z[idx]+uz_y[idx]);
                        } else if (side == 5) {
                            tvX[idx] = -(mu_*(ux_z[idx]+uz_x[idx]));
                            tvY[idx] = -(mu_*(uy_z[idx]+uz_y[idx]));
                            tvZ[idx] = -(lambda_*ux_x[idx]+lambda_*uy_y[idx]+(lambda_+2.*mu_)*uz_z[idx]);
                        } else if (side == 6) {
                            tvX[idx] =   mu_*(ux_z[idx]+uz_x[idx]);
                            tvY[idx] =   mu_*(uy_z[idx]+uz_y[idx]);
                            tvZ[idx] =   lambda_*ux_x[idx]+lambda_*uy_y[idx]+(lambda_+2.*mu_)*uz_z[idx];
                        } else {
                            throw std::invalid_argument("Side for 3D traction BC has to be 1..6.");
                        }
                    }
                    pointCtr += n;
                    ++sideCtr;
                }

                torch::Tensor tractionValues =
                    torch::stack({tvX, tvY, tvZ}, 1);

                if (!FORCE_SIDES_.empty()) {
                    int cutlength = 0;
                    int forceSize = static_cast<int>(FORCE_SIDES_.size());
                    for (int i = static_cast<int>(nPtsPerSide_.size()) - forceSize;
                         i < static_cast<int>(nPtsPerSide_.size()); ++i)
                        cutlength += nPtsPerSide_[i];

                    tractionFreeValues.emplace(
                        tractionValues.slice(0, 0, tractionValues.size(0) - cutlength));
                    tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));

                    forceValues.emplace(
                        tractionValues.slice(0, tractionValues.size(0) - cutlength));
                    targetForce.emplace(torch::zeros_like(*forceValues));

                    int offset   = 0;
                    int startIdx = static_cast<int>(nPtsPerSide_.size()) - forceSize;
                    for (size_t i = 0; i < FORCE_SIDES_.size(); ++i) {
                        int rPts = nPtsPerSide_[startIdx + static_cast<int>(i)];
                        auto row = (*targetForce).slice(0, offset, offset + rPts);
                        row.slice(1,0,1).fill_(std::get<1>(FORCE_SIDES_[i]));
                        row.slice(1,1,2).fill_(std::get<2>(FORCE_SIDES_[i]));
                        row.slice(1,2,3).fill_(std::get<3>(FORCE_SIDES_[i]));
                        offset += rPts;
                    }
                } else {
                    tractionFreeValues.emplace(tractionValues);
                    tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
                }
            }
        }

        auto hessianColl = this->template output<0>().ihess(
            this->template input<0>(), interiorCollPts_.first,
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_,   G_coeff_indices_interior_);

        auto& ux_xx=hessianColl(0,0,0); auto& ux_yy=hessianColl(1,1,0); auto& ux_zz=hessianColl(2,2,0);
        auto& uy_xy=hessianColl(0,1,1); auto& uz_xz=hessianColl(0,2,2);
        auto& uy_xx=hessianColl(0,0,1); auto& uy_yy=hessianColl(1,1,1); auto& uy_zz=hessianColl(2,2,1);
        auto& ux_yx=hessianColl(1,0,0); auto& uz_yz=hessianColl(1,2,2);
        auto& uz_xx=hessianColl(0,0,2); auto& uz_yy=hessianColl(1,1,2); auto& uz_zz=hessianColl(2,2,2);
        auto& ux_zx=hessianColl(2,0,0); auto& uy_zy=hessianColl(2,1,1);

        int64_t size = hessianColl(0,0,0).size(0);
        auto opts = hessianColl(0,0,0).options();

        torch::Tensor divStressX = torch::zeros({size}, opts);
        torch::Tensor divStressY = torch::zeros({size}, opts);
        torch::Tensor divStressZ = torch::zeros({size}, opts);

        for (int i = 0; i < size; ++i) {
            divStressX[i] = (lambda_+2.*mu_)*ux_xx[i] + mu_*ux_yy[i] + mu_*ux_zz[i]
                            + (lambda_+mu_)*(uy_xy[i]+uz_xz[i]);
            divStressY[i] = mu_*uy_xx[i] + (lambda_+2.*mu_)*uy_yy[i] + mu_*uy_zz[i]
                            + (lambda_+mu_)*(ux_yx[i]+uz_yz[i]);
            divStressZ[i] = mu_*uz_xx[i] + mu_*uz_yy[i] + (lambda_+2.*mu_)*uz_zz[i]
                            + (lambda_+mu_)*(ux_zx[i]+uy_zy[i]);
        }

        torch::Tensor divStress = torch::stack({divStressX, divStressY, divStressZ}, 1);

        torch::Tensor bodyForce = torch::tensor(
            {BODY_FORCE_[0], BODY_FORCE_[1], BODY_FORCE_[2]}, opts)
            .view({1,3}).repeat({divStress.size(0), 1});

 
        auto masked_side_loss = [&](const torch::Tensor& u0, const torch::Tensor& u1,
                                    const torch::Tensor& u2, const torch::Tensor& b0,
                                    const torch::Tensor& b1, const torch::Tensor& b2,
                                    int sideNr) -> torch::Tensor {
            auto keepMask = buildKeepMaskForSide(sideNr);
            auto keepIdx  = torch::nonzero(keepMask).reshape({-1});
            if (keepIdx.numel() == 0) return torch::zeros({}, outputs.options());
            return torch::mse_loss(u0.index_select(0,keepIdx), b0.index_select(0,keepIdx))
                 + torch::mse_loss(u1.index_select(0,keepIdx), b1.index_select(0,keepIdx))
                 + torch::mse_loss(u2.index_select(0,keepIdx), b2.index_select(0,keepIdx));
        };

        auto add_masked_side_loss = [&](const auto& u_side, const auto& b_side, int sNr) {
            *bcLoss += static_cast<double>(SUPERVISED_LEARNING_ ? 1 : 100000)
                * masked_side_loss(*u_side[0],*u_side[1],*u_side[2],
                                   *b_side[0],*b_side[1],*b_side[2], sNr);
        };

        // Unsupervised learning.
  
        if (!SUPERVISED_LEARNING_) {
            std::ostringstream log;
            elastLoss = torch::mse_loss(divStress, -bodyForce);
            totalLoss = elastLoss;
            log << "EL " << std::setw(11) << elastLoss.item<double>();

            if (!TFBC_SIDES_.empty()) {
                tfbcLoss  = torch::mse_loss(*tractionFreeValues, *tractionZeros);
                totalLoss += *tfbcLoss;
                log << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
            }
            if (!FORCE_SIDES_.empty()) {
                forceLoss  = torch::mse_loss(*forceValues, *targetForce);
                totalLoss += *forceLoss;
                log << " + FL " << std::setw(11) << (*forceLoss).item<double>();
            }
            if (!DIRI_SIDES_.empty()) {
                const double bcWeight = 1e5;
                bcLoss = torch::tensor(0.0, outputs.options());
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.second);
                auto bdr   = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);
                for (const auto& side : DIRI_SIDES_) {
                    int sNr = std::get<0>(side);
                    switch (sNr) {
                        case 1: add_masked_side_loss(std::get<0>(u_bdr),std::get<0>(bdr),1); break;
                        case 2: add_masked_side_loss(std::get<1>(u_bdr),std::get<1>(bdr),2); break;
                        case 3: add_masked_side_loss(std::get<2>(u_bdr),std::get<2>(bdr),3); break;
                        case 4: add_masked_side_loss(std::get<3>(u_bdr),std::get<3>(bdr),4); break;
                        case 5: add_masked_side_loss(std::get<4>(u_bdr),std::get<4>(bdr),5); break;
                        case 6: add_masked_side_loss(std::get<5>(u_bdr),std::get<5>(bdr),6); break;
                        default: std::cerr << "Invalid Dirichlet side!\n";
                    }
                }
                totalLoss += *bcLoss;
                log << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight
                    << " * 1e" << static_cast<int>(std::log10(bcWeight));
            }
            std::cout << std::setw(11) << totalLoss.item<double>()
                      << " = " << log.str() << std::endl;
        }

        // Supervised learning.

        else if (SUPERVISED_LEARNING_) {
            std::ostringstream log;

            torch::Tensor netDisp = torch::stack({
                outputs.slice(0, 0,              outputs.size(0)/3),
                outputs.slice(0, outputs.size(0)/3,   2*outputs.size(0)/3),
                outputs.slice(0, 2*outputs.size(0)/3, outputs.size(0))}, 1);

            torch::Tensor stdDisp = loadDisplacements(JSON_PATH_)
                                    .to(netDisp.options());

            const double supWeight = 1e7;
            supLoss   = supWeight * torch::mse_loss(netDisp, stdDisp);
            elastLoss = torch::mse_loss(divStress, -bodyForce);
            totalLoss = *supLoss + elastLoss;

            log << "SL " << std::setw(11) << (*supLoss).item<double>() / supWeight
                << " * 1e" << static_cast<int>(std::log10(supWeight))
                << " + EL " << std::setw(11) << elastLoss.item<double>();

            if (!TFBC_SIDES_.empty()) {
                tfbcLoss  = torch::mse_loss(*tractionFreeValues, *tractionZeros);
                totalLoss += *tfbcLoss;
                log << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
            }
            if (!FORCE_SIDES_.empty()) {
                forceLoss  = torch::mse_loss(*forceValues, *targetForce);
                totalLoss += *forceLoss;
                log << " + FL " << std::setw(11) << (*forceLoss).item<double>();
            }
            if (!DIRI_SIDES_.empty()) {
                const double bcWeight = 1e0;
                bcLoss = torch::tensor(0.0, outputs.options());
                auto u_bdr = this->template output<0>().template eval<iganet::functionspace::boundary>(collPts_.second);
                auto bdr   = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);
                for (const auto& side : DIRI_SIDES_) {
                    int sNr = std::get<0>(side);
                    switch (sNr) {
                        case 1: add_masked_side_loss(std::get<0>(u_bdr),std::get<0>(bdr),1); break;
                        case 2: add_masked_side_loss(std::get<1>(u_bdr),std::get<1>(bdr),2); break;
                        case 3: add_masked_side_loss(std::get<2>(u_bdr),std::get<2>(bdr),3); break;
                        case 4: add_masked_side_loss(std::get<3>(u_bdr),std::get<3>(bdr),4); break;
                        case 5: add_masked_side_loss(std::get<4>(u_bdr),std::get<4>(bdr),5); break;
                        case 6: add_masked_side_loss(std::get<5>(u_bdr),std::get<5>(bdr),6); break;
                        default: std::cerr << "Invalid Dirichlet side!\n";
                    }
                }
                totalLoss += *bcLoss;
                log << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight
                    << " * 1e" << static_cast<int>(std::log10(bcWeight));
            }
            std::cout << std::setw(11) << totalLoss.item<double>()
                      << " = " << log.str() << std::endl;
        } else {
            throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_");
        }

        return totalLoss;
    }

    /// @brief Exports derived stresses, displaced collocation points, and residual fields.
    void PostProc() {

        // Jacobian at all collocation points.
        auto jacobian = this->template output<0>().ijac(
            this->template input<0>(), collPts_.first,
            var_knot_indices_,  var_coeff_indices_,
            G_knot_indices_,    G_coeff_indices_);

        auto ux_x=*jacobian[0]; auto ux_y=*jacobian[1]; auto ux_z=*jacobian[2];
        auto uy_x=*jacobian[3]; auto uy_y=*jacobian[4]; auto uy_z=*jacobian[5];
        auto uz_x=*jacobian[6]; auto uz_y=*jacobian[7]; auto uz_z=*jacobian[8];

        const int64_t nPts = jacobian[0]->size(0);

        // Stress tensor components.
        torch::Tensor sigma_xx = torch::zeros({nPts});
        torch::Tensor sigma_xy = torch::zeros({nPts});
        torch::Tensor sigma_xz = torch::zeros({nPts});
        torch::Tensor sigma_yy = torch::zeros({nPts});
        torch::Tensor sigma_yz = torch::zeros({nPts});
        torch::Tensor sigma_zz = torch::zeros({nPts});
        torch::Tensor sigma_vm = torch::zeros({nPts});

        nlohmann::json netVmStresses_j = nlohmann::json::array();
        nlohmann::json netXStresses_j  = nlohmann::json::array();
        nlohmann::json netYStresses_j  = nlohmann::json::array();
        nlohmann::json netZStresses_j  = nlohmann::json::array();

        for (int i = 0; i < nPts; ++i) {
            // Hooke's law for isotropic linear elasticity.
            sigma_xx[i] = lambda_*(ux_x[i]+uy_y[i]+uz_z[i]) + 2.*mu_*ux_x[i];
            sigma_xy[i] = mu_*(uy_x[i]+ux_y[i]);
            sigma_xz[i] = mu_*(uz_x[i]+ux_z[i]);
            sigma_yy[i] = lambda_*(ux_x[i]+uy_y[i]+uz_z[i]) + 2.*mu_*uy_y[i];
            sigma_yz[i] = mu_*(uz_y[i]+uy_z[i]);
            sigma_zz[i] = lambda_*(ux_x[i]+uy_y[i]+uz_z[i]) + 2.*mu_*uz_z[i];

            // Von Mises equivalent stress.
            sigma_vm[i] = sqrt(0.5*(
                (sigma_xx[i]-sigma_yy[i])*(sigma_xx[i]-sigma_yy[i]) +
                (sigma_yy[i]-sigma_zz[i])*(sigma_yy[i]-sigma_zz[i]) +
                (sigma_zz[i]-sigma_xx[i])*(sigma_zz[i]-sigma_xx[i]) +
                6.*(sigma_xy[i]*sigma_xy[i]+sigma_yz[i]*sigma_yz[i]+sigma_xz[i]*sigma_xz[i])));

            netVmStresses_j.push_back({sigma_vm[i].item<double>()});
            netXStresses_j.push_back( {sigma_xx[i].item<double>()});
            netYStresses_j.push_back( {sigma_yy[i].item<double>()});
            netZStresses_j.push_back( {sigma_zz[i].item<double>()});
        }

        appendToJsonFile("net_VmStresses", netVmStresses_j);
        appendToJsonFile("net_XStresses",  netXStresses_j);
        appendToJsonFile("net_YStresses",  netYStresses_j);
        appendToJsonFile("net_ZStresses",  netZStresses_j);

        // Collocation points: reference and deformed positions.
        torch::Tensor cpRef = torch::stack(
            {std::get<0>(collPts_.first),
             std::get<1>(collPts_.first),
             std::get<2>(collPts_.first)}, 1);
        auto displ = this->template output<0>().eval(collPts_.first);
        torch::Tensor cpDispl = torch::stack({*displ[0],*displ[1],*displ[2]}, 1);

        nlohmann::json collPtsFirst_j      = nlohmann::json::array();
        nlohmann::json collPtsFirstDispl_j = nlohmann::json::array();
        for (int i = 0; i < cpRef.size(0); ++i) {
            collPtsFirst_j.push_back({
                cpRef[i][0].item<double>(),
                cpRef[i][1].item<double>(),
                cpRef[i][2].item<double>()});
            collPtsFirstDispl_j.push_back({
                cpRef[i][0].item<double>() + cpDispl[i][0].item<double>(),
                cpRef[i][1].item<double>() + cpDispl[i][1].item<double>(),
                cpRef[i][2].item<double>() + cpDispl[i][2].item<double>()});
        }
        appendToJsonFile("net_collPtsFirstAsTensor",
                         collPtsFirst_j);
        appendToJsonFile("net_collPtsFirstAfterDisplacementAsTensor",
                         collPtsFirstDispl_j);

        // Stress divergence for residual analysis.
        auto hessianColl = this->template output<0>().ihess(
            this->template input<0>(), interiorCollPts_.first,
            var_knot_indices_interior_, var_coeff_indices_interior_,
            G_knot_indices_interior_,   G_coeff_indices_interior_);

        auto& dux_xx=hessianColl(0,0,0); auto& dux_yy=hessianColl(1,1,0); auto& dux_zz=hessianColl(2,2,0);
        auto& duy_xy=hessianColl(0,1,1); auto& duz_xz=hessianColl(0,2,2);
        auto& duy_xx=hessianColl(0,0,1); auto& duy_yy=hessianColl(1,1,1); auto& duy_zz=hessianColl(2,2,1);
        auto& dux_yx=hessianColl(1,0,0); auto& duz_yz=hessianColl(1,2,2);
        auto& duz_xx=hessianColl(0,0,2); auto& duz_yy=hessianColl(1,1,2); auto& duz_zz=hessianColl(2,2,2);
        auto& dux_zx=hessianColl(2,0,0); auto& duy_zy=hessianColl(2,1,1);

        const int64_t szInner = hessianColl(0,0,0).size(0);
        auto optsInner = hessianColl(0,0,0).options();
        torch::Tensor divX = torch::zeros({szInner}, optsInner);
        torch::Tensor divY = torch::zeros({szInner}, optsInner);
        torch::Tensor divZ = torch::zeros({szInner}, optsInner);

        for (int i = 0; i < szInner; ++i) {
            divX[i] = (lambda_+2.*mu_)*dux_xx[i]+mu_*dux_yy[i]+mu_*dux_zz[i]+(lambda_+mu_)*(duy_xy[i]+duz_xz[i]);
            divY[i] = mu_*duy_xx[i]+(lambda_+2.*mu_)*duy_yy[i]+mu_*duy_zz[i]+(lambda_+mu_)*(dux_yx[i]+duz_yz[i]);
            divZ[i] = mu_*duz_xx[i]+mu_*duz_yy[i]+(lambda_+2.*mu_)*duz_zz[i]+(lambda_+mu_)*(dux_zx[i]+duy_zy[i]);
        }

        nlohmann::json divX_j = nlohmann::json::array();
        nlohmann::json divY_j = nlohmann::json::array();
        nlohmann::json divZ_j = nlohmann::json::array();
        for (int i = 0; i < szInner; ++i) {
            divX_j.push_back({divX[i].item<double>()});
            divY_j.push_back({divY[i].item<double>()});
            divZ_j.push_back({divZ[i].item<double>()});
        }
        appendToJsonFile("net_DivergenceX", divX_j);
        appendToJsonFile("net_DivergenceY", divY_j);
        appendToJsonFile("net_DivergenceZ", divZ_j);
    }


#ifdef IGANET_WITH_GISMO
    /// @brief Runs the original GISMO-based 2D reference simulation.
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RunGismoSimulation(
        int64_t NR_CTRL_PTS, int DEGREE, double YOUNG_MODULUS, double POISSON_RATIO,
        const std::vector<std::tuple<int,double,double>>& DIRI_SIDES,
        const std::vector<std::tuple<int,double,double>>& FORCE_SIDES,
        const std::pair<double,double>& BODY_FORCE)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
        const int64_t nPts = NR_CTRL_PTS * NR_CTRL_PTS;
        torch::Tensor ctrlPts       = torch::empty({nPts,2}, opts);
        torch::Tensor displacements = torch::empty({nPts,2}, opts);
        torch::Tensor stresses      = torch::empty({nPts,1}, opts);
        auto ctrlA = ctrlPts.accessor<double,2>();
        auto dispA = displacements.accessor<double,2>();
        auto strA  = stresses.accessor<double,2>();
        gismo::gsMatrix<double> ctrlPts_gs(nPts, 2);
        gismo::gsKnotVector<double> kv_u(0.,1., NR_CTRL_PTS-DEGREE-1, DEGREE+1);
        gismo::gsKnotVector<double> kv_v(0.,1., NR_CTRL_PTS-DEGREE-1, DEGREE+1);
        std::vector<double> gU = computeGrevilleAbscissae(
            std::vector<double>(kv_u.begin(),kv_u.end()), DEGREE, NR_CTRL_PTS);
        std::vector<double> gV = computeGrevilleAbscissae(
            std::vector<double>(kv_v.begin(),kv_v.end()), DEGREE, NR_CTRL_PTS);
        int64_t idx = 0;
        for (int j=0; j<NR_CTRL_PTS; ++j) for (int i=0; i<NR_CTRL_PTS; ++i) {
            ctrlA[idx][0]=gU[i]; ctrlA[idx][1]=gV[j];
            ctrlPts_gs(idx,0)=gU[i]; ctrlPts_gs(idx,1)=gV[j]; ++idx;
        }
        gismo::gsTensorBSpline<2,double> geometry(kv_u, kv_v, ctrlPts_gs);
        gismo::gsMultiPatch<double> mp; mp.addPatch(geometry);
        gismo::gsMultiBasis<> basis(mp);
        auto getGsSide = [](int s) -> gismo::boundary::side {
            switch(s){
                case 1: return gismo::boundary::west;
                case 2: return gismo::boundary::east;
                case 3: return gismo::boundary::south;
                case 4: return gismo::boundary::north;
                default: throw std::invalid_argument("Invalid side (must be 1..4)");
            }
        };
        gismo::gsBoundaryConditions<double> bcInfo;
        for (const auto& d : DIRI_SIDES) {
            auto gs = getGsSide(std::get<0>(d));
            bcInfo.addCondition(0, gs, gismo::condition_type::dirichlet,
                gismo::gsConstantFunction<double>(std::get<1>(d),2), 0);
            bcInfo.addCondition(0, gs, gismo::condition_type::dirichlet,
                gismo::gsConstantFunction<double>(std::get<2>(d),2), 1);
        }
        for (const auto& f : FORCE_SIDES) {
            auto gs = getGsSide(std::get<0>(f));
            gismo::gsFunctionExpr<> t(std::to_string(std::get<1>(f)),
                                      std::to_string(std::get<2>(f)), 2);
            bcInfo.addCondition(0, gs, gismo::condition_type::neumann, t);
        }
        gismo::gsConstantFunction<double> bf(BODY_FORCE.first, BODY_FORCE.second, 2);
        gismo::gsElasticityAssembler<double> asm_(geometry, basis, bcInfo, bf);
        asm_.options().setReal("YoungsModulus",  YOUNG_MODULUS);
        asm_.options().setReal("PoissonsRatio",  POISSON_RATIO);
        asm_.assemble();
        gismo::gsSparseSolver<>::CGDiagonal solver;
        gismo::gsMatrix<double> sol;
        solver.compute(asm_.matrix()); sol = solver.solve(asm_.rhs());
        gismo::gsMultiPatch<double> solPatch;
        asm_.constructSolution(sol, asm_.allFixedDofs(), solPatch);
        gismo::gsPiecewiseFunction<double> stressFn;
        asm_.constructCauchyStresses(solPatch, stressFn, gismo::stress_components::von_mises);
        for (int i=0; i<ctrlPts_gs.rows(); ++i) {
            gismo::gsMatrix<double> pt(2,1);
            pt(0,0)=ctrlPts_gs(i,0); pt(1,0)=ctrlPts_gs(i,1);
            auto u = solPatch.patch(0).eval(pt);
            dispA[i][0]=u(0); dispA[i][1]=u(1);
            gismo::gsMatrix<double> s(1,1);
            stressFn.piece(0).eval_into(pt, s);
            strA[i][0]=s(0,0);
        }
        return {ctrlPts, displacements, stresses};
    }
#endif
};
