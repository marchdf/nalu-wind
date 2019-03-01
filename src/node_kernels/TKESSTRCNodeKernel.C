// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKESSTRCNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

TKESSTRCNodeKernel::TKESSTRCNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKESSTRCNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
TKESSTRCNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  const std::string dofName = "turbulent_ke";
  relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  cROne_ = realm.get_turb_model_constant(TM_cROne);
}

void
TKESSTRCNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  DblType Pk = 0.0;
  DblType sijMag = 0.0;
  DblType wijMag = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset + j);
      const auto dudxji = dudx_.get(node, nDim_ * j + i);
      const auto rateOfStrain = 0.5 * (dudxij + dudxji);
      const auto vorticity = 0.5 * (dudxij - dudxji);
      Pk += dudxij * 2.0 * rateOfStrain;
      sijMag += rateOfStrain * rateOfStrain;
      wijMag += vorticity * vorticity;
    }
  }
  Pk *= tvisc;
  sijMag = stk::math::max(2.0 * sijMag, 1.0e-16);
  wijMag = stk::math::max(2.0 * wijMag, 1.0e-16);

  DblType Dk = betaStar_ * density * sdr * tke;

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk);

  // Rotation-curvature correction. We assume the Lagrangian
  // derivative of the strain rate tensor and the frame of calculation
  // is not rotating (\Omega^{rot} = 0). Therefore, rHat = 0 in the
  // formulation and we don't need cr2 and cr3.
  const DblType rStar = sijMag / wijMag;
  const DblType fRotation = (1 + cROne_) * (2 * rStar) / (1 + rStar) - cROne_;
  const DblType fROne = stk::math::max(stk::math::min(fRotation, 1.25), 0.0);

  rhs(0) += (fROne * Pk - Dk) * dVol;
  lhs(0, 0) += betaStar_ * density * sdr * dVol;
}

} // namespace nalu
} // namespace sierra
