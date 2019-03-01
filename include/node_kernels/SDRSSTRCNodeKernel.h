// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SDRSSTRCNODEKERNEL_H
#define SDRSSTRCNODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/Entity.hpp>

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class Realm;

class SDRSSTRCNodeKernel : public NGPNodeKernel<SDRSSTRCNodeKernel>
{
public:
  SDRSSTRCNodeKernel(const stk::mesh::MetaData&);

  KOKKOS_FORCEINLINE_FUNCTION
  SDRSSTRCNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~SDRSSTRCNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> tke_;
  ngp::Field<double> sdr_;
  ngp::Field<double> density_;
  ngp::Field<double> tvisc_;
  ngp::Field<double> dudx_;
  ngp::Field<double> dkdx_;
  ngp::Field<double> dwdx_;
  ngp::Field<double> dualNodalVolume_;
  ngp::Field<double> fOneBlend_;

  unsigned tkeID_{stk::mesh::InvalidOrdinal};
  unsigned sdrID_{stk::mesh::InvalidOrdinal};
  unsigned densityID_{stk::mesh::InvalidOrdinal};
  unsigned tviscID_{stk::mesh::InvalidOrdinal};
  unsigned dudxID_{stk::mesh::InvalidOrdinal};
  unsigned dkdxID_{stk::mesh::InvalidOrdinal};
  unsigned dwdxID_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_{stk::mesh::InvalidOrdinal};
  unsigned fOneBlendID_{stk::mesh::InvalidOrdinal};

  NodeKernelTraits::DblType betaStar_;
  NodeKernelTraits::DblType tkeProdLimitRatio_;
  NodeKernelTraits::DblType sigmaWTwo_;
  NodeKernelTraits::DblType betaOne_;
  NodeKernelTraits::DblType betaTwo_;
  NodeKernelTraits::DblType gammaOne_;
  NodeKernelTraits::DblType gammaTwo_;
  NodeKernelTraits::DblType cROne_;
  NodeKernelTraits::DblType relaxFac_;

  const int nDim_;
};

} // namespace nalu
} // namespace sierra

#endif /* SDRSSTRCNODEKERNEL_H */
