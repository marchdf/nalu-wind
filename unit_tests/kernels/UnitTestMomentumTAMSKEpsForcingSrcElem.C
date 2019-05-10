/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/MomentumTAMSKEpsForcingElemKernel.h"

namespace {
namespace hex8_golds {
namespace KE_forcing {
static constexpr double rhs[24] = {
  4.2793082631269352e-16, -0.05377930265730299100, -1.9424522584909472e-15,
  0.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000,
  2.1545831683824524e-15, -0.06639995300941596600, -2.8959201357299706e-14,
  3.9031425991144998e-15, -0.04062303723348814600, -1.7717041378370307e-14,
  0.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000,
 -6.2658813417021460e-16,  0.23316872456646012000, -1.8430384412468786e-15,
  0.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000,
  0.00000000000000000000,  0.00000000000000000000,  0.00000000000000000000,
};
} // KE_forcing
} // hex8_golds
} // anonymous namespace

#ifndef KOKKOS_HAVE_CUDA

TEST_F(TAMSKernelHex8Mesh, KE_forcing)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumTAMSKEpsForcingElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, visc_, tvisc_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.currentTime_ = 1.0;
  timeIntegrator.timeStepN_ = 0.1;
  timeIntegrator.timeStepNm1_ = 0.1;
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  // Populate LHS and RHS
  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace gold_values = ::hex8_golds::KE_forcing;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
}

#endif

