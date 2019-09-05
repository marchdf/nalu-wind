/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/MomentumSSTTAMSForcingElemKernel.h"

namespace {
namespace hex8_golds {
namespace TAMS_SST_forcing {
static constexpr double rhs[24] = {0, 0, 0, 8.2190254322942e-19, 0.020198256106954, -2.2640987110247e-18, 0, 0, 0, 0, 0, 0, -4.2412794242022e-19, -0.0032263677167747, -4.0707282406818e-18, 0, 0, 0, -3.142470837598e-19, -0.0046335735040102, -9.7436809098464e-18, -7.418707918195e-19, -0.0033860744365288, -7.1203853393004e-18, };
} // namespace TAMS_SST_forcing
} // namespace hex8_golds
} // anonymous namespace

TEST_F(TAMSKernelHex8Mesh, NGP_TAMS_SST_forcing)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = 0.0;
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumSSTTAMSForcingElemKernel<
      sierra::nalu::AlgTraitsHex8>(
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
  
  namespace gold_values = ::hex8_golds::TAMS_SST_forcing;
  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, gold_values::rhs);
}
