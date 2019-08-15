/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTETAMSSSTAVERAGESNODEALGORITHM_H
#define COMPUTETAMSSSTAVERAGESNODEALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeTAMSSSTAveragesNodeAlgorithm : public Algorithm {
public:
  ComputeTAMSSSTAveragesNodeAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeTAMSSSTAveragesNodeAlgorithm() {}

  virtual void execute();

  const double betaStar_;
  const bool meshMotion_;

  VectorFieldType *velocityRTM_{nullptr};
  ScalarFieldType *pressure_{nullptr};
  ScalarFieldType *density_{nullptr};
  ScalarFieldType *specDissipationRate_{nullptr};
  ScalarFieldType *turbKineticEnergy_{nullptr};
  GenericFieldType *dudx_{nullptr};
  VectorFieldType *avgVelocity_{nullptr};
  ScalarFieldType *avgPress_{nullptr};
  ScalarFieldType *avgDensity_{nullptr};
  ScalarFieldType *avgTkeRes_{nullptr};
  ScalarFieldType *avgTime_{nullptr};
  GenericFieldType *avgDudx_{nullptr};
  ScalarFieldType *avgProd_{nullptr};
  ScalarFieldType *visc_{nullptr};
  ScalarFieldType *tvisc_{nullptr};
  ScalarFieldType *alpha_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
