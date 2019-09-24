add_sources(GlobalSourceList
  # Algorithms
  NodalGradEdgeAlg.C
  NodalGradElemAlg.C
  NodalGradBndryElemAlg.C
  EffDiffFluxCoeffAlg.C
  EnthalpyEffDiffFluxCoeffAlg.C
  MdotEdgeAlg.C
  TAMSAvgMdotEdgeAlg.C
  TAMSAvgMdotElemAlg.C
  SSTTAMSAveragesAlg.C
  MetricTensorElemAlg.C
  TurbViscKsgsAlg.C
  WallFuncGeometryAlg.C
  ABLWallFrictionVelAlg.C
  TKEWallFuncAlg.C

  # Algorithm Drivers
  NgpAlgDriver.C
  NodalGradAlgDriver.C
  FieldUpdateAlgDriver.C
  )
