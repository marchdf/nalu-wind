add_sources(GlobalSourceList
  # Algorithms
  NodalGradEdgeAlg.C
  NodalGradElemAlg.C
  NodalGradBndryElemAlg.C
  EffDiffFluxCoeffAlg.C
  EffSSTDiffFluxCoeffAlg.C
  EnthalpyEffDiffFluxCoeffAlg.C
  MdotEdgeAlg.C
  TAMSAvgMdotEdgeAlg.C
  TAMSAvgMdotElemAlg.C
  SSTTAMSAveragesAlg.C
  MetricTensorElemAlg.C
  TurbViscKsgsAlg.C
  TurbViscSSTAlg.C
  WallFuncGeometryAlg.C
  ABLWallFrictionVelAlg.C
  TKEWallFuncAlg.C
  GeometryInteriorAlg.C
  GeometryBoundaryAlg.C

  # Algorithm Drivers
  NgpAlgDriver.C
  NodalGradAlgDriver.C
  FieldUpdateAlgDriver.C
  TKEWallFuncAlgDriver.C
  GeometryAlgDriver.C
  )
