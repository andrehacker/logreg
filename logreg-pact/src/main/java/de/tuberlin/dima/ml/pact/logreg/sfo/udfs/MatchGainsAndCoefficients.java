package de.tuberlin.dima.ml.pact.logreg.sfo.udfs;

import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.MatchStub;
import eu.stratosphere.pact.common.type.PactRecord;

public class MatchGainsAndCoefficients extends MatchStub {
  
  public static final int IDX_INPUT1_DIMENSION = 0;
  public static final int IDX_INPUT1_GAIN = 1;
  
  public static final int IDX_INPUT2_DIMENSION = 0;
  public static final int IDX_INPUT2_COEFFICIENT = 1;
  
  public static final int IDX_OUT_DIMENSION = 0;
  public static final int IDX_OUT_GAIN = 1;
  public static final int IDX_OUT_COEFFICIENT = 2;
  
  private PactRecord recordOut = new PactRecord(3);

  // TODO _SFO: Define Constant fields
  @Override
  public void match(PactRecord gain, PactRecord coefficient,
      Collector<PactRecord> out) throws Exception {
    
    recordOut.copyFrom(gain, new int[] {IDX_INPUT1_DIMENSION, IDX_INPUT1_GAIN}, 
        new int[] {IDX_OUT_DIMENSION, IDX_OUT_GAIN});

    recordOut.copyFrom(coefficient, new int[] {IDX_INPUT2_COEFFICIENT}, 
        new int[] {IDX_OUT_COEFFICIENT});

    out.collect(recordOut);
  }


}
