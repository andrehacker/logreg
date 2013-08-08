package de.tuberlin.dima.ml.pact.logreg.sfo.udfs;

import org.apache.mahout.math.Vector;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFOGlobalSettings;
import de.tuberlin.dima.ml.pact.logreg.sfo.PactIncrementalModel;
import de.tuberlin.dima.ml.pact.types.PactVector;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.CrossStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactDouble;
import eu.stratosphere.pact.common.type.base.PactInteger;

public class TrainComputeProbabilities extends CrossStub {
  
  public static final int IDX_INPUT1_INPUT_RECORD = 0;
  public static final int IDX_INPUT1_LABEL = 1;
  
  public static final int IDX_INPUT2_BASEMODEL = 0;

  // This contract is chained (hardcoded)
  public static final int IDX_OUT_DIMENSION = TrainDimensions.IDX_DIMENSION;
  public static final int IDX_OUT_LABEL = TrainDimensions.IDX_LABEL;
  public static final int IDX_OUT_XID = TrainDimensions.IDX_XID;
  public static final int IDX_OUT_PI = TrainDimensions.IDX_PI;
  
  private final PactRecord recordOut = new PactRecord();

  @Override
  public void cross(PactRecord trainingVector, PactRecord model,
      Collector<PactRecord> out) throws Exception {

    int y = trainingVector.getField(IDX_INPUT1_INPUT_RECORD, PactInteger.class).getValue();
    Vector xi = trainingVector.getField(IDX_INPUT1_LABEL, PactVector.class).getValue();
    IncrementalModel baseModel = model.getField(IDX_INPUT2_BASEMODEL, PactIncrementalModel.class).getValue();
    
//    System.out.println("TRAIN CROSS: y=" + y + " xi non-zeros=" + xi.getNumNonZeroElements());
    
    double pi = LogRegMath.predict(xi, baseModel.getW(), SFOGlobalSettings.INTERCEPT);
    for (Vector.Element feature : xi.nonZeroes()) {
      if (! baseModel.isFeatureUsed(feature.index())) {
        recordOut.setField(IDX_OUT_DIMENSION, new PactInteger(feature.index()));
        recordOut.setField(IDX_OUT_LABEL, new PactInteger(y));
        recordOut.setField(IDX_OUT_XID, new PactDouble(feature.get()));
        recordOut.setField(IDX_OUT_PI, new PactDouble(pi));
        out.collect(recordOut);
      }
    }
  }

}
