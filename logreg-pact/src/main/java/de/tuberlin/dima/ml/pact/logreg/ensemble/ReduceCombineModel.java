package de.tuberlin.dima.ml.pact.logreg.ensemble;

import java.util.Iterator;

import de.tuberlin.dima.ml.pact.types.PactVector;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactInteger;

public class ReduceCombineModel extends ReduceStub {

  @Override
  public void reduce(Iterator<PactRecord> records, Collector<PactRecord> out)
      throws Exception {
    PactRecord recordOut = new PactRecord();
    
    PactRecord record = null;
    int numModels=0;
    while (records.hasNext()) {
      record = records.next();
      recordOut.setField(
          EnsembleJob.ID_EVAL_IN_FIRST_MODEL + numModels,
          record.getField(EnsembleJob.ID_COMBINE_IN_MODEL, PactVector.class));
      
      ++numModels;
    }
    recordOut.setField(EnsembleJob.ID_EVAL_IN_NUM_MODELS, new PactInteger(numModels));
    
    recordOut.setField(EnsembleJob.ID_EVAL_IN_MODEL_ID, 
        record.getField(EnsembleJob.ID_COMBINE_IN_MODEL_ID, PactInteger.class));
    
    out.collect(recordOut);
  }

}
