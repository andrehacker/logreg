package de.tuberlin.dima.ml.pact.logreg.ensemble;

import java.util.Iterator;

import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactInteger;

public class ReduceEvalSum extends ReduceStub{

  @Override
  public void reduce(Iterator<PactRecord> records, Collector<PactRecord> out)
      throws Exception {
    
    int total = 0;
    int correct = 0;
    PactRecord record = null;
    while(records.hasNext()) {
      record = records.next();
      total += record.getField(EnsembleJob.ID_EVAL_OUT_TOTAL, PactInteger.class).getValue();
      correct += record.getField(EnsembleJob.ID_EVAL_OUT_CORRECT, PactInteger.class).getValue();
    }

    System.out.println("ACCURACY (training-data): " + ((double)correct / (double)total) + " (= " + correct + " / " + total + ")");
    
    // TODO Collect results (model and evaluation)
    
  }

}
