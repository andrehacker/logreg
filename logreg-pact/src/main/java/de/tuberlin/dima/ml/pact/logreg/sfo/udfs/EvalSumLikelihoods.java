package de.tuberlin.dima.ml.pact.logreg.sfo.udfs;

import java.util.Iterator;

import de.tuberlin.dima.ml.pact.util.PactUtils;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactDouble;
import eu.stratosphere.pact.common.type.base.PactInteger;

public class EvalSumLikelihoods extends ReduceStub {
  
  public static final int IDX_DIMENSION = 0;
  public static final int IDX_LL_BASE = 1;
  public static final int IDX_LL_NEW = 2;
  
  public static final int IDX_OUT_DIMENSION = ApplyBest.IDX_INPUT1_DIMENSION;
  public static final int IDX_OUT_GAIN = ApplyBest.IDX_INPUT1_GAIN;
  public static final int IDX_OUT_KEY_CONST_ONE = ApplyBest.IDX_INPUT1_KEY_CONST_ONE;

  @Override
  public void reduce(Iterator<PactRecord> records, Collector<PactRecord> out)
      throws Exception {
    double sumLlBase=0;
    double sumLlNew=0;
    
    PactRecord record = null;
    while (records.hasNext()) {
      record = records.next();
      sumLlBase += record.getField(IDX_LL_BASE, PactDouble.class).getValue();
      sumLlNew += record.getField(IDX_LL_NEW, PactDouble.class).getValue();
    }
    int dim = record.getField(IDX_DIMENSION, PactInteger.class).getValue();
    double gain = sumLlNew - sumLlBase;
    PactRecord recordOut = new PactRecord(2);
    recordOut.setField(IDX_OUT_KEY_CONST_ONE, PactUtils.pactOne);
    recordOut.setField(IDX_OUT_DIMENSION, new PactInteger(dim));
    recordOut.setField(IDX_OUT_GAIN, new PactDouble(gain));
    out.collect(recordOut);
  }

}
