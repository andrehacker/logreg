package de.tuberlin.dima.ml.pact.logreg.sfo.udfs;

import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.sfo.NewtonSingleFeatureOptimizer;
import de.tuberlin.dima.ml.pact.udfs.ReduceFlattenToVector;
import de.tuberlin.dima.ml.pact.util.PactUtils;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactDouble;
import eu.stratosphere.pact.common.type.base.PactInteger;

public class TrainNewFeatures extends ReduceStub {
  
  public static final int IDX_DIMENSION = 0;
  public static final int IDX_LABEL = 1;
  public static final int IDX_XID = 2;
  public static final int IDX_PI = 3;

  // Always chained
  // Key is just a workaround to send to single reducer
  public static final int IDX_OUT_KEY_CONST_ONE = ReduceFlattenToVector.IDX_KEY_CONST_ONE;
  public static final int IDX_OUT_DIMENSION = ReduceFlattenToVector.IDX_DIMENSION;
  public static final int IDX_OUT_COEFICCIENT = ReduceFlattenToVector.IDX_DOUBLE_VALUE;
  
  private static final int MAX_ITERATIONS = 5;
  private static final double LAMBDA = 0;
  private static final double TOLERANCE = 10E-6;
  
  private final PactRecord recordOut = new PactRecord(2);

  @Override
  public void reduce(Iterator<PactRecord> records, Collector<PactRecord> out)
      throws Exception {
    
    List<NewtonSingleFeatureOptimizer.Record> cache = Lists.newArrayList();
    
    // Cache all records
    PactRecord record = null;
    while (records.hasNext()) {
      record = records.next();
      cache.add(new NewtonSingleFeatureOptimizer.Record(
          record.getField(IDX_XID, PactDouble.class).getValue(), 
          record.getField(IDX_LABEL, PactInteger.class).getValue(), 
          record.getField(IDX_PI, PactDouble.class).getValue()));
    }
    int d = record.getField(IDX_DIMENSION, PactInteger.class).getValue();
    
    // Train single dimension using Newton Raphson
    double betad = NewtonSingleFeatureOptimizer.train(cache, MAX_ITERATIONS, LAMBDA, TOLERANCE);
    
//    System.out.println("TRAIN Reducer for dimension " + d);
//    System.out.println("- betad=" + betad);
    
    recordOut.setField(IDX_OUT_KEY_CONST_ONE, PactUtils.pactOne);
    recordOut.setField(IDX_OUT_DIMENSION, new PactInteger(d));
    recordOut.setField(IDX_OUT_COEFICCIENT, new PactDouble(betad));
    out.collect(recordOut);
  }

}
