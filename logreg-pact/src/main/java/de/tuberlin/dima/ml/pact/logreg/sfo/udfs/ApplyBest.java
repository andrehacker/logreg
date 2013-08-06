package de.tuberlin.dima.ml.pact.logreg.sfo.udfs;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.datasets.DatasetInfo;
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.pact.logreg.sfo.PactIncrementalModel;
import eu.stratosphere.pact.common.stubs.CoGroupStub;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactDouble;
import eu.stratosphere.pact.common.type.base.PactInteger;

public class ApplyBest extends CoGroupStub {

  public static final int IDX_INPUT1_DIMENSION = 0;
  public static final int IDX_INPUT1_GAIN = 1;
  public static final int IDX_INPUT1_KEY_CONST_ONE = 2;
  
  public static final int IDX_INPUT2_BASEMODEL = 0;

  @Override
  public void coGroup(Iterator<PactRecord> gainRecords,
      Iterator<PactRecord> modelRecord, Collector<PactRecord> out)
      throws Exception {

    IncrementalModel baseModel = modelRecord.next().getField(IDX_INPUT2_BASEMODEL, PactIncrementalModel.class).getValue();
    
    PactRecord gainRecord = null;
    int bestDim = 0;
    double bestGain = 0;
    List<FeatureGain> gains = Lists.newArrayList();
    while (gainRecords.hasNext()) {
      gainRecord = gainRecords.next();
      int dim = gainRecord.getField(IDX_INPUT1_DIMENSION, PactInteger.class).getValue();
      double gain = gainRecord.getField(IDX_INPUT1_GAIN, PactDouble.class).getValue();
      gains.add(new FeatureGain(dim, gain));
    }
    Collections.sort(gains, Collections.reverseOrder());
    
    printTopGains(gains, RCV1DatasetInfo.get());
  }
  
  private void printTopGains(List<FeatureGain> gains, DatasetInfo datasetInfo) {
    for (int i=0; i<10 && i<gains.size(); ++i) {
      System.out.println("d " + gains.get(i).getDimension() + 
          " (" + datasetInfo.getFeatureName(gains.get(i).getDimension()) 
          + ") gain: " + gains.get(i).getGain());
    }
  }

}
