package de.tuberlin.dima.ml.pact.logreg.ensemble;

import java.util.ArrayList;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.inputreader.LibSvmVectorReader;
import de.tuberlin.dima.ml.logreg.LogRegEnsembleModel;
import de.tuberlin.dima.ml.logreg.LogRegEnsembleModel.VotingSchema;
import de.tuberlin.dima.ml.pact.types.PactVector;
import eu.stratosphere.nephele.configuration.Configuration;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.CrossStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactInteger;
import eu.stratosphere.pact.common.type.base.PactString;

/**
 * Currently evaluates only a single model.
 * 
 * @author Andre Hacker
 *
 */
public class CrossEval extends CrossStub {
  
  private int numFeatures;
  
  private boolean modelCached = false;
  private LogRegEnsembleModel model = null; // this will be cached in first local run by the udf
  
//  private OnlineAccuracy onlineAccuracy = new OnlineAccuracy(0.5);
  
  private PactRecord recordOut = new PactRecord();
  
  private PactInteger zero = new PactInteger(0);
  private PactInteger one = new PactInteger(1);
  
  @Override
  public void open(Configuration parameters) throws Exception {
    super.open(parameters);
    this.numFeatures = parameters.getInteger(EnsembleJob.CONF_KEY_NUM_FEATURES, 0);
  }

  @Override
  public void cross(PactRecord modelRecord, PactRecord dataRecord,
      Collector<PactRecord> out) throws Exception {
    
    // Read test item
    PactString line = dataRecord.getField(0, PactString.class);
    Vector x = new RandomAccessSparseVector(numFeatures);
    short label = LibSvmVectorReader.readVector(x, line.getValue());

    if (! modelCached) {
      // Read ensemble model
      // TODO Major: Bad that we build the model for every call - should always stay the same!
      int numModels = modelRecord.getField(EnsembleJob.ID_EVAL_IN_NUM_MODELS, PactInteger.class).getValue();
      System.out.println("Num Models: " + numModels);
      ArrayList<Vector> ensembleModels = Lists.newArrayList();
      for (int i=0; i<numModels; ++i) {
        ensembleModels.add(modelRecord.getField(EnsembleJob.ID_EVAL_IN_FIRST_MODEL + i, PactVector.class).getValue());
      }
      model = new LogRegEnsembleModel(ensembleModels, 0.5d, VotingSchema.MAJORITY_VOTE);
      modelCached = true;
    }
    
    double prediction = model.predict(x);
    recordOut.setField(EnsembleJob.ID_EVAL_OUT_MODEL_ID, one);
    recordOut.setField(EnsembleJob.ID_EVAL_OUT_TOTAL, one);
    recordOut.setField(EnsembleJob.ID_EVAL_OUT_CORRECT, (prediction == label)?one:zero);
    out.collect(recordOut);
    
//    onlineAccuracy.addSample(
//        label,
//        prediction);
//    
//    System.out.println("Total: " + onlineAccuracy.getTotal() + " Correct: " + onlineAccuracy.getCorrect());
  }
  
}
