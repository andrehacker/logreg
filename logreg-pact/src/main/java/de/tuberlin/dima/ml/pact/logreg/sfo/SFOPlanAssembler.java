package de.tuberlin.dima.ml.pact.logreg.sfo;

import de.tuberlin.dima.ml.pact.io.LibsvmBinaryInputFormat;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.ApplyBest;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.EvalComputeLikelihoods;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.EvalSumLikelihoods;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.TrainComputeProbabilities;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.TrainNewFeatures;
import de.tuberlin.dima.ml.pact.udfs.CrossTwoToOne;
import de.tuberlin.dima.ml.pact.udfs.ReduceFlattenToVector;
import eu.stratosphere.pact.common.contract.CoGroupContract;
import eu.stratosphere.pact.common.contract.CrossContract;
import eu.stratosphere.pact.common.contract.FileDataSink;
import eu.stratosphere.pact.common.contract.FileDataSource;
import eu.stratosphere.pact.common.contract.GenericDataSource;
import eu.stratosphere.pact.common.contract.ReduceContract;
import eu.stratosphere.pact.common.io.RecordOutputFormat;
import eu.stratosphere.pact.common.plan.Plan;
import eu.stratosphere.pact.common.type.base.PactDouble;
import eu.stratosphere.pact.common.type.base.PactInteger;

public class SFOPlanAssembler {
  
  public Plan createPlan(
      int numSubTasks, 
      String inputPathTrain, 
      String inputPathTest, 
      String outputPath, 
      int numFeatures, 
      int labelIndex,
      boolean applyBest) {
    
    // ----- Data Sources -----
    
    FileDataSource trainingVectors = new FileDataSource(
        LibsvmBinaryInputFormat.class, inputPathTrain, "Training Input Vectors");
    trainingVectors.setParameter(LibsvmBinaryInputFormat.CONF_KEY_POSITIVE_CLASS, labelIndex);
    trainingVectors.setParameter(LibsvmBinaryInputFormat.CONF_KEY_NUM_FEATURES,
        numFeatures);
    
    FileDataSource testVectors = new FileDataSource(
        LibsvmBinaryInputFormat.class, inputPathTest, "Test Input Vectors");
    testVectors.setParameter(LibsvmBinaryInputFormat.CONF_KEY_POSITIVE_CLASS, labelIndex);
    testVectors.setParameter(LibsvmBinaryInputFormat.CONF_KEY_NUM_FEATURES,
        numFeatures);

    // ----- Base Model -----
    
    GenericDataSource<EmptyBaseModelInputFormat> initialBaseModel = new GenericDataSource<EmptyBaseModelInputFormat>(EmptyBaseModelInputFormat.class);
    initialBaseModel.setParameter(EmptyBaseModelInputFormat.CONF_KEY_NUM_FEATURES, numFeatures);

    // ----- Cross: Train over x -----
    
    CrossContract trainComputeProbabilities = CrossContract.builder(TrainComputeProbabilities.class)
        .input1(trainingVectors)
        .input2(initialBaseModel)
        .name("Train: Compute probabilities (Cross)")
        .build();
    
    // ----- Reduce: Train over d -----
    
    ReduceContract trainNewFeatures = ReduceContract.builder(TrainNewFeatures.class, PactInteger.class, TrainNewFeatures.IDX_DIMENSION)
        .input(trainComputeProbabilities)
        .name("Train: Train new Features (Reduce)")
        .build();
    
    // ----- Workaround 1: Flatten Coefficients -----
    
    ReduceContract flattenCoefficients = ReduceContract.builder(ReduceFlattenToVector.class, PactInteger.class, ReduceFlattenToVector.IDX_KEY_CONST_ONE)
        .input(trainNewFeatures)
        .name("Flatten trained coefficients (Reduce)")
        .build();
    flattenCoefficients.setParameter(ReduceFlattenToVector.CONF_KEY_NUM_FEATURES, numFeatures);
    
    // ----- Workaround 2: Make 1 out of 2 records -----

    CrossContract basemodelAndCoefficients = CrossContract.builder(CrossTwoToOne.class)
        .input1(initialBaseModel)
        .input2(flattenCoefficients)
        .name("Flatten two to one (Cross)")
        .build();
    basemodelAndCoefficients.setParameter(CrossTwoToOne.CONF_KEY_IDX_OUT_VALUE1, EmptyBaseModelInputFormat.IDX_OUT_BASEMODEL);
    basemodelAndCoefficients.setParameter(CrossTwoToOne.CONF_KEY_IDX_OUT_VALUE2, ReduceFlattenToVector.IDX_OUT_VECTOR);
    
    // ----- Cross: Eval Compute Likelihoods over records -----

    CrossContract evalComputeLikelihoods = CrossContract.builder(EvalComputeLikelihoods.class)
        .input1(testVectors)
        .input2(basemodelAndCoefficients)
        .name("Eval: Compute likelihoods (Cross)")
        .build();
    
    // ----- Reduce: Sum up likelihoods -----

    ReduceContract evalSumUpLikelihoods = ReduceContract
        .builder(EvalSumLikelihoods.class, PactInteger.class, EvalSumLikelihoods.IDX_DIMENSION)
        .input(evalComputeLikelihoods)
        .name("Eval: Sum up likelihoods (Reduce)")
        .build();
    
    FileDataSink dataSink = null;
    if (applyBest) {
    
      // ----- CoGroup: Sort & Apply best to base model -----
      
      CoGroupContract sortAndApplyBest = CoGroupContract
          .builder(ApplyBest.class, PactInteger.class,
              EvalSumLikelihoods.IDX_OUT_KEY_CONST_ONE,
              EmptyBaseModelInputFormat.IDX_OUT_KEY_CONST_ONE)
          .input1(evalSumUpLikelihoods)
          .input2(initialBaseModel)
          .name("ApplyBest")
          .build();
      // TODO: This does not work!
  //    sortAndApplyBest.setGroupOrderForInputOne(new Ordering(EvalSumLikelihoods.IDX_OUT_GAIN,
  //                PactDouble.class, Order.ASCENDING));
      
      // ----- Data Sink & Output Format -----
      
      dataSink = new FileDataSink(RecordOutputFormat.class,
          outputPath, sortAndApplyBest, "Output");

      RecordOutputFormat.configureRecordFormat(dataSink).recordDelimiter('\n')
          .fieldDelimiter(' ')
          .field(PactInteger.class, 0)
          .field(PactDouble.class, 1);

    } else {
      
      // ----- Data Sink & Output Format -----
      
      dataSink = new FileDataSink(RecordOutputFormat.class,
          outputPath, evalSumUpLikelihoods, "Output");

      RecordOutputFormat.configureRecordFormat(dataSink).recordDelimiter('\n')
          .fieldDelimiter(' ')
          .field(PactInteger.class, EvalSumLikelihoods.IDX_OUT_DIMENSION)
          .field(PactDouble.class, EvalSumLikelihoods.IDX_OUT_GAIN);
    }

    // ----- Plan -----

    Plan plan = new Plan(dataSink, "BatchGD Plan");
    plan.setDefaultParallelism(numSubTasks);
    
    return plan;
  }

}
