package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.IOException;

import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.pact.io.LibsvmBinaryInputFormat;
import de.tuberlin.dima.ml.pact.io.SingleValueDataSource;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.ApplyBest;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.EvalComputeLikelihoods;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.EvalSumLikelihoods;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.MatchGainsAndCoefficients;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.TrainComputeProbabilities;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.TrainDimensions;
import de.tuberlin.dima.ml.pact.udfs.CrossTwoToOne;
import de.tuberlin.dima.ml.pact.udfs.ReduceFlattenToVector;
import eu.stratosphere.pact.common.contract.CoGroupContract;
import eu.stratosphere.pact.common.contract.CrossContract;
import eu.stratosphere.pact.common.contract.FileDataSink;
import eu.stratosphere.pact.common.contract.FileDataSource;
import eu.stratosphere.pact.common.contract.GenericDataSource;
import eu.stratosphere.pact.common.contract.MatchContract;
import eu.stratosphere.pact.common.contract.ReduceContract;
import eu.stratosphere.pact.common.io.RecordOutputFormat;
import eu.stratosphere.pact.common.plan.Plan;
import eu.stratosphere.pact.common.plan.PlanAssembler;
import eu.stratosphere.pact.common.plan.PlanAssemblerDescription;
import eu.stratosphere.pact.common.type.base.PactDouble;
import eu.stratosphere.pact.common.type.base.PactInteger;
import eu.stratosphere.pact.generic.contract.Contract;

public class SFOPlanAssembler implements PlanAssembler, PlanAssemblerDescription {

  @Override
  public String getDescription() {
    return "Parameters: <numSubStasks> <inputPathTrain> <inputPathTest> <outputPath> <numFeatures> <labelIndex> <applyBest>";
  }

  /**
   * TODO _SFO Major: This method currently only supports empty base models - no idea how to resolve this!?
   */
  @Override
  public Plan getPlan(String... args) {
    int numArgs = 7;
    if (args.length < numArgs) throw new RuntimeException("You didn't pass all required arguments");
    Plan plan = null;
    try {
      plan = createPlan(
          Integer.parseInt(args[0]),
          args[1],
          args[2],
          args[3],
          Integer.parseInt(args[4]),
          Integer.parseInt(args[5]),
          Boolean.parseBoolean(args[6]),
          new IncrementalModel(Integer.parseInt(args[4])));
    } catch (NumberFormatException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return plan;
  }
  
  public static String[] buildArgs(
      int numSubTasks, 
      String inputPathTrain, 
      String inputPathTest, 
      String outputPath, 
      int numFeatures, 
      int labelIndex,
      boolean applyBest) {
    return new String[] {
        Integer.toString(numSubTasks),
        inputPathTrain,
        inputPathTest,
        outputPath,
        Integer.toString(numFeatures),
        Integer.toString(labelIndex),
        Boolean.toString(applyBest)
    };
  }
  
  public Plan createPlan(
      int numSubTasks, 
      String inputPathTrain, 
      String inputPathTest, 
      String outputPath, 
      int numFeatures, 
      int labelIndex,
      boolean applyBest,
      IncrementalModel baseModel) throws IOException {
    
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
    String baseModelTmpPath = "file:///tmp/tmp-base-model";

    Contract baseModelSource = null;
    if (baseModel != null && baseModel.getUsedDimensions().size() > 0) {
      baseModelSource = new SingleValueDataSource(new PactIncrementalModel(baseModel), baseModelTmpPath);
    } else {
      baseModelSource = new GenericDataSource<EmptyBaseModelInputFormat>(EmptyBaseModelInputFormat.class);
      baseModelSource.setParameter(EmptyBaseModelInputFormat.CONF_KEY_NUM_FEATURES, numFeatures);
    }

    // ----- Cross: Train over x -----
    
    CrossContract trainComputeProbabilities = CrossContract.builder(TrainComputeProbabilities.class)
        .input1(trainingVectors)
        .input2(baseModelSource)
        .name("Train: Compute probabilities (Cross)")
        .build();
    
    // ----- Reduce: Train over d -----
    
    ReduceContract trainDimensions = ReduceContract.builder(TrainDimensions.class, PactInteger.class, TrainDimensions.IDX_DIMENSION)
        .input(trainComputeProbabilities)
        .name("Train: Train new Features (Reduce)")
        .build();
    
    // ----- Workaround 1: Flatten Coefficients -----
    
    ReduceContract flattenCoefficients = ReduceContract.builder(ReduceFlattenToVector.class, PactInteger.class, ReduceFlattenToVector.IDX_KEY_CONST_ONE)
        .input(trainDimensions)
        .name("Flatten trained coefficients (Reduce)")
        .build();
    flattenCoefficients.setParameter(ReduceFlattenToVector.CONF_KEY_NUM_FEATURES, numFeatures);
    
    // ----- Workaround 2: Make 1 out of 2 records -----

    CrossContract basemodelAndCoefficients = CrossContract.builder(CrossTwoToOne.class)
        .input1(baseModelSource)
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
          .input2(baseModelSource)
          .name("ApplyBest")
          .build();
      // TODO _SFO: Sorting does not work!
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
      
      // ----- Match Gains & Coefficients -----

      MatchContract matchGainsCoefficients = MatchContract
          .builder(MatchGainsAndCoefficients.class, PactInteger.class,
              EvalSumLikelihoods.IDX_OUT_DIMENSION,
              TrainDimensions.IDX_OUT_DIMENSION)
              .input1(evalSumUpLikelihoods)
              .input2(trainDimensions)
              .name("Match Gains and Coefficients")
              .build();
      
      // ----- Data Sink & Output Format -----
      
      dataSink = new FileDataSink(RecordOutputFormat.class,
          outputPath, matchGainsCoefficients, "Output");

      RecordOutputFormat.configureRecordFormat(dataSink).recordDelimiter('\n')
          .fieldDelimiter(' ')
          .field(PactInteger.class, MatchGainsAndCoefficients.IDX_OUT_DIMENSION)
          .field(PactDouble.class, MatchGainsAndCoefficients.IDX_OUT_GAIN)
          .field(PactDouble.class, MatchGainsAndCoefficients.IDX_OUT_COEFFICIENT);
    }

    // ----- Plan -----

    Plan plan = new Plan(dataSink, "BatchGD Plan");
    plan.setDefaultParallelism(numSubTasks);
    
    return plan;
  }

}
