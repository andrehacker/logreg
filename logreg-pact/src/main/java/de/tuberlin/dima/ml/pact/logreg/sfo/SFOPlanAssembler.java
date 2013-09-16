package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.IOException;

import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.pact.io.LibsvmInputFormat;
import de.tuberlin.dima.ml.pact.io.SingleValueDataSource;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.ApplyBest;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.EvalComputeLikelihoods;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.EvalSumLikelihoods;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.MatchGainsAndCoefficients;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.TrainComputeProbabilities;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.TrainDimensions;
import de.tuberlin.dima.ml.pact.udfs.CrossTwoToOne;
import de.tuberlin.dima.ml.pact.udfs.ReduceFlattenToVector;
import de.tuberlin.dima.ml.pact.util.PactUtils;
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
import eu.stratosphere.pact.compiler.PactCompiler;
import eu.stratosphere.pact.generic.contract.BulkIteration;
import eu.stratosphere.pact.generic.contract.Contract;

public class SFOPlanAssembler implements PlanAssembler, PlanAssemblerDescription {

  @Override
  public String getDescription() {
    return "Parameters: <numSubStasks> <inputPathTrain> <inputPathTest> <outputPath> <numFeatures> <labelIndex> <iterations> <addPerIteration> <Optional: baseModel (base64 encoded)>";
  }
  
  public static String[] buildArgs(
      int numSubTasks, 
      String inputPathTrain, 
      String inputPathTest, 
      String outputPath, 
      int numFeatures, 
      int labelIndex,
      int iterations,
      int addPerIteration,
      IncrementalModel baseModel) {
    return new String[] {
        Integer.toString(numSubTasks),
        inputPathTrain,
        inputPathTest,
        outputPath,
        Integer.toString(numFeatures),
        Integer.toString(labelIndex),
        Integer.toString(iterations),
        Integer.toString(addPerIteration),
        PactUtils.encodeValueAsBase64(new PactIncrementalModel(baseModel))
    };
  }

  /**
   * TODO _SFO Major: This method currently only supports empty base models.
   * TODO Optional: Transfer basemodel via config (does not work for iterations)
   */
  @Override
  public Plan getPlan(String... args) {
    // The default values just exist to be able to view this job in pact-web
    int numSubTasks = (args.length > 0 ? Integer.parseInt(args[0]) : 1);
    String inputPathTrain = (args.length > 1 ? args[1] : "");
    String inputPathTest = (args.length > 2 ? args[2] : "");
    String outputPath = (args.length > 3 ? args[3] : "");
    int numFeatures = (args.length > 4 ? Integer.parseInt(args[4]) : 0);
    int labelIndex = (args.length > 5 ? Integer.parseInt(args[5]) : 0);
    int iterations = (args.length > 6 ? Integer.parseInt(args[6]) : 1);
    int addPerIteration = (args.length > 7 ? Integer.parseInt(args[7]) : 1);
    IncrementalModel baseModel = (args.length > 8 ? PactUtils.decodeValueFromBase64(args[8], PactIncrementalModel.class).getValue() : new IncrementalModel(numFeatures));
    
    // ----- HINTS / OPTIMIZATION -----

    boolean giveBroadcastHints = true;
    boolean giveFineGradeDopHints = true;

    // ----- Data Sources -----
    
    FileDataSource trainingVectors = new FileDataSource(
        LibsvmInputFormat.class, inputPathTrain, "Training Input Vectors");
    trainingVectors.setParameter(LibsvmInputFormat.CONF_KEY_POSITIVE_CLASS, labelIndex);
    trainingVectors.setParameter(LibsvmInputFormat.CONF_KEY_NUM_FEATURES,
        numFeatures);
    trainingVectors.setParameter(LibsvmInputFormat.CONF_KEY_MULTI_LABEL_INPUT, true);
    
    FileDataSource testVectors = new FileDataSource(
        LibsvmInputFormat.class, inputPathTest, "Test Input Vectors");
    testVectors.setParameter(LibsvmInputFormat.CONF_KEY_POSITIVE_CLASS, labelIndex);
    testVectors.setParameter(LibsvmInputFormat.CONF_KEY_NUM_FEATURES,
        numFeatures);
    testVectors.setParameter(LibsvmInputFormat.CONF_KEY_MULTI_LABEL_INPUT, true);

    // ----- Initial Base Model -----
    
    Contract initialBaseModelContract = null;
    if (baseModel != null && baseModel.getUsedDimensions().size() > 0) {
      try {
        String baseModelTmpPath = "file:///tmp/tmp-base-model";
        initialBaseModelContract = new SingleValueDataSource(new PactIncrementalModel(baseModel), baseModelTmpPath);
      } catch (IOException e) {
        e.printStackTrace();
      }
    } else {
      initialBaseModelContract = new GenericDataSource<EmptyBaseModelInputFormat>(EmptyBaseModelInputFormat.class);
      initialBaseModelContract.setParameter(EmptyBaseModelInputFormat.CONF_KEY_NUM_FEATURES, numFeatures);
    }
    initialBaseModelContract.setName("BaseModel");
    if (giveFineGradeDopHints) {
      initialBaseModelContract.setDegreeOfParallelism(1);
    }

    // ----- Iterations -----
    
    Contract baseModelContract = null;
    BulkIteration iteration = null;
    if (iterations > 1) {
      iteration = new BulkIteration("Forward Feature Selection");
      iteration.setInput(initialBaseModelContract);
      iteration.setMaximumNumberOfIterations(iterations);
      baseModelContract = iteration.getPartialSolution();
    } else {
      baseModelContract = initialBaseModelContract;
    }
    
    // ----- Cross: Train over x -----
    
    CrossContract trainComputeProbabilities = CrossContract.builder(TrainComputeProbabilities.class)
        .input1(trainingVectors)
        .input2(baseModelContract)
        .name("Train: Compute probabilities (Cross)")
        .build();
    if (giveBroadcastHints) {
      trainComputeProbabilities.getParameters().setString(PactCompiler.HINT_SHIP_STRATEGY_FIRST_INPUT,
          PactCompiler.HINT_SHIP_STRATEGY_FORWARD);
      trainComputeProbabilities.getParameters().setString(PactCompiler.HINT_SHIP_STRATEGY_SECOND_INPUT,
          PactCompiler.HINT_SHIP_STRATEGY_BROADCAST);
    }
    
    // ----- Reduce: Train over d -----
    
    ReduceContract trainDimensions = ReduceContract.builder(TrainDimensions.class, PactInteger.class, TrainDimensions.IDX_DIMENSION)
        .input(trainComputeProbabilities)
        .name("Train: Train new Features (Reduce)")
        .build();
    
    // ----- Workaround 1: Flatten Coefficients -----
    
    // Keyless-Reducer now works, but pact-web visualization fails to visualize jobs using the feature:(
//    ReduceContract flattenCoefficients = ReduceContract.builder(ReduceFlattenToVector.class)
    ReduceContract flattenCoefficients = ReduceContract.builder(ReduceFlattenToVector.class, PactInteger.class, ReduceFlattenToVector.IDX_KEY_CONST_ONE)
        .input(trainDimensions)
        .name("Workaround: Flatten trained coefficients (Reduce)")
        .build();
    flattenCoefficients.setParameter(ReduceFlattenToVector.CONF_KEY_NUM_FEATURES, numFeatures);
    if (giveFineGradeDopHints) {
      flattenCoefficients.setDegreeOfParallelism(1);
    }
    
    // ----- Workaround 2: Make 1 out of 2 records -----

    CrossContract basemodelAndCoefficients = CrossContract.builder(CrossTwoToOne.class)
        .input1(baseModelContract)
        .input2(flattenCoefficients)
        .name("Workaround: Flatten two to one (Cross)")
        .build();
    basemodelAndCoefficients.setParameter(CrossTwoToOne.CONF_KEY_IDX_OUT_VALUE1, EmptyBaseModelInputFormat.IDX_OUT_BASEMODEL);
    basemodelAndCoefficients.setParameter(CrossTwoToOne.CONF_KEY_IDX_OUT_VALUE2, ReduceFlattenToVector.IDX_OUT_VECTOR);
    if (giveFineGradeDopHints) {
      basemodelAndCoefficients.setDegreeOfParallelism(1);
    }
    
    // ----- Cross: Eval Compute Likelihoods over records -----

    CrossContract evalComputeLikelihoods = CrossContract.builder(EvalComputeLikelihoods.class)
        .input1(testVectors)
        .input2(basemodelAndCoefficients)
        .name("Eval: Compute likelihoods (Cross)")
        .build();
    if (giveBroadcastHints) {
      evalComputeLikelihoods.getParameters().setString(PactCompiler.HINT_SHIP_STRATEGY_FIRST_INPUT,
          PactCompiler.HINT_SHIP_STRATEGY_FORWARD);
      evalComputeLikelihoods.getParameters().setString(PactCompiler.HINT_SHIP_STRATEGY_SECOND_INPUT,
          PactCompiler.HINT_SHIP_STRATEGY_BROADCAST);
    }
    
    // ----- Reduce: Sum up likelihoods -----

    ReduceContract evalSumUpLikelihoods = ReduceContract
        .builder(EvalSumLikelihoods.class, PactInteger.class, EvalSumLikelihoods.IDX_DIMENSION)
        .input(evalComputeLikelihoods)
        .name("Eval: Sum up likelihoods (Reduce)")
        .build();
    
    // ----- Match Gains & Coefficients -----

    MatchContract matchGainsCoefficients = MatchContract
        .builder(MatchGainsAndCoefficients.class, PactInteger.class,
            EvalSumLikelihoods.IDX_OUT_DIMENSION,
            TrainDimensions.IDX_OUT_DIMENSION)
            .input1(evalSumUpLikelihoods)
            .input2(trainDimensions)
            .name("Match Gains and Coefficients")
            .build();
    matchGainsCoefficients.getCompilerHints().setAvgRecordsEmittedPerStubCall(1);
    
    FileDataSink dataSink = null;
    if (iterations > 1) {
    
      // ----- CoGroup: Sort & Apply best to base model -----
      
      CoGroupContract applyBest = CoGroupContract
          .builder(ApplyBest.class, PactInteger.class,
              MatchGainsAndCoefficients.IDX_OUT_KEY_CONST_ONE,
              EmptyBaseModelInputFormat.IDX_OUT_KEY_CONST_ONE)
          .input1(matchGainsCoefficients)
          .input2(baseModelContract)
          .name("ApplyBest")
          .build();
      applyBest.setParameter(ApplyBest.CONF_KEY_ADD_PER_ITERATION, addPerIteration);
      if (giveFineGradeDopHints) {
        applyBest.setDegreeOfParallelism(1);
      }
      // TODO _SFO: Sorting does not work!
  //    sortAndApplyBest.setGroupOrderForInputOne(new Ordering(EvalSumLikelihoods.IDX_OUT_GAIN,
  //                PactDouble.class, Order.ASCENDING));
      
      iteration.setNextPartialSolution(applyBest);
      
      // ----- Data Sink & Output Format -----
      
      dataSink = new FileDataSink(RecordOutputFormat.class,
          outputPath, iteration, "Output");

      RecordOutputFormat.configureRecordFormat(dataSink).recordDelimiter('\n')
          .fieldDelimiter(' ')
          .field(PactIncrementalModel.class, ApplyBest.IDX_OUT_BASEMODEL);

    } else {
      
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
