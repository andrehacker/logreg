package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFODriver;
import de.tuberlin.dima.ml.mapred.GlobalSettings;

/**
 * Implements the Single Feature Optimization Algorithm as proposed by Singh et
 * al. [1] using Hadoop.
 * 
 * REFERENCES [1] Singh, S., Kubica, J., Larsen, S., & Sorokina, D. (2009).
 * Parallel Large Scale Feature Selection for Logistic Regression. Optimization,
 * 1172–1183. Retrieved from http://www.additivegroves.net/papers/fslr.pdf
 * 
 * @author André Hacker
 */
public class SFODriverHadoop implements SFODriver {

  private String inputFile;

  private String trainOutputPath;
  private String testOutputPath;

  private int reducersTrain;
  private int reducersTest;

  private int numFeatures;

  private IncrementalModel baseModel;

  private List<FeatureGain> gains = Lists.newArrayList();

  /**
   * Initializes a new empty base model
   */
  public SFODriverHadoop(String inputFile, String outputPathTrain,
      String outputPathTest, int reducersTrain, int reducersTest,
      int numFeatures) {
    this.inputFile = inputFile;
    this.trainOutputPath = outputPathTrain;
    this.testOutputPath = outputPathTest;
    this.reducersTrain = reducersTrain;
    this.reducersTest = reducersTest;
    this.numFeatures = numFeatures;

    // Create empty model
    baseModel = new IncrementalModel(numFeatures);
  }

  /**
   * Runs a single SFO-Iteration. Computes the gain in metric (e.g.
   * log-likelihood) for all possible models with one more feature added to the
   * current base model
   * 
   * Does not add the best feature to the base model, allows the client to
   * decide whether to add it or not.
   */
  @Override
  public void runSFO() throws Exception {

    // Make base model available for train/test mappers
    SFOToolsHadoop.writeBaseModel(baseModel);

    // ----- TRAIN -----
    ToolRunner.run(new SFOTrainJob(inputFile, trainOutputPath, reducersTrain),
        null);

    System.out.println("Done training");

    // ----- TEST -----
    ToolRunner.run(new SFOEvalJob(inputFile, testOutputPath, reducersTest,
        numFeatures, trainOutputPath), null);
    System.out.println("Done validation");

    // ----- Postprocess -----
    readGains();
  }

  @Override
  public void addBestFeature() throws IOException {
    // Read coefficients
    // TODO Minor Optimization: Don't read all into memory, just search the single
    // coefficient
    Configuration conf = new Configuration();
    conf.addResource(new Path(GlobalSettings.CONFIG_FILE_PATH));
    List<Double> coefficients = SFOToolsHadoop.readTrainedCoefficients(conf,
        numFeatures, trainOutputPath);

    // Add best to base model
    int bestDimension = gains.get(0).getDimension();
    baseModel.addDimensionToModel(bestDimension,
        coefficients.get(bestDimension));

    // Write updated model to hdfs
    SFOToolsHadoop.writeBaseModel(baseModel);

    System.out
        .println("Added dimension " + bestDimension
            + " to base model with coefficient: "
            + coefficients.get(bestDimension));
    System.out.println("- New base model: " + baseModel.getW().toString());
  }

  /**
   * In Forward feature selection we can add multiple features in each
   * iteration.
   */
  @Override
  public void addNBestFeatures(int n) {
    // TODO Implement add n best features!
    throw new UnsupportedOperationException("Not yet implemented");
  }

  /**
   * After adding n features to the base model (based on approximate computation
   * of the gain) we retrain the whole model.
   * 
   * "After evaluating them to choose the best feature (or group of features) to
   * add, we rerun the full logistic regression to produce an exact model that
   * includes the newly added feature(s). We begin with this exact model for the
   * next iteration of features selection, so the approximation error does not
   * add up."
   */
  @Override
  public void retrainBaseModel() {
    // TODO Major: Retrain Base Model!
    System.out.println("Retraining base model not yet implemented");
  }

  @Override
  public List<FeatureGain> getGains() {
    return gains;
  }

  private void readGains() throws IOException {
    // Read results from hdfs into memory
    gains = SFOToolsHadoop.readEvalResult(testOutputPath);

    // Sort by gain
    Collections.sort(gains, Collections.reverseOrder());
  }

}
