package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFODriver;
import de.tuberlin.dima.ml.mapred.util.HadoopUtils;

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

  private String trainInputFile;
  private String testInputFile;

  private String trainOutputPath;
  private String testOutputPath;
  
  private String jobTrackerAddress;
  private String hdfsAddress;
  private String jarPath;   // might be empty

  private int numFeatures;

  private IncrementalModel baseModel;

  private List<FeatureGain> gains = Lists.newArrayList();

  /**
   * Initializes a new empty base model
   */
  public SFODriverHadoop(
      String trainInputFile,
      String testInputFile,
      String outputPathTrain,
      String outputPathTest,
      int numFeatures,
      String jobTrackerAddress,
      String hdfsAddress,
      String jarPath) {
    this.trainInputFile = trainInputFile;
    this.testInputFile = testInputFile;
    this.trainOutputPath = outputPathTrain;
    this.testOutputPath = outputPathTest;
    this.numFeatures = numFeatures;
    this.jobTrackerAddress = jobTrackerAddress;
    this.hdfsAddress = hdfsAddress;
    this.jarPath = jarPath;

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
  public List<FeatureGain> computeGainsSFO(int dop) throws Exception {

    // Make base model available for train/test mappers
    SFOToolsHadoop.writeBaseModel(baseModel, hdfsAddress);

    // ----- TRAIN -----
    
    Configuration conf = HadoopUtils.createConfiguration(hdfsAddress, jobTrackerAddress, jarPath);
    ToolRunner.run(conf, new SFOTrainJob(trainInputFile, trainOutputPath, dop),
        null);
    System.out.println("Done training");

    // ----- TEST -----
    
    ToolRunner.run(conf, new SFOEvalJob(testInputFile, testOutputPath, dop,
        numFeatures, trainOutputPath), null);
    System.out.println("Done validation");

    // ----- READ RESULTS -----

    gains = SFOToolsHadoop.readEvalResult(testOutputPath, hdfsAddress);
    Collections.sort(gains, Collections.reverseOrder());
    
    return getGains();
  }
  
  @Override
  public void addBestFeature() throws IOException {
    addNBestFeatures(1);
  }

  /**
   * In Forward feature selection we can add multiple features in each
   * iteration.
   * @throws IOException 
   */
  @Override
  public void addNBestFeatures(int n) throws IOException {
    // Read coefficients
    Configuration conf = HadoopUtils.createConfiguration(hdfsAddress, jobTrackerAddress);
    List<Double> coefficients = SFOToolsHadoop.readTrainedCoefficients(conf,
        numFeatures, trainOutputPath);

    // Add best to base model
    for (int i=0; i<n; ++i) {
      int bestDimension = gains.get(i).getDimension();
      baseModel.addDimensionToModel(bestDimension,
          coefficients.get(bestDimension));
      System.out
      .println("Added d=" + bestDimension
          + " to base model with c="
          + coefficients.get(bestDimension));
    }

    // Write updated model to hdfs
    SFOToolsHadoop.writeBaseModel(baseModel, hdfsAddress);

    System.out.println("- New base model: " + baseModel.getW().toString());
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

}
