package de.tuberlin.dima.experiments.ml.experiments

import java.text.SimpleDateFormat
import java.util.Calendar
import scala.collection.JavaConversions.mapAsScalaMap
import scala.collection.JavaConversions.seqAsJavaList
import org.slf4j.LoggerFactory
import de.tuberlin.dima.experiments.Experiment
import de.tuberlin.dima.experiments.HadoopSUT
import de.tuberlin.dima.experiments.OzoneSUT
import de.tuberlin.dima.experiments.SUT
import de.tuberlin.dima.ml.datasets.DatasetInfo
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo
import de.tuberlin.dima.ml.logreg.sfo.FeatureGain
import de.tuberlin.dima.ml.logreg.sfo.SFODriver
import de.tuberlin.dima.ml.mapred.logreg.sfo.SFODriverHadoop
import de.tuberlin.dima.ml.pact.logreg.sfo.SFODriverPact
import com.google.common.base.Throwables
import scala.collection.JavaConversions
import org.apache.hadoop.conf.Configuration
import de.tuberlin.dima.ml.mapred.util.HadoopUtils
import org.apache.hadoop.fs.FileSystem
import com.google.common.base.Stopwatch
import java.util.concurrent.TimeUnit

object SFOExperiment extends Experiment {
  
  private val logger = LoggerFactory.getLogger(this.getClass())
  
  def main(args: Array[String]) {
    runExperiment(args)
  }
  
  /**
   * DON'T FORGET:
   * - Compile job jar first!
   * - Create new SUT deployment package if SUT changed (e.g. changed hadoop-dependencies for ozone)
   * - Make sure that ozone files in local repository are pointing to your hadoop version
   *   - Compile ozone with the hdfs dependencies you use (central ozone pom contains cloudera repository)
   * -Make sure that HADOOP_PREFIX and HADOOP_YARN_HOME does either not exist or point to correct directory (otherwise startup scripts call wrong scripts) 
   */
  def runExperiment(args: Array[String]) = {
    
    try {
    
      val sysConfPath = if (args.length>=2) { args(0) }
          else {
            "/home/andre/dev/logreg-repo/logreg-experiments/conf-templates/andre-sam-ubuntu/sysconf-hadoop-2.1.0-beta.properties" 
          }
      val experimentConfPath = if (args.length>=2) args(1)
          else "/home/andre/dev/logreg-repo/logreg-experiments/sfo-experiment-andre-sam-ubuntu.properties";
      
      init(sysConfPath, experimentConfPath)
      
  
      // --------------- EXPERIMENT PARAMETERS ----------
      
      val experimentName = getProperty("experiment_name")
      val currentSut = getProperty("sut")
  
      // --------------- JOB PARAMETERS ----------
      
//      val dataset = RCV1DatasetInfo.get()
//      val predictorNamePath = "/home/andre/dev/datasets/RCV1-v2/stem.termid.idf.map.txt"
//      RCV1DatasetInfo.readPredictorNames(predictorNamePath)
  
      // --------------- JOB PARAMETERS GLOBAL ----------
      val newtonTolerance = getProperty("newton_tolerance").toDouble
      val newtonMaxIterations = getProperty("newton_max_iterations").toInt
      val regularization = getProperty("regularization").toDouble
  
      // --------------- JOB PARAMETERS HADOOP ----------
  
      val jarPathHadoop = getProperty("jar_hadoop")
      val inputTrainLocalHadoop = getProperty("input_local_hadoop")
      val inputTrainHadoop = getProperty("input_hadoop")
      val inputTestLocalHadoop = getProperty("input_test_local_hadoop")
      val inputTestHadoop = getProperty("input_test_hadoop")
      val isMultilabelInputHadoop = getProperty("input_hadoop_is_multilabel").toBoolean
      val positiveClassHadoop = if (isMultilabelInputHadoop) getProperty("positive_class_hadoop").toInt else -1
      val numFeaturesHadoop = getProperty("num_features_hadoop").toInt
      val outputTrainHadoop = getProperty("output_train_hadoop")
      val outputTestHadoop = getProperty("output_test_hadoop")
      val intraNodeDopHadoop = getProperty("intra_node_dop_hadoop").toInt
  
      // --------------- JOB PARAMETERS OZONE ----------
      
      val jarPathOzone = getProperty("jar_ozone")
      val inputTrainLocalOzone = getProperty("input_local_ozone")
      val inputTrainOzone = getProperty("input_ozone")
      val inputTestLocalOzone = getProperty("input_test_local_ozone")
      val inputTestOzone = getProperty("input_test_ozone")
      val isMultilabelInputOzone = getProperty("input_ozone_is_multilabel").toBoolean
      val positiveClassOzone = if (isMultilabelInputOzone) getProperty("positive_class_ozone").toInt else -1
      val numFeaturesOzone = getProperty("num_features_ozone").toInt
      val outputOzone = getProperty("output_ozone")
      val intraNodeDopOzone = getProperty("intra_node_dop_ozone").toInt
      
      // --------------- JOB DRIVER ----------
  
      val hadoopConfPath = getSysProperty("hadoop_conf")
      
      val sfoDriverHadoop = new SFODriverHadoop(
        inputTrainHadoop,
        inputTestHadoop,
        isMultilabelInputHadoop,
        positiveClassHadoop,
        outputTrainHadoop,
        outputTestHadoop,
        numFeaturesHadoop,
        newtonTolerance,
        newtonMaxIterations,
        regularization,
        hadoopConfPath,
        jarPathHadoop)
      
      val ozoneConfPath = getSysProperty("ozone_conf")
  
      //  Here we use the conf-path currently
      val runLocal = false
      val sfoDriverPact = new SFODriverPact(
        inputTrainOzone,
        inputTestOzone,
        isMultilabelInputOzone,
        positiveClassOzone,
        outputOzone,
        numFeaturesOzone,
        newtonTolerance,
        newtonMaxIterations,
        regularization,
        runLocal,
        ozoneConfPath,
        jarPathOzone)
      
      // --------------- EXPERIMENT ---------------
      
      val dops = getPropertyArrayAsInt("dops")
      val numRepetitions = getProperty("repetitions").toInt
      val driverIterations = getProperty("driver_iterations").toInt
      val iterations = getProperty("iterations").toInt
      val addPerIteration = getProperty("add_per_iteration").toInt
      val datasetName = getProperty("dataset_name")
      val experimentPrefix = "%s-%s-%s-%s".format(experimentName, currentSut, datasetName, getDate("yyyy-MM-dd-HHmmss"))
  
      if (currentSut == "hadoop") {
        val hadoop = new HadoopSUT(sysConfPath)
        
        val dataToLoadHadoop = Array((inputTrainLocalHadoop, inputTrainHadoop), (inputTestLocalHadoop, inputTestHadoop))
        val logFilesToBackupHadoop = Array((outputTrainHadoop, "sfo-train"), (outputTestHadoop, "sfo-test"))
        val outputToRemoveHadoop = Array(outputTrainHadoop, outputTestHadoop)
        runExperimentSingleSUT(hadoop, dops, intraNodeDopHadoop, driverIterations, iterations, addPerIteration, numRepetitions, dataToLoadHadoop, outputToRemoveHadoop, logFilesToBackupHadoop, sfoDriverHadoop, experimentPrefix)
      }
      
      if (currentSut == "ozone") {
        val ozone = new OzoneSUT(sysConfPath)
        
        val dataToLoadPact = Array((inputTrainLocalOzone, inputTrainOzone),(inputTestLocalOzone, inputTestOzone))
        val logFilesToBackupPact = Array((outputOzone, "sfo"))
        val outputToRemovePact = Array(outputOzone)
        runExperimentSingleSUT(ozone, dops, intraNodeDopOzone, driverIterations, iterations, addPerIteration, numRepetitions, dataToLoadPact, outputToRemovePact, logFilesToBackupPact, sfoDriverPact, experimentPrefix)
      }
      
    } catch {
      case ex: Throwable => logger.error("Exception: " + Throwables.getStackTraceAsString(ex))
      false
    }
  }
  
  def runExperimentSingleSUT(
      sut: SUT, 
      dops: Array[Int],
      intraNodeDop: Int,
      driverIterations: Int,
      iterations: Int,
      addPerIteration: Int,
      numRepetitions: Int, 
      dataToLoad: Array[(String, String)], 
      outputToRemove: Array[String],
      logFilesToBackup: Array[(String, String)],
      jobDriver: SFODriver,
      experimentPrefix: String) = {
    
    val deploySut = getProperty("deploy_sut").toBoolean
    val startSut = getProperty("start_sut").toBoolean
    val stopSut = getProperty("stop_sut").toBoolean
    val runExperiments = getProperty("run_experiments").toBoolean   // only makes sense if a single dop is defined
    
    if (deploySut) sut.deploy()
    
    for (dop <- dops) {
      
      if (startSut) {
        sut.adaptSlaves(dop)
        sut.fsFormatStartWait(dop)
        sut.startWait(dop)
      }
      
      if (runExperiments) {
        for ((src, target) <- dataToLoad) {
          sut.fsLoadData(src, target)
        }
        
        for (rep <- 1 to numRepetitions) {
          
          val experimentID = experimentPrefix + "-dop%04d-run%02d".format(dop, rep)
          
          for (outputFolder <- outputToRemove) {
            sut.removeOutputFolder(outputFolder)
          }
          
          jobDriver.resetModel();
          logger.info("-------------------- RUN EXPERIMENT --------------------\n")
          if (iterations <= 1) {
            val stopDriver = new Stopwatch().start()
            for (i <- 1 to driverIterations) {
              jobDriver.computeGains(dop * intraNodeDop)
              printTopGains(JavaConversions.asScalaBuffer(jobDriver.getGains()))
              jobDriver.addBestFeatures(addPerIteration)
              logTimers(jobDriver, experimentID)
              
              for (outputFolder <- outputToRemove) {
                sut.removeOutputFolder(outputFolder)
              }
            }
            stopDriver.stop()
            if (driverIterations > 1) logger.info("Time for all iterations: " + stopDriver.elapsed(TimeUnit.MILLISECONDS))
          } else {
            jobDriver.forwardFeatureSelection(dop * intraNodeDop, iterations, addPerIteration)
            logTimers(jobDriver, experimentID)
          }
  
          for ((outputFolder, logName) <- logFilesToBackup) {
            sut.backupJobLogs(outputFolder, experimentID, "job-logs-" + logName)
          }
        }
      }
      
      if (stopSut) {
        sut.stop()
        sut.fsCleanStop()
      }
      
    }
  }
  
  def logTimers(driver: SFODriver, description: String) = {
    logger.info("--------------------------------------------------")
    logger.info("Statistics for: " + description)
    logger.info("WALL CLOCK TIME: " + driver.getLastWallClockTime())
    logger.info("Additional timers")
    for ((key, value) <- driver.getAllCounters()) {
      logger.info(" - " + key + ": " + value)
    }
    logger.info("--------------------------------------------------")
  }
  
  def getDate(format: String) = {
    val df = new SimpleDateFormat(format)
    val today = Calendar.getInstance().getTime()
    df.format(today)
  }
  
  // , datasetInfo: DatasetInfo
  def printTopGains(gains: scala.collection.mutable.Buffer[FeatureGain]) = {
    val limit = 100
    for (i <- 0 until limit) {
      logger.info("d " + gains.get(i).getDimension() 
          + " gain: " + gains.get(i).getGain() + " coefficient(pact-only): " + gains.get(i).getCoefficient())
      // + " (" + datasetInfo.getFeatureName(gains.get(i).getDimension())+ ")"
    }
  }
}
