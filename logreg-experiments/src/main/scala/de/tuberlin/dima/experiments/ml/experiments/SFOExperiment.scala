package de.tuberlin.dima.experiments.ml.experiments

import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import scala.collection.JavaConversions._
import de.tuberlin.dima.experiments.HadoopSUT
import de.tuberlin.dima.experiments.OzoneSUT
import de.tuberlin.dima.experiments.SUT
import de.tuberlin.dima.ml.datasets.DatasetInfo
import de.tuberlin.dima.ml.datasets.DatasetInfo
import de.tuberlin.dima.ml.datasets.DatasetInfo
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo
import de.tuberlin.dima.ml.logreg.sfo.FeatureGain
import de.tuberlin.dima.ml.logreg.sfo.FeatureGain
import de.tuberlin.dima.ml.logreg.sfo.FeatureGain
import de.tuberlin.dima.ml.logreg.sfo.SFODriver
import de.tuberlin.dima.ml.logreg.sfo.SFODriver
import de.tuberlin.dima.ml.logreg.sfo.SFODriver
import de.tuberlin.dima.ml.mapred.logreg.sfo.SFODriverHadoop
import de.tuberlin.dima.ml.pact.logreg.sfo.SFODriverPact
import eu.stratosphere.nephele.configuration.ConfigConstants
import de.tuberlin.dima.experiments.Experiment

object SFOExperiment extends Experiment {
  
  def main(args: Array[String]) {
    runExperiment(args)
  }
  
  /**
   * Prerequisites:
   * - passwordless ssh access from executing computer to slaves (or other way round?)
   * - Download archives
   * - Create config file templates and all-slaves file.
   *   - See http://archive.cloudera.com/cdh4/cdh/4/hadoop/hadoop-project-dist/hadoop-common/DeprecatedProperties.html for new properties
   * - Relies of a nfs share for cluster deployment
   * - Build jar file for experiments
   * - Compile ozone with the hdfs dependencies you use (central ozone pom contains cloudera repository)
   *   - See http://www.cloudera.com/content/cloudera-content/cloudera-docs/CDH4/latest/CDH4-Release-Notes/cdh4rn_topic_2.html for CDH4
   * - Make sure that HADOOP_PREFIX and HADOOP_YARN_HOME does either not exist or point to correct directory (otherwise startup scripts call wrong scripts) 
   * 
   * ATTENTION
   * - Compile job jar first!
   * - Create new SUT deployment package if SUT changed (e.g. changed hadoop-dependencies for ozone)
   * 
   * If you run with yarn
   * - Ozone is configured for CDH4.2.1 (yarn)
   *   - recompile with regular hadoop 2.0.5-alpha
   * - Mahout has to be compiled with yarn
   *   - change mahout version to distinguish from other!
   *   - change 
   *   - Only BuildTools, Math and Core will compile, integration fails due to hbase dependency 
   *   - run mvn install for needed tools
   *   - Alternative: set -Dhadoop.profile=2.0 (see what profile mahout uses)
   * - Own Modules:
   *   - Change all references to yarn
   *   - Change mahout version to 0.8-yarn
   * 
   * If you run with Hadoop MapRedv1 (1.2.* or 1.1.*)
   * - Compile Ozone with old hadoop references (only dependency to hadoop-core)
   *   - Change pom of ozone, nephele-hdfs and pact-runtime
   * - Compile Mahout with hadoop-core 1.2.1. Change artifact-id!
   * 
   * Ozone uses
   * - repo url: https://repository.cloudera.com/artifactory/cloudera-repos
   * - repo id: cloudera-releases
   * - hadoop-common 2.0.0-cdh4.2.1
   * - hadoop-hdfs 2.0.0-cdh4.2.1
   * - CDH4.2.1 is based on 2.0.0-alpha, but contains a ton of patches from later releases on top of it;-)
   * 
   * Measure time
   * - TODO Measure time
   * - https://database.cs.brown.edu/svn/mr-benchmarks/ how they measure the time
   *   - time -f %e hadoop jar ... 
   * - jp-scripts
   *   - startTS=`date +%s`
   *   - hadoop jar ...
   *   - endTS=`date +%s`
   *   - (( jobDuration=$endTS - $startTS ))
   * 
   * TODO Drop cache??? This is what stonebraker does in flushCache
   */
  def runExperiment(args: Array[String]) = {
    
    val sysConfPath = if (args.length>=2) args(0)
        else "/home/andre/dev/logreg-repo/logreg-experiments/conf-templates/andre-sam-ubuntu/sysconf-hadoop-1.2.1.properties"
    val experimentConfPath = if (args.length>=2) args(1)
        else "/home/andre/dev/logreg-repo/logreg-experiments/sfo-experiment-andre-sam-ubuntu.properties";
    
    init(sysConfPath, experimentConfPath)
    
    val experimentName = getProperty("experiment_name")
    
    val currentSut = getProperty("sut")

    // --------------- JOB PARAMETERS ----------
    
    val dataset = RCV1DatasetInfo.get()
//    val predictorNamePath = "/home/andre/dev/datasets/RCV1-v2/stem.termid.idf.map.txt"
//    RCV1DatasetInfo.readPredictorNames(predictorNamePath)

    // --------------- JOB PARAMETERS HADOOP ----------

    val jarPathHadoop = getProperty("jar_hadoop")
    val inputTrainLocalHadoop = getProperty("input_local_hadoop")
    val inputTrainHadoop = getProperty("input_hadoop")
    val outputTrainHadoop = getProperty("output_train_hadoop")
    val outputTestHadoop = getProperty("output_test_hadoop")

    // --------------- JOB PARAMETERS OZONE ----------
    
    val jarPathOzone = getProperty("jar_ozone")
    val inputTrainLocalOzone = getProperty("input_local_ozone")
    val inputTrainOzone = getProperty("input_ozone")
    val labelIndex = getProperty("label_index_ozone").toInt
    val outputOzone = getProperty("output_ozone")
    
    // --------------- JOB DRIVER ----------

    val jobTrackerAddress = getSysProperty("hadoop_jobtracker_address")
    val hdfsAddress = getSysProperty("hdfs_address")

    val sfoDriverHadoop = new SFODriverHadoop(
      inputTrainHadoop,
      inputTrainHadoop,
      outputTrainHadoop,
      outputTestHadoop,
      dataset.getNumFeatures().toInt,
      jobTrackerAddress,
      hdfsAddress,
      jarPathHadoop)
    println("jarPathHadoop: " + jarPathHadoop)
    
    val jobManagerAddress = getSysProperty("job_manager_address")
    val jobManagerPort = getSysProperty("job_manager_port")
    val ozoneConfPath = getSysProperty("ozone_conf")

    val sfoDriverPact = new SFODriverPact(
      inputTrainOzone,
      inputTrainOzone,
      labelIndex,
      outputOzone,
      dataset.getNumFeatures().toInt,
      false,
      ozoneConfPath,
      jarPathOzone,
      jobManagerAddress,
      jobManagerPort.toString)
    
    // --------------- EXPERIMENT ---------------
    
    val dops = getPropertyArrayAsInt("dops")
    val numRepetitions = getProperty("repetitions").toInt
    val addPerIteration = getProperty("add_per_iteration").toInt
    val datasetName = getProperty("dataset_name")
    val experimentPrefix = "%s-%s-%s-%s".format(experimentName, currentSut, datasetName, getDate("yyyy-MM-dd-HHmmss"))

    if (currentSut == "hadoop") {
      val hadoop = new HadoopSUT(sysConfPath)
      
      val dataToLoadHadoop = Array((inputTrainLocalHadoop, inputTrainHadoop))
      val logFilesToBackupHadoop = Array(outputTrainHadoop, outputTestHadoop)
      val outputToRemoveHadoop = Array(outputTrainHadoop, outputTestHadoop)
      runExperimentSingleSUT(hadoop, dops, numRepetitions, dataToLoadHadoop, outputToRemoveHadoop, logFilesToBackupHadoop, sfoDriverHadoop, experimentPrefix)
    }
    
    if (currentSut == "ozone") {
      val ozone = new OzoneSUT(sysConfPath)
      
      val dataToLoadPact = Array((inputTrainLocalOzone, inputTrainOzone))
      val logFilesToBackupPact = Array(outputOzone)
      val outputToRemovePact = Array(outputOzone)
      runExperimentSingleSUT(ozone, dops, numRepetitions, dataToLoadPact, outputToRemovePact, logFilesToBackupPact, sfoDriverPact, experimentPrefix)
    }
  }
  
  def runExperimentSingleSUT(
      sut: SUT, 
      dops: Array[Int], 
      numRepetitions: Int, 
      dataToLoad: Array[(String, String)], 
      outputToRemove: Array[String],
      logFilesToBackup: Array[String],
      jobDriver: SFODriver,
      experimentPrefix: String) = {
    
    sut.deploy()
    
    for (dop <- dops) {
      
      sut.adaptSlaves(dop)
  
      sut.fsFormatStartWait(dop)
      
      for ((src, target) <- dataToLoad) {
        sut.fsLoadData(src, target)
      }
      
      sut.startWait(dop)
      
      for (rep <- 1 to numRepetitions) {
        
        for (outputFolder <- outputToRemove) {
          sut.removeOutputFolder(outputFolder)
        }

        println("\n-------------------- RUN EXPERIMENT --------------------\n")
        jobDriver.computeGainsSFO(dop)

        val experimentID = experimentPrefix + "-dop%04d-run%02d".format(dop, rep)
        for (outputFolder <- logFilesToBackup) {
          sut.backupJobLogs(outputFolder, experimentID, "job-logs-" + outputFolder)
        }
        
      }
      
      sut.stop()
      
      sut.fsCleanStop()
    }
  }
  
  def getDate(format: String) = {
    val df = new SimpleDateFormat(format)
    val today = Calendar.getInstance().getTime()
    df.format(today)
  }
  
  def printTopGains(gains: List[FeatureGain], datasetInfo: DatasetInfo) = {
    for (i <- 0 until 10) {
      println("d " + gains.get(i).getDimension() + 
          " (" + datasetInfo.getFeatureName(gains.get(i).getDimension()) 
          + ") gain: " + gains.get(i).getGain() + " coefficient(pact-only): " + gains.get(i).getCoefficient())
    }
  }

}

