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

object SFOExperiment {
  
  def main(args: Array[String]) {
    runExperiment()
  }
  
  val experimentName = "sfo"
  
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
   */
  def runExperiment() = {

    // --------------- SYSTEM INFORMATION ----------
    
    val jobTrackerAddress = "localhost:9001"    // mapred.job.tracker
    val hdfsAddress = "hdfs://localhost:9000"   // fs.default.name
    
    val jobManagerAddress = "127.0.0.1"
    val jobManagerPort = ConfigConstants.DEFAULT_JOB_MANAGER_IPC_PORT

    // --------------- JOB PARAMETERS ----------
    
    val dataset = RCV1DatasetInfo.get()
    val predictorNamePath = "/home/andre/dev/datasets/RCV1-v2/stem.termid.idf.map.txt"
    RCV1DatasetInfo.readPredictorNames(predictorNamePath)

    // --------------- JOB PARAMETERS HADOOP ----------

    val jarPathHadoop = "/home/andre/dev/logreg-repo/logreg-mapred/target/logreg-mapred-0.0.1-SNAPSHOT-job.jar"
    val inputTrainLocalHadoop = "/home/andre/dev/datasets/RCV1-v2/sequencefiles/lyrl2004_vectors_ecat_train_1000.seq"
    val inputTrainHadoop = "experiments/input/rcv1/lyrl2004_vectors_ecat_train_1000.seq"
    val outputTrainHadoop = "output-sfo-train"
    val outputTestHadoop = "output-sfo-test"

    // --------------- JOB PARAMETERS OZONE ----------
    
    val jarPathPact = "/home/andre/dev/logreg-repo/logreg-pact/target/logreg-pact-0.0.1-SNAPSHOT-job.jar"
    val inputTrainLocalPact = "file:///home/andre/dev/datasets/libsvm-rcv1v2-topics/rcv1_topics_train_1000.svm"
    val inputTrainPact = hdfsAddress + "/experiments/input/rcv1/rcv1_topics_train_1000.svm"
    val labelIndex = 59 // CCAT=33, ECAT=59, GCAT=70, MCAT=102
    val outputPact = hdfsAddress + "/output-sfo-pact"
    
    // ---------------------------------------------------

    val hadoopDriver = new SFODriverHadoop(
      inputTrainHadoop,
      inputTrainHadoop,
      outputTrainHadoop,
      outputTestHadoop,
      dataset.getNumFeatures().toInt,
      jobTrackerAddress,
      hdfsAddress,
      jarPathHadoop)

    val pactDriver = new SFODriverPact(
      inputTrainPact,
      inputTrainPact,
      labelIndex,
      outputPact,
      dataset.getNumFeatures().toInt,
      false,
      jarPathPact,
      jobManagerAddress,
      jobManagerPort.toString)
    
    
    // --------------- EXPERIMENT ---------------
    
    val dops = Array[Int](1)
    val iterations = 1
    val numRepetitions = 1
    val addPerIteration = 1

    // --------------- SMALL YARN TEST ----------
    
//    val yarn = new HadoopSUT("/home/andre/experiments/conf-templates/andre-sam-ubuntu/sysconf-hadoop-2.0.5-alpha.properties")
//    yarn.deploy
//    yarn.adaptSlaves(1)
//    yarn.fsFormatStartWait(1)
//    yarn.fsCleanStop

    // --------------- RUN HADOOP EXPERIMENT ----------
    
    val hadoop = new HadoopSUT("/home/andre/dev/logreg-repo/logreg-experiments/conf-templates/andre-sam-ubuntu/sysconf-hadoop-1.2.1.properties")
    
    val experimentPrefixHadoop = "%s-%s-%s-%s".format(experimentName, "hdp", "rcv1small", getDate("yyyy-MM-dd-HHmmss"))
    val dataToLoadHadoop = Array((inputTrainLocalHadoop, inputTrainHadoop))
    val logFilesToBackupHadoop = Array((outputTrainHadoop, "job-logs-train"), (outputTestHadoop, "job-logs-test"))
    val outputToRemoveHadoop = Array(outputTrainHadoop, outputTestHadoop)
//    runExperimentSingleSUT(hadoop, dops, numRepetitions, dataToLoadHadoop, outputToRemoveHadoop, logFilesToBackupHadoop, hadoopDriver, experimentPrefixHadoop)
    

    // --------------- RUN OZONE EXPERIMENT ----------
    
    val ozone = new OzoneSUT("/home/andre/dev/logreg-repo/logreg-experiments/conf-templates/andre-sam-ubuntu/sysconf-hadoop-1.2.1.properties")
//    val ozone = new OzoneSUT("/home/andre/dev/logreg-repo/logreg-experiments/conf-templates/andre-sam-ubuntu/sysconf-hadoop-2.0.5-alpha.properties")
    
    val experimentPrefixPact = "%s-%s-%s-%s".format(experimentName, "ozn", "rcv1small", getDate("yyyy-MM-dd-HHmmss"))
    val dataToLoadPact = Array((inputTrainLocalPact, inputTrainPact))
    val logFilesToBackupPact = Array((outputPact, "job-logs"))
    val outputToRemovePact = Array(outputPact)
    runExperimentSingleSUT(ozone, dops, numRepetitions, dataToLoadPact, outputToRemovePact, logFilesToBackupPact, pactDriver, experimentPrefixPact)
  }
  
  def runExperimentSingleSUT(
      sut: SUT, 
      dops: Array[Int], 
      numRepetitions: Int, 
      dataToLoad: Array[(String, String)], 
      outputToRemove: Array[String],
      logFilesToBackup: Array[(String, String)],
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

        jobDriver.computeGainsSFO(dop)

        val experimentID = experimentPrefix + "-dop%04d-run%02d".format(dop, rep)
        for ((outputFolder, logname) <- logFilesToBackup) {
          sut.backupJobLogs(outputFolder, experimentID, logname)
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

