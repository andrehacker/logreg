package de.tuberlin.dima.experiments

import scala.io.Source
import java.io.File
import org.apache.hadoop.fs.Path
import org.apache.commons.io.FileUtils
import org.slf4j.LoggerFactory

/**
 * Ozone uses
 * - repo url: https://repository.cloudera.com/artifactory/cloudera-repos
 * - repo id: cloudera-releases
 * - hadoop-common 2.0.0-cdh4.2.1
 * - hadoop-hdfs 2.0.0-cdh4.2.1
 * - CDH4.2.1 is based on 2.0.0-alpha, but contains a ton of patches from later releases on top of it;-)
 */
class OzoneSUT(confFile: String) extends HdfsBasedSUT(confFile) {
  
  private val logger = LoggerFactory.getLogger(this.getClass())
  
  val ozoneTar = getProperty("ozone_tar")
  val ozoneSystemHome = getProperty("ozone_home")
  val ozoneConfTemplatePath = getProperty("ozone_conf_template")
  val ozoneConfPath = getProperty("ozone_conf")
  val ozoneLog = getProperty("ozone_log")
  val ozoneSlavesFile = getProperty("ozone_slaves_file")
  val ozonePidFolder = getProperty("ozone_pid_folder")

  val experimentLogDir = getProperty("experiment_log_dir")

  val SEARCH_STRING_TASKMANAGERS_CONNECTED = "Creating instance"
  
  override def deploy() = {
    
    logger.info("Deploy hdfs before deploying ozone")
    
    // Deploy hdfs 
    super.deploy

    logger.info("-------------------- DEPLOY OZONE --------------------\n")
    
    // Nephele jobmanager running?
    if (isNepheleRunning()) {
      throw new RuntimeException("Nephele jobmanager is already running. Please stop it before.");
    }

    deployFromTar(ozoneTar, ozoneSystemHome, ozoneConfTemplatePath, ozoneConfPath, user, group)
    
    // Overwrite nephele-config.sh in bin, e.g. for custom pid folder
    val nepheleConfScriptOverwrite = getOptionalProperty("nephele_conf_script_overwrite", "")
    if (nepheleConfScriptOverwrite != "") {
      bash("cp " + nepheleConfScriptOverwrite + " " + ozoneSystemHome + "/bin/nephele-config.sh")
    }
    
    // Optional: Copy libs to hadoop lib dir
    val ozoneLibDirToCopy = getOptionalProperty("ozone_lib_dir_to_copy", "")
    if (ozoneLibDirToCopy != "") {
      logger.info("Copy additional libs to sut's lib dir")
      bash("cp " + ozoneLibDirToCopy + "/* " + ozoneSystemHome + "/lib/")
    }

  }
  
  override def adaptSlaves(numSlaves: Int) = {
    
    // Adapt slaves for hdfs
    super.adaptSlaves(numSlaves)
    
    // Adapt slaves for nephele / ozone
    adaptSlavesFile(allSlavesFilePath, ozoneSlavesFile, numSlaves)
  }
  
  override def startWait(numSlaves: Int) = {

    logger.info("-------------------- START OZONE --------------------\n")
    
    // TODO Create pid dir if it does not exist
    
    // Nephele jobmanager running?
    if (isNepheleRunning()) {
      throw new RuntimeException("Nephele jobmanager is already running. Please stop it before.");
    }
    
    // Delete JobManager logfiles
    bash("rm -Rf " + ozoneLog + "/nephele-" + user + "-*.log*")
    bash("rm -Rf " + ozoneLog + "/nephele-" + user + "-*.out*")
    
    // Start cluster
    bash(ozoneSystemHome + "/bin/start-cluster.sh")
    
    logger.info("Waiting for " + numSlaves + " taskmanager to connect")
    waitForNodesConnected(
        numSlaves, 
        ozoneLog + "/nephele-" + user + "-jobmanager-" + hdfsNameNodeHostname + ".log",
        SEARCH_STRING_TASKMANAGERS_CONNECTED)

  }
  
  override def stop() = {
    logger.info("-------------------- STOP OZONE --------------------\n")
    bash(ozoneSystemHome + "/bin/stop-cluster.sh")
  }
  
  override def backupJobLogs(outputPath: String, experimentID: String, logName: String) = {

    logger.info("-------------------- LOG BACKUP --------------------\n")
    
    // Create local folder for log backup
    (new File(experimentLogDir)).mkdirs()
    
    // Copy job logfiles (stored locally by ozone)
    val src = ozoneSystemHome + "/log"
    val target = experimentLogDir + "/" + experimentID + "/" + logName + "/"
    logger.info("Backup job logs from " + src + " to " + target)
    if (new File(src).exists()) {
      FileUtils.copyDirectory(new File(src), new File(target))
      true
    } else {
      logger.error("Log folder {} does not exist\n", src)
      false
    }
  }
  
  private def isNepheleRunning(): Boolean = {
    logger.info("Check if nephele jobmanager is running")
    checkPIDRunning(ozonePidFolder + "/nephele-" + user + "-jobmanager.pid")
  }

}