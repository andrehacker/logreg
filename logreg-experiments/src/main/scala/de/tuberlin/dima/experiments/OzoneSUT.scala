package de.tuberlin.dima.experiments

import scala.io.Source
import java.io.File
import org.apache.hadoop.fs.Path
import org.apache.commons.io.FileUtils

class OzoneSUT(confFile: String) extends HdfsBasedSUT(confFile) {
  
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
    
    println("Deploy hdfs before deploying ozone")
    
    // Deploy hdfs 
    super.deploy

    println("\n-------------------- DEPLOY OZONE --------------------\n")
    
    // Nephele jobmanager running?
    if (isNepheleRunning()) {
      throw new RuntimeException("Nephele jobmanager is already running. Please stop it before.");
    }

    deployFromTar(ozoneTar, ozoneSystemHome, ozoneConfTemplatePath, ozoneConfPath, user, group)
    
  }
  
  override def adaptSlaves(numSlaves: Int) = {
    
    // Adapt slaves for hdfs
    super.adaptSlaves(numSlaves)
    
    // Adapt slaves for nephele / ozone
    adaptSlavesFile(allSlavesFilePath, ozoneSlavesFile, numSlaves)
  }
  
  override def startWait(numSlaves: Int) = {

    println("\n-------------------- START OZONE --------------------\n")
    
    // Nephele jobmanager running?
    if (isNepheleRunning()) {
      throw new RuntimeException("Nephele jobmanager is already running. Please stop it before.");
    }
    
    // Delete JobManager logfiles
    p("rm -Rf " + ozoneLog + "/nephele-" + user + "-*.log*") !;
    p("rm -Rf " + ozoneLog + "/nephele-" + user + "-*.out*") !;
    
    // Start cluster
    p(ozoneSystemHome + "/bin/start-cluster.sh") !;
    
    println("Waiting for " + numSlaves + " taskmanager to connect")
    waitForNodesConnected(
        numSlaves, 
        ozoneLog + "/nephele-" + user + "-jobmanager-" + hdfsNameNodeHostname + ".log",
        SEARCH_STRING_TASKMANAGERS_CONNECTED)

  }
  
  override def stop() = {
    println("\n-------------------- STOP OZONE --------------------\n")
    p(ozoneSystemHome + "/bin/stop-cluster.sh") !;
  }
  
  override def backupJobLogs(outputPath: String, experimentID: String, logName: String) = {

    println("\n-------------------- LOG BACKUP --------------------\n")
    
    // Create local folder for log backup
    (new File(experimentLogDir)).mkdirs()
    
    // Copy job logfiles (stored locally by ozone)
    val src = ozoneSystemHome + "/log"
    val target = experimentLogDir + "/" + experimentID + "/" + logName + "/"
    printf("Backup job logs from %s to %s\n", src, target)
    FileUtils.copyDirectory(new File(src), new File(target))
  }
  
  private def isNepheleRunning(): Boolean = {
    println("Check if nephele jobmanager is running")
    checkPIDRunning(ozonePidFolder + "/nephele-" + user + "-jobmanager.pid")
  }

}