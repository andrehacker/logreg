package de.tuberlin.dima.experiments

import java.io.File

import org.apache.hadoop.fs.Path
import org.slf4j.LoggerFactory

class HadoopSUT(confFile: String) extends HdfsBasedSUT(confFile) {
  
  private val logger = LoggerFactory.getLogger(this.getClass())
  
  val experimentLogDir = getProperty("experiment_log_dir")
  
  val SEARCH_STRING_TASKTRACKER_CONNECTED = "adding a new node:"
  val SEARCH_STRING_TASKTRACKER_CONNECTED_YARN = "Added node "
  
  override def startWait(numSlaves: Int) = {
    
    logger.info("-------------------- START MAPRED --------------------\n")
    
    // Jobtracker already running?
    if (isJobtrackerRunning()) {
      throw new RuntimeException("Jobtracker is already running. Please stop it before.");
    }
    
    // Delete Jobtracker logfiles
    bash("rm -Rf " + hadoopLog + "/hadoop-" + user + "-*tracker-*.log*")
    
    // Start Hadoop MapRed
    if (isYarn) {
      bash(hadoopSystemHome + "/sbin/yarn-daemon.sh --config " + hadoopConfPath + " start resourcemanager")
      bash(hadoopSystemHome + "/sbin/yarn-daemons.sh --config " + hadoopConfPath + " start nodemanager")
      bash(hadoopSystemHome + "/bin/yarn start proxyserver --config " + hadoopConfPath)
      bash(hadoopSystemHome + "/sbin/mr-jobhistory-daemon.sh start historyserver --config " + hadoopConfPath)
    } else{
      bash(hadoopSystemHome + "/bin/start-mapred.sh")
    }
    
    logger.info("Waiting for " + numSlaves + " nodes (tasktracker or nodemanager) to connect")
    waitForNodesConnected(
        numSlaves, 
        if (isYarn) hadoopLog + "/yarn-" + user + "-resourcemanager-" + hdfsNameNodeHostname + ".log"
        else hadoopLog + "/hadoop-" + user + "-jobtracker-" + hdfsNameNodeHostname + ".log", 
        if (isYarn) SEARCH_STRING_TASKTRACKER_CONNECTED_YARN 
        else SEARCH_STRING_TASKTRACKER_CONNECTED)
  }
  
  override def stop() = {
    
    logger.info("-------------------- STOP MAPRED --------------------\n")
    
    if (isYarn) {
      bash(hadoopSystemHome + "/sbin/yarn-daemon.sh --config " + hadoopConfPath + " stop resourcemanager")
      bash(hadoopSystemHome + "/sbin/yarn-daemons.sh --config " + hadoopConfPath + " stop nodemanager")
      bash(hadoopSystemHome + "/bin/yarn stop proxyserver --config " + hadoopConfPath)
      bash(hadoopSystemHome + "/sbin/mr-jobhistory-daemon.sh stop historyserver --config " + hadoopConfPath)
    } else{
      bash(hadoopSystemHome + "/bin/stop-mapred.sh")
    }
    
    // TODO Minor: Why do the jp-scripts sleep here and look for ghost JVMs?
  }
  
  override def backupJobLogs(outputPath: String, experimentID: String, logName: String): Boolean = {
    
    logger.info("-------------------- JOB LOG BACKUP --------------------\n")
    
    // Create local folder for log backup
    (new File(experimentLogDir)).mkdirs()
    
    // Copy job logfiles
    val src = outputPath + "/_logs/history"
    val target = experimentLogDir + "/" + experimentID + "/" + logName
    
    logger.info("Backup job logs from " + src + " to " + target)
    if (getHDFSFileSystem.exists(new Path(src))) {
      getHDFSFileSystem.copyToLocalFile(
          new Path(src),
          new Path(target))
          true
    } else {
      logger.error("Log folder {} does not exist", src)
      false
    }
  }

  private def isJobtrackerRunning(): Boolean = {
    logger.info("Check if jobtracker is running")
    checkPIDRunning(hadoopPidFolder + "/hadoop-" + user + "-jobtracker.pid")
  }

}