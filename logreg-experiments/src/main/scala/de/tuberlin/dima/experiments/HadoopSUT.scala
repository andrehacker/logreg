package de.tuberlin.dima.experiments

import java.io.File
import scala.io.Source
import scala.sys.process._
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import de.tuberlin.dima.ml.mapred.GlobalSettings
import java.io.IOException
import java.util.Properties
import java.io.FileInputStream
import java.io.PrintWriter

class HadoopSUT(confFile: String) extends HdfsBasedSUT(confFile) {
  
  val experimentLogDir = getProperty("experiment_log_dir")
  val hadoopPidFolder = getProperty("hadoop_pid_folder")
  
  val SEARCH_STRING_TASKTRACKER_CONNECTED = "adding a new node:"
  val SEARCH_STRING_TASKTRACKER_CONNECTED_YARN = "Added node "
  
  override def startWait(numSlaves: Int) = {
    
    println("\n-------------------- START MAPRED --------------------\n")
    
    // Jobtracker already running?
    if (isJobtrackerRunning()) {
      throw new RuntimeException("Jobtracker is already running. Please stop it before.");
    }
    
    // Delete Jobtracker logfiles
    p("rm -Rf " + hadoopLog + "/hadoop-" + user + "-*tracker-*.log*") !;
    
    // Start Hadoop MapRed
    if (isYarn) {
      p(hadoopSystemHome + "/sbin/yarn-daemon.sh --config " + hadoopConfPath + " start resourcemanager") !;
      p(hadoopSystemHome + "/sbin/yarn-daemons.sh --config " + hadoopConfPath + " start nodemanager") !;
//      p(hadoopSystemHome + "/bin/yarn start proxyserver --config " + hadoopConfPath)!;
//      p(hadoopSystemHome + "/sbin/mr-jobhistory-daemon.sh start historyserver --config " + hadoopConfPath)!;
    } else{
      p(hadoopSystemHome + "/bin/start-mapred.sh") !;
    }
    
    println("Waiting for " + numSlaves + " nodes (tasktracker or nodemanager) to connect")
    waitForNodesConnected(
        numSlaves, 
        if (isYarn) hadoopLog + "/yarn-" + user + "-resourcemanager-" + hdfsNameNodeHostname + ".log"
        else hadoopLog + "/hadoop-" + user + "-jobtracker-" + hdfsNameNodeHostname + ".log", 
        if (isYarn) SEARCH_STRING_TASKTRACKER_CONNECTED_YARN 
        else SEARCH_STRING_TASKTRACKER_CONNECTED)
  }
  
  override def stop() = {
    
    println("\n-------------------- STOP MAPRED --------------------\n")
    
    if (isYarn) {
      p(hadoopSystemHome + "/sbin/yarn-daemon.sh --config " + hadoopConfPath + " stop resourcemanager") !;
      p(hadoopSystemHome + "/sbin/yarn-daemons.sh --config " + hadoopConfPath + " stop nodemanager") !;
//      p(hadoopSystemHome + "/bin/yarn stop proxyserver --config " + hadoopConfPath)!;
//      p(hadoopSystemHome + "/sbin/mr-jobhistory-daemon.sh stop historyserver --config " + hadoopConfPath)!;
    } else{
      p(hadoopSystemHome + "/bin/stop-mapred.sh") !;
    }
    
    // TODO Minor: Why do the jp-scripts sleep here and look for ghost JVMs?
  }
  
  override def backupJobLogs(outputPath: String, experimentID: String, logName: String) = {
    
    println("\n-------------------- LOG BACKUP --------------------\n")
    
    // Create local folder for log backup
    (new File(experimentLogDir)).mkdirs()
    
    // Copy job logfiles
    val src = outputPath + "/_logs/history"
    val target = experimentLogDir + "/" + experimentID + "/" + logName
    
    printf("Backup job logs from %s to %s\n", src, target)
    getHDFSFileSystem.copyToLocalFile(
        new Path(src),
        new Path(target))
  }

  private def isJobtrackerRunning(): Boolean = {
    println("Check if jobtracker is running")
    checkPIDRunning(hadoopPidFolder + "/hadoop-" + user + "-jobtracker.pid")
  }

}