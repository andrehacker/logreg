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

abstract class HdfsBasedSUT(confFile: String) extends SUT(confFile) {
  
  val isYarn = getProperty("is_yarn").toBoolean
  val allSlavesFilePath = getProperty("all_slaves")
  val user = getProperty("user")
  val group = getProperty("group")
  val hadoopTar = getProperty("hadoop_tar")
  val hadoopSystemHome = getProperty("hadoop_home")
  val hadoopConfTemplatePath = getProperty("hadoop_conf_template")
  val hadoopConfPath = getProperty("hadoop_conf")
  val hadoopLog = getProperty("hadoop_log")
  val hadoopSlavesFile = getProperty("hadoop_slaves_file")
  val hadoopPidFolder = getProperty("hadoop_pid_folder")
  val hdfsDataDir = getProperty("hdfs_data_dir")
  val hdfsAddress = getProperty("hdfs_address")
  val hdfsNameNodeHostname = getProperty("hdfs_namenode_hostname")

  val CONF_KEY_HDFS_ADDRESS = "fs.default.name"
  val SEARCH_STRING_DATANODE_CONNECTED = "registerdatanode"
  
  override def deploy() = {
    
    println("\n-------------------- DEPLOY HDFS (incl. Hadoop) --------------------\n")
    
    // HDFS already running?
    if (isNameNodeRunning()) {
      throw new RuntimeException("Hdfs is already running. Please stop it before.");
    }
    
    deployFromTar(hadoopTar, hadoopSystemHome, hadoopConfTemplatePath, hadoopConfPath, user, group)
    
    // Workaround for /share -> /export link. Overwrite hadoop-config.sh in bin and libexec
    val hadoopConfScriptOverwrite = getOptionalProperty("hadoop_conf_script_overwrite")
    if (hadoopConfScriptOverwrite != "") {
      p("cp " + hadoopConfScriptOverwrite + " " + hadoopSystemHome + "/bin/hadoop-config.sh") !;
      if (new File(hadoopSystemHome + "/libexec/hadoop-config.sh").exists) {
        p("cp " + hadoopConfScriptOverwrite + " " + hadoopSystemHome + "/libexec/hadoop-config.sh") !;
      }
    } 
  }
  
  
  override def adaptSlaves(numSlaves: Int) = {
    
    println("\n-------------------- ADAPT HADOOP SLAVES --------------------\n")
    
    adaptSlavesFile(allSlavesFilePath, hadoopSlavesFile, numSlaves)
  }
  
  
  override def fsFormatStartWait(numSlaves: Int) = {
    
    println("\n-------------------- HDFS FORMAT START WAIT --------------------\n")
    
    // Delete hdfs log files
    // Filename: hadoop-<user-running-hadoop>-<daemon>-<hostname>.log
    // We ignore .out files
    // See http://blog.cloudera.com/blog/2009/09/apache-hadoop-log-files-where-to-find-them-in-cdh-and-what-info-they-contain/
    p("rm -Rf " + hadoopLog + "/hadoop-" + user + "-*node-*.log*") !
    
    // Delete data dir on all slaves
    val slaves = Source.fromFile(new File(hadoopSlavesFile)).getLines;
    for (slave <- slaves) {
      printf("Delete %s on slave %s", hdfsDataDir + "\n", slave)
      Process("ssh", Seq(user + "@" + slave, "rm -Rf " + hdfsDataDir)).!
//      Seq("sh", "-c", "ulimit -n").!!;
//      p("ssh " + user + "@" + slave + " 'rm -Rf " + dataDir + "'") !;
    }
    
    if (isYarn) {
      p(hadoopSystemHome + "/bin/hdfs namenode -format " + hdfsNameNodeHostname + " -force") !;
    } else {
      p(hadoopSystemHome + "/bin/hadoop namenode -format -force") !
    }
    
    // The last parameter is the ClusterID, which is only relevant if we have multiple namenodes (HA HDFS)
    // Description of ClusterID: A new identifier ClusterID is added to identify all the nodes in the cluster. When a Namenode is formatted, this identifier is provided or auto generated. This ID should be used for formatting the other Namenodes into the cluster.
    if (isYarn) {
      p(hadoopSystemHome + "/sbin/hadoop-daemon.sh --config " + hadoopConfPath + " --script hdfs start namenode")!;
      p(hadoopSystemHome + "/sbin/hadoop-daemons.sh --config " + hadoopConfPath + " --script hdfs start datanode")!;
//      p(hadoopSystemHome + "/bin/start-dfs.sh") !
    } else{
      p(hadoopSystemHome + "/bin/start-dfs.sh") !
    }

    println("Waiting for safe mode to be off")
    var safeModeOff = false
    while(!safeModeOff) {
      val safeModeResult = p(hadoopSystemHome + "/bin/hadoop dfsadmin -safemode get", false) !!;
      safeModeOff = safeModeResult.toLowerCase().contains("off");
      printf("- Safemode-OFF=%s\n", safeModeOff.toString)
      Thread.sleep(500);
    }
    
    println("Waiting for " + numSlaves + " datanodes to connect")
    waitForNodesConnected(
        numSlaves, 
        hadoopLog + "/hadoop-" + user + "-namenode-" + hdfsNameNodeHostname + ".log", 
        SEARCH_STRING_DATANODE_CONNECTED)
    
    // Hdfs Webinterface shows the number of connected datanodes. Tried to use their internal logic, but it is easy to use
    // jspHelper.DFSNodesStatus(live, dead);
    // - http://svn.apache.org/viewvc/hadoop/common/tags/release-1.2.1/src/webapps/hdfs/dfshealth.jsp?view=markup
    // fsn.DFSNodesStatus(live, dead);
    // - http://svn.apache.org/viewvc/hadoop/common/tags/release-1.2.1/src/hdfs/org/apache/hadoop/hdfs/server/namenode/JspHelper.java?view=markup
    // FSNamesystem.DFSNodesStatus(live, dead)
    // - calls getDatanodeListForReport(DatanodeReportType.ALL);
    // - http://svn.apache.org/viewvc/hadoop/common/tags/release-1.2.1/src/hdfs/org/apache/hadoop/hdfs/server/namenode/FSNamesystem.java?view=markup
    
  }
  
  
  override def fsCleanStop() = {
    
    println("\n-------------------- HDFS CLEAN STOP --------------------\n")
    
//    p(hadoopSystemHome + "/bin/hadoop fs -rmr '/*'") !;
    val recursive = true
    getHDFSFileSystem.delete(new Path("/"), recursive)
    
    if (isYarn) {
      p(hadoopSystemHome + "/sbin/hadoop-daemon.sh --config " + hadoopConfPath + " --script hdfs stop namenode")!;
      p(hadoopSystemHome + "/sbin/hadoop-daemons.sh --config " + hadoopConfPath + " --script hdfs stop datanode")!;
//      p(hadoopSystemHome + "/bin/stop-dfs.sh") !;
    } else {
      p(hadoopSystemHome + "/bin/stop-dfs.sh") !;
    }
  }
  
  
  override def fsLoadData(localPath: String, destinationPath: String): Boolean = {
    
    println("\n-------------------- LOAD DATA --------------------\n")
    printf("Copy %s to %s using hdfs %s\n", localPath, destinationPath, hdfsAddress)
    
    val fs = getHDFSFileSystem
    
    // See copy method in http://svn.apache.org/viewvc/hadoop/common/tags/release-1.2.1/src/core/org/apache/hadoop/fs/FileUtil.java?view=markup
    // Old: ${HDFS_BIN}/hadoop fs -copyFromLocal $INPUT $OUTPUT
    val delSource = false; val overWrite = false
    try {
      fs.copyFromLocalFile(delSource, overWrite, new Path(localPath), new Path(destinationPath))
      true
    } catch {
      case ex: Exception => println("Exception: " + ex.toString())
      false
    }
  }

  override def removeOutputFolder(outputPath: String): Boolean = {
    val fs = getHDFSFileSystem
    if (fs.exists(new Path(outputPath))) {
      printf("Remove output folder %s\n", outputPath)
      val recursive = true
      fs.delete(new Path(outputPath), recursive)
      true
    } else {
      printf("Output path %s is empty, nothing to delete\n", outputPath)
      false
    }
  }
 
  
  
  protected def adaptSlavesFile(sourceFile: String, targetFile: String, numSlaves: Int) = {
    requirePathExists(sourceFile)
    val allSlaves = Source.fromFile(new File(sourceFile)).getLines;
    val writer = new PrintWriter(new File(targetFile))
    println("Write current slaves (" + numSlaves + ") to " + targetFile + ":")
    for (i <- 1 to numSlaves) {
      if (!allSlaves.hasNext) {
        throw new Exception("All slaves file has less slaves specified than you specified in numSlaves (dop)")
      }
      val slave = allSlaves.next
      writer.println(slave)
      println("- " + slave)
    }
    writer.close()
  }
  
  protected def waitForNodesConnected(numSlaves: Int, logfile: String, searchString: String) = {
    // TODO Make this work for YARN!
    var nodesConnected = 0
    while(nodesConnected<numSlaves) {
      // TODO Add timeout
      if ((new File(logfile)).exists()) {
        val logLines = Source.fromFile(logfile).getLines.toArray
        nodesConnected = logLines.count(str => str.toLowerCase.contains(searchString.toLowerCase));
        printf("- Nodes connected=%d\n", nodesConnected)
      } else {
        nodesConnected = 0
        println("- Nodes connected unknown (logfile not existing)")
      }
      Thread.sleep(1000);
    }
  }
  
  protected def getHDFSFileSystem(): FileSystem = {
    val hdfsConf = new Configuration()
    hdfsConf.set(CONF_KEY_HDFS_ADDRESS, hdfsAddress)
    FileSystem.get(hdfsConf)
  }
  
  protected def isNameNodeRunning(): Boolean = {
    println("Check if hdfs namenode is running")
    checkPIDRunning(hadoopPidFolder + "/hadoop-" + user + "-namenode.pid")
  }
  
  protected def checkPIDRunning(pidFile: String): Boolean = {
    if (new File(pidFile).exists) {
      val pid = Source.fromFile(pidFile).mkString.trim()
      val killExitValue = p("kill -0 " + pid) !;
      (killExitValue == 0)    // exit value = 0 => process is running
    } else false
  }

}