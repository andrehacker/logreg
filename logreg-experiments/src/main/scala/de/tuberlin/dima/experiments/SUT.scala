package de.tuberlin.dima.experiments

import scala.sys.process._
import java.util.Properties
import java.io.FileInputStream
import java.io.File
import org.apache.commons.io.FileUtils

/**
 * Abstract class for a system under test such as Hadoop or Stratosphere.
 * The system under test is assumed to be based on hdfs.
 * Wraps functionality for deployment, running and logging
 * 
 * Convention: All directory have no trailing /. E.g. /path/to/config-dir 
 */
abstract class SUT(confFile: String) {
  
  // ---------- PRIMARY CONSTRUCTOR ----------
  
  val prop = new Properties()
  prop.load(new FileInputStream(confFile))
  
  // ---------- ABSTRACT METHODS ----------
  
  def deploy()
  
  def adaptSlaves(numSlaves: Int)
  
  def fsFormatStartWait(numSlaves: Int)
  
  def fsCleanStop()
  
  def fsLoadData(localPath: String, destinationPath: String): Boolean
  
  def startWait(numSlaves: Int)
  
  /**
   * Stop the system under test.
   * Convention: Call this before calling fsCleanStop()
   */
  def stop()
  
  /**
   * Remove output from a previous job (e.g. hadoop, ozone).
   * Does nothing if the folder does not exists.
   */
  def removeOutputFolder(outputPath: String)
  
  /**
   * Create a backup of the files of a single job (e.g. one hadoop mapred job).
   * Files will be copied to experiment-log-folder/experimentID/logName/.
   * You have to call this multiple times if your experiment is a chained job,
   * because every job has it's own output directory
   * 
   * TODO What about the backup of the job results/output?
   */
  def backupJobLogs(outputPath: String, experimentID: String, logName: String)
  
  // ---------- BASE METHODS ----------
  
  def p(str: String, verbose: Boolean = true) = {
    if (verbose) { printf("- exec %s\n", str) }
    stringToProcess(str)
  }
  
  def getProperty(name: String): String = {
    val value = prop.getProperty(name)
    if (value != null) {
      printf("Loaded Property: %s = %s\n", name, value)
      value
    } else throw new RuntimeException("Could not read property " + name)
  }
  
  def requirePathExists(path: String) = {
    // Check if this path exists
    if (! (new File(path).exists()))
      throw new RuntimeException("Required file or directory does not exist: " + path);
  }
  
  /**
   * Deploy a SUT from a tar file.
   * Logic is the same for multiple systems like Hadoop or Ozone, so we can reuse it
   */
  protected def deployFromTar(tarPath: String, systemHome: String, confTemplatePath: String, confPath: String, user: String, group: String) = {

    requirePathExists(tarPath)
    requirePathExists(confTemplatePath)

    println("- Removing old SUT home folder")
    p("rm -Rf " + systemHome) !;
    
    val systemHomeParent = systemHome.substring(0, systemHome.lastIndexOf("/"))
    if (!(new File(systemHomeParent)).exists()) {
      (new File(systemHomeParent)).mkdir()
    }
    println("- Unpacking tar")
    p("tar -xzvf " + tarPath + " -C " + systemHomeParent) !!;
    
    printf("Copy config files from %s to %s\n", confTemplatePath, confPath)
    (new File(confPath)).mkdirs()  // create folder if it does not yet exist (the case for yarn config folder)
    FileUtils.copyDirectory(new File(confTemplatePath), new File(confPath));
    
    println("- Setting proper rights in SUT home")
    p("chown -R " + user + ":" + group + " " + systemHome) !;
    p("find " + systemHome + " -type f") #| p("xargs -I{} chmod g+w {}") !;
    p("find " + systemHome + " -type d") #| p("xargs -I{} chmod g+w {}") !;
  }

}