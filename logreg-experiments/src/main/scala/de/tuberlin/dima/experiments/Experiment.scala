package de.tuberlin.dima.experiments

import java.io.File
import java.util.Properties
import java.io.FileInputStream
import org.slf4j.LoggerFactory

abstract class Experiment {
  
  private val logger = LoggerFactory.getLogger(this.getClass())
  
  val sysProp = new Properties()
  val experimentProp = new Properties()

  // ---------- PRIMARY CONSTRUCTOR ----------
  def init(sysConfFile: String, experimentConfFile: String) = {
    sysProp.load(new FileInputStream(sysConfFile))
    experimentProp.load(new FileInputStream(experimentConfFile))
  }
  
  // ---------- BASE IMPLEMENTATIONS ----------
  
  def getSysProperty(name: String): String = {
    return getProperty(name, sysProp)
  }
  
  def getProperty(name: String): String = {
    return getProperty(name, experimentProp)
  }
  
  def getPropertyArray(name: String): Array[String] = {
    val arraySeparator = ","
    getProperty(name, experimentProp).split(arraySeparator)
  }
  
  def getPropertyArrayAsInt(name: String): Array[Int] = {
    val arraySeparator = ","
    getProperty(name, experimentProp).split(arraySeparator).map(_.trim.toInt)
  }
  
  private def getProperty(name: String, prop: Properties): String = {
    val value = prop.getProperty(name)
    if (value != null) {
      logger.info("Loaded Property: " + name + " = " + value)
      value
    } else throw new RuntimeException("Could not read property " + name)
  }
  
  def requirePathExists(path: String) = {
    // Check if this path exists
    if (! (new File(path).exists()))
      throw new RuntimeException("Required file or directory does not exist: " + path);
  }

}