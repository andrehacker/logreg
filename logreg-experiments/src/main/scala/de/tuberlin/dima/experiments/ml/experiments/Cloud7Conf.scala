package de.tuberlin.dima.experiments.ml.experiments

import de.tuberlin.dima.experiments.HadoopSUT
import de.tuberlin.dima.experiments.Experiment

object Cloud7Conf extends Experiment {

  def main(args: Array[String]): Unit = {
    
    val sysConfPath = args(0)
    val experimentConfPath = args(1)
    init(sysConfPath, experimentConfPath)

    val hadoop = new HadoopSUT(sysConfPath)

    val dop = 4
    hadoop.deploy()

    hadoop.adaptSlaves(dop)

    hadoop.fsFormatStartWait(dop)

    hadoop.startWait(dop)

  }

}