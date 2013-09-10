#!/bin/bash
# Make sure that the dependencies don't contain the actual code for experiments!
DEPENDENCIES_JARS=libs/*:/home/ahacker/experiments/share/hadoop/common/lib/*
EXP_JAR=jars/logreg-experiments-0.0.1-SNAPSHOT-experiments.jar
MAIN_CLASS=de.tuberlin.dima.experiments.ml.experiments.SFOExperiment
SYS_PROPERTIES=/home/ahacker/experiments/conf-templates/sysconf-hadoop-2.1.0-beta.properties
EXP_PROPERTIES=/home/ahacker/experiments/sfo-experiment-cloud-11.properties
LOG4J_PROPERTIES=/home/ahacker/experiments/log4j

java -Dlog4j.debug=true -cp ${LOG4J_PROPERTIES}:${DEPENDENCIES_JARS}:${EXP_JAR} ${MAIN_CLASS} $SYS_PROPERTIES $EXP_PROPERTIES
