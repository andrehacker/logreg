#!/bin/bash

scp target/logreg-experiments-0.0.1-SNAPSHOT-experiments.jar ahacker@cloud-7.dima.tu-berlin.de:~/experiments/jars/
scp ../logreg-pact/target/logreg-pact-0.0.1-SNAPSHOT-job.jar ahacker@cloud-7.dima.tu-berlin.de:~/experiments/jars/
scp ../logreg-mapred/target/logreg-mapred-0.0.1-SNAPSHOT-job.jar ahacker@cloud-7.dima.tu-berlin.de:~/experiments/jars/
