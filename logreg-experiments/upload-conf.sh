#!/bin/bash
# experiment properties
scp sfo-experiment-cloud-7.properties ahacker@cloud-7.dima.tu-berlin.de:~/experiments/

# hadoop system properties
scp conf-templates/cloud-7/sysconf-hadoop-1.2.1.properties ahacker@cloud-7.dima.tu-berlin.de:~/experiments/conf-templates/

# all slaves
scp conf-templates/cloud-7/all_slaves ahacker@cloud-7.dima.tu-berlin.de:~/experiments/conf-templates/

# conf templates hadoop
scp -r conf-templates/cloud-7/hadoop-1.2.1 ahacker@cloud-7.dima.tu-berlin.de:~/experiments/conf-templates/

# hadoop-config.sh 
scp conf-templates/cloud-7/hadoop-config.sh ahacker@cloud-7.dima.tu-berlin.de:~/experiments/conf-templates/