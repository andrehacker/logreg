#!/bin/bash
# experiment properties
scp sfo-experiment-cloud-11.properties ahacker@cloud-11.dima.tu-berlin.de:~/experiments/

# system properties
scp conf-templates/cloud-11/sysconf-hadoop-2.1.0-beta.properties ahacker@cloud-11.dima.tu-berlin.de:~/experiments/conf-templates/

# all slaves
scp conf-templates/cloud-11/all_slaves ahacker@cloud-11.dima.tu-berlin.de:~/experiments/conf-templates/

# conf templates hadoop + stratosphere
scp -r conf-templates/cloud-11/hadoop-2.1.0-beta ahacker@cloud-11.dima.tu-berlin.de:~/experiments/conf-templates/
scp -r conf-templates/cloud-11/stratosphere-0.2-ozone ahacker@cloud-11.dima.tu-berlin.de:~/experiments/conf-templates/

# hadoop-config.sh + nephele-config.sh
scp conf-templates/cloud-11/hadoop-config.sh ahacker@cloud-11.dima.tu-berlin.de:~/experiments/conf-templates/
scp conf-templates/cloud-11/nephele-config.sh ahacker@cloud-11.dima.tu-berlin.de:~/experiments/conf-templates/

# Script
scp run-sfo-cloud11.sh ahacker@cloud-11.dima.tu-berlin.de:~/experiments/
ssh ahacker@cloud-11.dima.tu-berlin.de 'chmod -R g+x experiments/run-sfo-cloud11.sh'

ssh ahacker@cloud-11.dima.tu-berlin.de 'chgrp -R hadoop experiments/conf-templates'
ssh ahacker@cloud-11.dima.tu-berlin.de 'chmod -R g+w experiments/conf-templates'