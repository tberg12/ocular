#!/bin/bash

set -e

./make_jar.sh
scp ocular-0.3-SNAPSHOT-with_dependencies.jar k:public_html/maven-repository/snapshots/edu/berkeley/cs/nlp/ocular/0.3-SNAPSHOT/

