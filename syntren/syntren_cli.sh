#!/bin/sh
#
# Run SynTReN from a jar file
# this is a linux-only version
#-------------------------------------------------------------------------------

java -Xmx512M -cp SynTReN.jar islab.bayesian.genenetwork.generation.NetworkGeneratorCLI $*
