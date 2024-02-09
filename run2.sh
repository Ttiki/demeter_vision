#!/bin/bash

##############################################
# DO NOT MODIFY THIS FILE
##############################################

# Script for running your simulation
##############################################

run(){

    echo "Running the model ..." 
    docker image rm -f genhack3  # delete the existing image
    docker build -t genhack3 .  # build new image
    docker run --name container-sim -v ${PWD}/data:/data genhack3 # run the container and mount the data as volume
    # copy output files to host
    docker cp container-sim:/check.log ${PWD}/check.log
    docker cp container-sim:/output.npy ${PWD}/output.npy
    # remove the contaier
    docker rm container-sim
    echo "... end of run"
}

run