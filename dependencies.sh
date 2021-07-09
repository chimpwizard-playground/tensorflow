#!/bin/bash

while [[ $# -gt 0 ]]
do

key="$1"
case $key in
    -p|--install-python)
    export INSTALL_PITHON="$2"
    shift # past argument
    ;;

    --default)
    ;;
    *)
        # unknown option
    ;;
esac

shift # past argument or value
done 

if [[ "$INSTALL_PITHON" == "yes" || "$INSTALL_PITHON" == "true" ]]; then
    echo "Installing python..."
    pyenv versions
    pyenv install 3.7.9
    pyenv local 3.7.9
fi

pip install tensorflow==2.0.0-alpha0 
pip install argparse


#
# REFERENCE:
#
# - https://github.com/pyenv/pyenv