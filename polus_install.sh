#!/bin/bash

#https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

echo $SCRIPTPATH
cd $SCRIPTPATH

# remove the old polus
pip uninstall polus -y

python setup.py bdist_wheel

pip install dist/polus-0.1.0-py3-none-any.whl
cd -