#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

env PYTHONPATH=src:thirdparty/Trajectron_plus_plus/trajectron:thirdparty/PECNet/utils python3 -m pytest test -v