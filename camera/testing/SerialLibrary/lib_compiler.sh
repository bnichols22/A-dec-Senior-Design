#!/bin/bash

set -e

cd ~/senior_design/A-dec-Senior-Design/camera/testing/SerialLibrary/serialAPI

echo "Removing old build artifacts..."
rm -f libsimplebgc.so sbgc32.o shim_update.o simplebgc_shim.o serialAPI_MakeCpp.o

echo "Compiling SerialAPI sources..."
gcc -fPIC -c sbgc32.c -o sbgc32.o

echo "Compiling shim..."
gcc -fPIC -c shim_update.c -o shim_update.o

echo "Linking shared library..."
gcc -shared -o libsimplebgc.so sbgc32.o shim_update.o -lpthread

echo "Build complete."
