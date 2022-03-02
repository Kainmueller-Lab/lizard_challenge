#!/usr/bin/env bash

#./build.sh

docker save conic-inference3 | gzip -c > conic-inference2.tar.gz
