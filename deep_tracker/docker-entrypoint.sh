#!/bin/sh
set -e

cd /opt/pytorch/vision
git config --global user.email "Lukas.Bommes@gmx.de"
git config --global user.name "LukasBommes"
git pull https://github.com/sampepose/vision.git
pip install -v .
cd /workspace

apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

exec "$@"
