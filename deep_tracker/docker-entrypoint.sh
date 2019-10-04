#!/bin/sh
set -e

# cd /opt/pytorch/vision
# git config --global user.email "Lukas.Bommes@gmx.de"
# git config --global user.name "LukasBommes"
# git pull https://github.com/LukasBommes/vision.git
# pip install -v .
# cd /workspace

# apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
# pip install opencv-python tb-nightly
#
# cd /workspace
# pip install pkgconfig
# pip install video_cap-1.1.0-cp36-cp36m-linux_x86_64.whl

exec "$@"
