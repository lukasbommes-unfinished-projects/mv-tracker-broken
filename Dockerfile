FROM tensorflow/tensorflow:1.13.2-gpu-py3

# Install tools
RUN apt-get update && apt-get install -y \
	git && \
	rm -rf /var/lib/apt/lists/*


###############################################################################
#
#						 sfmt-videocap module (+ OpenCV & FFMPEG)
#
###############################################################################

ENV HOME "/home"

# Download and build sfmt-videocap from source
RUN cd $HOME && \
  git clone -b "v1.0.0" https://sfmt-auto:Ow36ODbBoSSezciC@github.com/LukasBommes/sfmt-videocap.git video_cap && \
  cd video_cap && \
  chmod +x install.sh && \
  ./install.sh

# Set environment variables
ENV PATH "$PATH:$HOME/bin"
ENV PKG_CONFIG_PATH "$PKG_CONFIG_PATH:$HOME/ffmpeg_build/lib/pkgconfig"

RUN cd $HOME/video_cap && \
  python3 setup.py install


###############################################################################
#
#							Python Packages
#
###############################################################################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	python3-tk && \
	rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt


###############################################################################
#
#							Container Startup & Command
#
###############################################################################

ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH":/usr/local/cuda-10.0/compat/

WORKDIR /mvt

CMD tail -f /dev/null
