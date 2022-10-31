################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
CUDA_TRIPLE=x86_64-linux
CUBLAS_TRIPLE=x86_64-linux-gnu
DLSW_TRIPLE=x86_64-linux-gnu
ifeq ($(TARGET), aarch64)
CUDA_TRIPLE=aarch64-linux
CUBLAS_TRIPLE=aarch64-linux-gnu
DLSW_TRIPLE=aarch64-linux-gnu
endif
ifeq ($(TARGET), qnx)
CUDA_TRIPLE=aarch64-qnx
CUBLAS_TRIPLE=aarch64-qnx-gnu
DLSW_TRIPLE=aarch64-unknown-nto-qnx
endif
ifeq ($(TARGET), ppc64le)
CUDA_TRIPLE=ppc64le-linux
CUBLAS_TRIPLE=ppc64le-linux
DLSW_TRIPLE=ppc64le-linux
endif
ifeq ($(TARGET), android64)
DLSW_TRIPLE=aarch64-linux-androideabi
CUDA_TRIPLE=$(DLSW_TRIPLE)
CUBLAS_TRIPLE=$(DLSW_TRIPLE)
endif
export TARGET
export VERBOSE
export LIBDIR
export CUDA_TRIPLE
export CUBLAS_TRIPLE
export DLSW_TRIPLE

ifeq ($(SAFE_PDK), 1)
  # Only dlaSafetyRuntime is currently able to execute with safety pdk.
  samples = dlaSafetyRuntime
else
  samples = sampleAlgorithmSelector sampleCharRNN sampleDynamicReshape sampleFasterRCNN sampleGoogleNet sampleINT8 sampleINT8API sampleMLP sampleMNIST sampleMNISTAPI sampleNMT sampleMovieLens sampleOnnxMNIST sampleUffPluginV2Ext sampleReformatFreeIO sampleSSD sampleUffFasterRCNN sampleUffMaskRCNN sampleUffMNIST sampleUffSSD trtexec samplePlugin


  # sampleMovieLensMPS should only be compiled for Linux targets.
  # sample uses Linux specific shared memory and IPC libraries.
  ifeq ($(TARGET),x86_64)
    samples += sampleMovieLensMPS
  endif

  # sampleNvmedia/dlaSafetyRuntime/dlaSafetyBuilder should only be compiled with DLA enabled.
  ifeq ($(ENABLE_DLA),1)
	samples += sampleNvmedia
	samples += dlaSafetyRuntime
	samples += dlaSafetyBuilder
  endif
endif

.PHONY: all clean help
all:
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample) &&) :

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo -e "\t$(sample)" &&) :
	$(AT)echo -e "\nCommands:"
	$(AT)echo -e "\tall - build all samples."
	$(AT)echo -e "\tclean - clean all samples."
	$(AT)echo -e "\nVariables:"
	$(AT)echo -e "\tTARGET - Specify the target to build for."
	$(AT)echo -e "\tVERBOSE - Specify verbose output."
	$(AT)echo -e "\tCUDA_INSTALL_DIR - Directory where cuda installs to."
