#!/bin/bash

# Copyright (c) 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo
echo "OpenVINO silent install, install openvino package"
echo
tar -xvzf l_openvino_toolkit_p_2019.3.334.tgz
cp silent.cfg l_openvino_toolkit_p_2019.3.334/silent.cfg
cd l_openvino_toolkit_p_2019.3.334/
sudo ./install.sh --silent=silent.cfg

echo
echo "OpenVINO Silent Install is finished"
echo

echo
echo "OpenVINO Toolkit dependencies installation using install_openvino_dependencies.sh"
echo
cd /opt/intel/openvino/install_dependencies

sudo -E ./install_openvino_dependencies.sh
	
echo
echo "Go to install Openvino Model Optimizer prerequisities"
echo "/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh"
echo
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
sudo ./install_prerequisites.sh

	
echo
echo "run verification script to verify installation"
echo "run demos under /opt/intel/openvino/deployment_tools/demo"
echo
cd /opt/intel/openvino/deployment_tools/demo

sudo ./demo_security_barrier_camera.sh

echo
echo "End of OPENVINO installation and vlidation"
echo	
