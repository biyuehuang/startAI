echo
echo "Dependencies installation Start"
echo

#python3 -m pip install testresources
echo
echo ! Start openvino samples Environment Initialization
echo
#sudo apt install -y python-pip
#apt-get -y install python-pip

python3 -m pip install --upgrade pip
python3 -m pip install jupyter

python3 -m pip install testresources
python3 -m pip install -U pip setuptools
#sudo python3 -m pip  install -U opencv-python matplotlib

python3 -m pip install opencv-python matplotlib
python3 -m pip install networkx==2.3
#pip3 install -U opencv-python matplotlib
#pip3 install networkx==2.3

#sudo pip3 install networkx==2.3

#export PATH=$PATH:~/.local/bin

apt-get -y install vim git libgflags-dev


echo
echo ! Start Annotation Tool Environment Initialization
echo

cur_path=$PWD
cd Annotation_Tool/labelImg/
apt-get install -y pyqt5-dev-tools
#python3 -m pip3 install -r requirements/requirements-linux-python3.txt
#sudo pip3 install -r requirements/requirements-linux-python3.txt
python3 -m pip install -r requirements/requirements-linux-python3.txt
make qt5py3
cd $cur_path

echo
echo ! Start Tensorflow_models Environment Initialization
echo

apt-get install -y protobuf-compiler python-pil python-lxml python-tk
#pip3 install Cython
#pip3 install contextlib2

#CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


tf_research_dir="$cur_path/Tensorflow_models/research"
cd $tf_research_dir
protoc object_detection/protos/*.proto --python_out=.

sudo apt-get install unzip
unzip -o protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

tf_slim_path="$cur_path/Tensorflow_models/research/slim/"
#cd $tf_slim_path
#wget https://pypi.python.org/packages/source/s/setuptools/setuptools-0.6c11.tar.gz
#tar zxvf setuptools-0.6c11.tar.gz
#cd setuptools-0.6c11
#sudo python3 setup.py install
cd $tf_slim_path
python3 setup.py build   
sudo python3 setup.py install

cd ../../..

echo
echo ! Finished Quick Start Demo Environment Initialization
echo

### Use Visualization Demo need to install tensorflow
