sudo yum update -y
sudo yum install -y tmux htop git cmake
sudo yum groupinstall -y "Development Tools"

sudo yum erase nvidia cuda
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.66/NVIDIA-Linux-x86_64-375.66.run
sudo /bin/bash ./NVIDIA-Linux-x86_64-375.66.run
rm ./NVIDIA-Linux-x86_64-375.66.run

wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sudo bash cuda_8.0.61_375.26_linux-run
rm cuda_8.0.61_375.26_linux-run
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64" >> ~/.profile

sudo reboot

tmux

sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -ac 2505,875


wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
chmod u+x Anaconda3-4.3.1-Linux-x86_64.sh
./Anaconda3-4.3.1-Linux-x86_64.sh

conda install -y pytorch torchvision cuda80 -c soumith
conda install -y gensim scikit-learn networkx
conda install -y -c conda-forge xgboost
conda install -y -c spacy spacy

pip install plumbum distance pandas luigi tqdm hyperopt nltk nose mmh3 coloredlogs joblib networkx keras tensorflow-gpu


# LIGHTGBM INSTALL
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j
cd ../python-package
python3 setup.py install

cd
# /LIGHTGBM INSTALL

aws s3 cp s3://riri-machine-learning/kaggle-quora-datasets.tar.gz .
tar -xvvf kaggle-quora-datasets.tar.gz
rm kaggle-quora-datasets.tar.gz

mkdir Documents
sudo mount -t tmpfs -o size=16G tmpfs Documents/


# LOCALLY
rsync --progress -a --exclude .idea --exclude cache --exclude rf-cache --exclude \*.pyc kaggle-quora/ ec2-user@34.201.1.245:~/Documents/kaggle-quora
cd ~/Documents/kaggle-quora
export PYTHONPATH=.
luigi --workers=4 --module kq.refold.rf_stacker Stacker