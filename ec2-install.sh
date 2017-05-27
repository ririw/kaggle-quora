sudo yum update -y
sudo yum install -y tmux htop git cmake
sudo yum groupinstall -y "Development Tools"

tmux
sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -ac 2505,875

wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
chmod u+x Anaconda3-4.3.1-Linux-x86_64.sh
./Anaconda3-4.3.1-Linux-x86_64.sh

    sudo reboot

conda install -y pytorch torchvision cuda80 -c soumith
conda install -y -c conda-forge xgboost
conda install -y -c spacy spacy
conda install -y gensim scikit-learn networkx

pip install plumbum distance pandas tensorflow-gpu luigi tqdm hyperopt nltk nose mmh3 coloredlogs joblib networkx keras
pip install git+https://github.com/ncullen93/torchsample.git


# LIGHTGBM INSTALL
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j
cd ..
cd python-package;
python setup.py install

cd
# /LIGHTGBM INSTALL

aws s3 cp s3://riri-machine-learning/kaggle-quora-dataset.tar.gz .
tar -xvvf kaggle-quora-dataset.tar.gz

mkdir Documents
sudo mount -t tmpfs -o size=16G tmpfs Documents/
