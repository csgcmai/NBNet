Clone the project from the github with 

git clone git@github.com:csgcmai/NBNet.git

Install the required packages, the usage of the conda enviornment is recommended.
Run the following commands

conda create -n nbnet python=2.7 anaconda
pip install --upgrade pip

Piror to the next step, please refer https://www.tensorflow.org/install/gpu 
and http://mxnet.io/install/index.html for installing the necessary packages 
for installing the tensorflow and mxnet with GPU support


cd NBNet
pip install -r requirement.txt

cd src
python train_of2img_mae.py --gpus 0,1,2,3





