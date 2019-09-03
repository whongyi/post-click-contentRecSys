# install dependencies
sudo apt-get update -y
sudo apt-get install python3-pip

# pip install
pip3 install tensorflow
pip3 install numpy
pip3 install termcolor
pip3 install tqdm

# install openrec 
git clone https://github.com/whongyi/openrec.git
cd openrec
git checkout logging
sudo python3 setup.py install
