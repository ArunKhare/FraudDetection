echo ["$(date)"] : "START"
echo ["$(date)"] : "Creating env with python 3.10 python"
conda create --prefix ./FraudDetection python=3.10 -y
echo ["$(date)"] : "activating the environment"
source activate ./FraudDetection
echo ["$(date)"] : "installing the dev requirements"
pip install -r requirements_dev.txt
echo["$(date)"] : "END"
