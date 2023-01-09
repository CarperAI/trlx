
# to install correct versions
pip install --upgrade --force-reinstall numexpr==2.7.3 numpy==1.23.*
pip install --upgrade --force-reinstall torch --extra-index-url https://download.pytorch.org/whl/cu116

git config --global user.name "Jeremy Gillen"
git config --global user.email "jez.gillen@gmail.com"

# to launch using config file
accelerate launch --config_file configs/deepspeed_configs/default_configs.yml examples/simulacra_tmp.py