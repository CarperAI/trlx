
# to install correct versions
pip install --upgrade --force-reinstall numexpr==2.7.3 numpy==1.23.*
pip install --upgrade --force-reinstall torch --extra-index-url https://download.pytorch.org/whl/cu116

pip install -e .

# Add personal config
git config --global user.name "Jeremy Gillen"
git config --global user.email "jez.gillen@gmail.com"
# W&B
echo "
machine api.wandb.ai
  login user
  password 3544b54157a10a78836e75b7b161e42ac012c646" >> ~/.netrc

# to launch using config file
accelerate launch --config_file configs/deepspeed_configs/default_configs.yml examples/simulacra_tmp.py