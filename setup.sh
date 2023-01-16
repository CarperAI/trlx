# Add personal config
if [ $1 = '-j' ]
then
  git config --global user.name "Jeremy Gillen"
  git config --global user.email "jez.gillen@gmail.com"
fi
if [ $1 = '-n' ]
then
  git config --global user.name "Nicky Pochinkov"
  git config --global user.email "work@nicky.pro"
fi
if [ $1 = '-a' ]
then
  git config --global user.name "Alan Cooney"
  git config --global user.email "alancooney@gmail.com"
fi


# to install correct versions
pip install --upgrade --force-reinstall numexpr==2.7.3 numpy==1.23.*
pip install --upgrade --force-reinstall torch --extra-index-url https://download.pytorch.org/whl/cu116

pip install -e .




# W&B
echo "
machine api.wandb.ai
  login user
  password 3544b54157a10a78836e75b7b161e42ac012c646" >> ~/.netrc


# ssh keys
"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICCdk4fVv4MPf4cYED0HQQ09XYdYZ7YBDzteqrHEckf1 jez.gillen@gmail.com
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCxPwvoy4ESaSs+UqeD/ByZsZ52SzaCoZNBY3gqGApH38r5R8vBBuXHyT0GezrDpDhIIzaaiJRtFojXn/5LCRo+a2pfVpMZ80nMD5o9bnxwUpF8JTGOQ7tArhYvYp66Bf+va2JCk9FJKDwRI/aAYTbUxbZ3j9kFEKJZ2n8Ka9s7qjW4eJLQ45W7++RTLzO7C9rEKGuLpQ3sEG8RzA6Oq8VRTtwTGKcfqECo2LdhQ6mdYlE6GF8D1YVD57G5AU6gLlE41lnKrA3xxlipSs5BZz/HSMzV5c56hNu09RMgKz+mg0r1sNvlF/8RhGBNpFudiFSWBq4JmJ2ukU0spVlkzuCPm8Sv+59IFNONYq6fRIsyxUF6qpKpljU4aaDb75ehEIqUcGhWd6r99rhUDdbZR5D3Aqo3S0w8ALQ4VsYhnS4yW9qqKrekNl4dPU6+qB3DVelWKDbw8bO7UuxXp838mUl7WtNcGFxo+3eqE+UreitCb3y9JJlJXJ5llzOKv9alzwM= mac@macs-MacBook-Air.local
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC36K0qvo1/t5bUY0Va8hJOLqePmBSr7HEP2Wqb6JGjManHZ03dZKV66FBKLIO8Y7i0zG0Z2zKluHM9k+IhQiBfiHOoVK6dUXfRW9UZZcNkjygxFNbU/EGoKZ11/rYy4XDxWOqtt1slTwMlu196dAROkUL12pjR6p6WQukV10Q5O3a06xpwCjjvAXEk6n5eNr86w4g3FOvAsJIKTXpdvgrRaqtk4GJbZuFKt/bVho7SlaNn1FfYZ6wQqHn0SscsZxM60bn6CAjiqitfv2AzdgannrYCl1JBbz7/JhJnF6AY4RtduOJPLM7w4srlI/lP595uLQbR9xtVmzrDUY14Gx/K5DueoaCkBXHZeYuAZ1sveebQ3S5841z+sP6HT+Qa7I8Oq7IRgaHrgrB2RDNyBrk3jKLcE0aAl8CCYES5beQio6bwwbJo8fw3PfbIYzyAZWr0ucTJ1TSnCH1JCo+ocZ+ZvXDliye/FUVIJqIR8Mx366bxFzWw/ZgNqEaXbca3kQk= alan@Siyas-MacBook-Air.local" > ~/.ssh/authorized_keys

# to launch using config file
accelerate launch --config_file configs/deepspeed_configs/default_configs.yml examples/simulacra_tmp.py