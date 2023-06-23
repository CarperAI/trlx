## Using pretrained NeMo models
To use a NeMo models in `.nemo` format, like [NeMo Megatron-GPT-20B](https://huggingface.co/nvidia/nemo-megatron-gpt-20B), download and un-tar it:
```
tar xvf nemo_gpt20B_bf16_tp4.nemo
```
This will extract the model weights and the model config.

Then set `train.trainer_kwargs.pretrained_model` to the path to the directory containing the parameters. The model hyperparameters in the `train.trainer_kwargs.megatron_cfg` should match the ones in the model config.

## Inference ILQL trained NeMo models
To load a checkpoint, run
```
python examples/nemo_ilql_inference.py configs/nemo_configs/megatron_20b.yaml "/path/to/ilql_sentiments_logs/checkpoints"
```
To save checkpoints, ensure the following is set in the NeMo config:
```
exp_manager:
  explicit_log_dir: ilql_sentiments_logs
  create_checkpoint_callback: True
```

## Resume Training
To resume training, ensure the following is set in the NeMo config:
```
exp_manager:
  resume_if_exists: True
```

## NeMo Megatron setup
Clone https://github.com/NVIDIA/NeMo/tree/r1.15.0 (currently only up to `r1.15.0` is supoprted) and apex from https://github.com/NVIDIA/apex/.

1) install conda (or mamba/micromamba)

2) srun into a compute node with a gpu (if running on HPC cluster)
```
srun --pty bash -i
```

3) copy the conda env export below and change the name and prefix
```
conda env create -f env.yaml
```

4) install nemo
```
git clone https://github.com/NVIDIA/NeMo/
cd NeMo
git checkout r1.15.0
pip install '.[all]'
```

6) install apex (or clone the github)
```
git clone https://github.com/NVIDIA/apex/
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./
```


# conda env export
```
name: nemo-113
prefix: /mnt/nvme/jobs/nemo/nemo-source
channels:
  - anaconda
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_gnu
  - bzip2=1.0.8=h7f98852_4
  - c-ares=1.18.1=h7f8727e_0
  - ca-certificates=2022.9.24=ha878542_0
  - curl=7.84.0=h5eee18b_0
  - expat=2.4.4=h295c915_0
  - gettext=0.21.1=h27087fc_0
  - git=2.34.1=pl5262hc120c5b_0
  - krb5=1.19.2=hac12032_0
  - lame=3.100=h166bdaf_1003
  - ld_impl_linux-64=2.39=hcc3a1bd_1
  - libcurl=7.84.0=h91b91d3_0
  - libedit=3.1.20210910=h7f8727e_0
  - libev=4.33=h7f8727e_1
  - libffi=3.2.1=he1b5a44_1007
  - libflac=1.4.2=h27087fc_0
  - libgcc-ng=12.2.0=h65d4601_19
  - libgomp=12.2.0=h65d4601_19
  - libnghttp2=1.46.0=hce63b2e_0
  - libnsl=2.0.0=h7f98852_0
  - libogg=1.3.4=h7f98852_1
  - libopus=1.3.1=h7f98852_1
  - libsndfile=1.1.0=h27087fc_0
  - libsqlite=3.39.4=h753d276_0
  - libssh2=1.10.0=h8f2d780_0
  - libstdcxx-ng=12.2.0=h46fd767_19
  - libuuid=2.32.1=h7f98852_1000
  - libvorbis=1.3.7=h9c3ff4c_0
  - libzlib=1.2.12=h166bdaf_2
  - mpg123=1.30.2=h27087fc_1
  - ncurses=6.3=h27087fc_1
  - openssl=1.1.1q=h7f8727e_0
  - pcre2=10.37=he7ceb23_1
  - perl=5.26.2=h14c3975_0
  - pip=22.3.1=pyhd8ed1ab_0
  - python=3.8.2=he5300dc_7_cpython
  - readline=8.1.2=h0f457ee_0
  - sqlite=3.39.4=h4ff8645_0
  - tk=8.6.12=h1ccaba5_0
  - wheel=0.38.4=pyhd8ed1ab_0
  - xz=5.2.6=h166bdaf_0
  - zlib=1.2.12=h7f8727e_2
  - pip:
    - absl-py==1.3.0
    - aiohttp==3.8.3
    - aiosignal==1.3.1
    - alabaster==0.7.12
    - aniso8601==9.0.1
    - antlr4-python3-runtime==4.9.3
    - appdirs==1.4.4
    - asttokens==2.1.0
    - async-timeout==4.0.2
    - attrdict==2.0.1
    - attrs==22.1.0
    - audioread==3.0.0
    - babel==2.11.0
    - backcall==0.2.0
    - beautifulsoup4==4.11.1
    - black==19.10b0
    - boto3==1.26.13
    - botocore==1.29.13
    - braceexpand==0.1.7
    - cachetools==5.2.0
    - certifi==2022.9.24
    - cffi==1.15.1
    - charset-normalizer==2.1.1
    - click==8.0.2
    - colorama==0.4.6
    - commonmark==0.9.1
    - contourpy==1.0.6
    - cycler==0.11.0
    - cython==0.29.32
    - debugpy==1.6.3
    - decorator==5.1.1
    - distance==0.1.3
    - docker-pycreds==0.4.0
    - docopt==0.6.2
    - docutils==0.19
    - editdistance==0.6.1
    - einops==0.6.0
    - entrypoints==0.4
    - exceptiongroup==1.0.4
    - executing==1.2.0
    - faiss-cpu==1.7.3
    - fasttext==0.9.2
    - filelock==3.8.0
    - flask==2.2.2
    - flask-restful==0.3.9
    - fonttools==4.38.0
    - frozenlist==1.3.3
    - fsspec==2022.11.0
    - ftfy==6.1.1
    - g2p-en==2.1.0
    - gdown==4.5.3
    - gitdb==4.0.9
    - gitpython==3.1.29
    - google-auth==2.14.1
    - google-auth-oauthlib==0.4.6
    - grpcio==1.50.0
    - h5py==3.7.0
    - huggingface-hub==0.11.0
    - hydra-core==1.2.0
    - idna==3.4
    - ijson==3.1.4
    - imagesize==1.4.1
    - importlib-metadata==5.0.0
    - importlib-resources==5.10.0
    - inflect==6.0.2
    - iniconfig==1.1.1
    - ipadic==1.0.0
    - ipykernel==6.17.1
    - ipython==8.6.0
    - ipywidgets==8.0.2
    - isort==4.3.21
    - itsdangerous==2.1.2
    - jedi==0.18.1
    - jieba==0.42.1
    - jinja2==3.1.2
    - jiwer==2.5.1
    - jmespath==1.0.1
    - joblib==1.2.0
    - jupyter-client==7.4.7
    - jupyter-core==5.0.0
    - jupyterlab-widgets==3.0.3
    - kaldi-python-io==1.2.2
    - kaldiio==2.17.2
    - kiwisolver==1.4.4
    - latexcodec==2.0.1
    - levenshtein==0.20.2
    - librosa==0.9.2
    - llvmlite==0.39.1
    - loguru==0.6.0
    - lxml==4.9.1
    - markdown==3.4.1
    - markupsafe==2.1.1
    - marshmallow==3.19.0
    - matplotlib==3.6.2
    - matplotlib-inline==0.1.6
    - mecab-python3==1.0.5
    - mpmath==1.2.1
    - multidict==6.0.2
    - nest-asyncio==1.5.6
    - nltk==3.7
    - numba==0.56.4
    - numpy==1.23.4
    - nvidia-cublas-cu11==11.10.3.66
    - nvidia-cuda-nvrtc-cu11==11.7.99
    - nvidia-cuda-runtime-cu11==11.7.99
    - nvidia-cudnn-cu11==8.5.0.96
    - oauthlib==3.2.2
    - omegaconf==2.2.3
    - onnx==1.12.0
    - opencc==1.1.4
    - packaging==21.3
    - pandas==1.5.1
    - pangu==4.0.6.1
    - parameterized==0.8.1
    - parso==0.8.3
    - pathspec==0.10.2
    - pathtools==0.1.2
    - pesq==0.0.4
    - pexpect==4.8.0
    - pickleshare==0.7.5
    - pillow==9.3.0
    - pip-api==0.0.30
    - pipreqs==0.4.11
    - plac==1.3.5
    - platformdirs==2.5.4
    - pluggy==1.0.0
    - pooch==1.6.0
    - portalocker==2.6.0
    - progress==1.6
    - promise==2.3
    - prompt-toolkit==3.0.32
    - protobuf==3.20.1
    - psutil==5.9.4
    - ptyprocess==0.7.0
    - pure-eval==0.2.2
    - pyannote-core==4.5
    - pyannote-database==4.1.3
    - pyannote-metrics==3.2.1
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pybind11==2.10.1
    - pybtex==0.24.0
    - pybtex-docutils==1.0.2
    - pycparser==2.21
    - pydantic==1.10.2
    - pydeprecate==0.3.2
    - pydub==0.25.1
    - pygments==2.13.0
    - pynini==2.1.5
    - pyparsing==3.0.9
    - pypinyin==0.47.1
    - pysocks==1.7.1
    - pystoi==0.3.3
    - pytest==7.2.0
    - pytest-runner==6.0.0
    - python-dateutil==2.8.2
    - pytorch-lightning==1.7.7
    - pytz==2022.6
    - pyyaml==5.4.1
    - pyzmq==24.0.1
    - rapidfuzz==2.13.2
    - regex==2022.10.31
    - requests==2.28.1
    - requests-oauthlib==1.3.1
    - resampy==0.4.2
    - rich==12.6.0
    - rsa==4.9
    - ruamel-yaml==0.17.21
    - ruamel-yaml-clib==0.2.7
    - s3transfer==0.6.0
    - sacremoses==0.0.53
    - scikit-learn==1.1.3
    - scipy==1.9.3
    - sentence-transformers==2.2.2
    - sentencepiece==0.1.97
    - sentry-sdk==1.11.0
    - setproctitle==1.3.2
    - setuptools==59.5.0
    - shellingham==1.5.0
    - shortuuid==1.0.11
    - simplejson==3.18.0
    - six==1.16.0
    - smmap==5.0.0
    - snowballstemmer==2.2.0
    - sortedcontainers==2.4.0
    - soundfile==0.11.0
    - soupsieve==2.3.2.post1
    - sox==1.4.1
    - sphinx==5.3.0
    - sphinxcontrib-applehelp==1.0.2
    - sphinxcontrib-bibtex==2.5.0
    - sphinxcontrib-devhelp==1.0.2
    - sphinxcontrib-htmlhelp==2.0.0
    - sphinxcontrib-jsmath==1.0.1
    - sphinxcontrib-qthelp==1.0.3
    - sphinxcontrib-serializinghtml==1.1.5
    - stack-data==0.6.1
    - sympy==1.11.1
    - tabulate==0.9.0
    - tensorboard==2.11.0
    - tensorboard-data-server==0.6.1
    - tensorboard-plugin-wit==1.8.1
    - termcolor==2.1.0
    - text-unidecode==1.3
    - textdistance==4.5.0
    - texterrors==0.4.4
    - threadpoolctl==3.1.0
    - tokenizers==0.12.1
    - toml==0.10.2
    - tomli==2.0.1
    - torch==1.13.0
    - torchaudio==0.13.0
    - torchmetrics==0.10.3
    - torchvision==0.14.0
    - tornado==6.2
    - tqdm==4.64.1
    - traitlets==5.5.0
    - transformers==4.21.2
    - typed-ast==1.5.4
    - typer==0.7.0
    - typing-extensions==4.4.0
    - urllib3==1.26.12
    - wandb==0.13.5
    - wcwidth==0.2.5
    - webdataset==0.1.62
    - werkzeug==2.2.2
    - wget==3.2
    - widgetsnbextension==4.0.3
    - wrapt==1.14.1
    - yarg==0.1.9
    - yarl==1.8.1
    - youtokentome==1.0.6
    - zipp==3.10.0
```
