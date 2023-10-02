.. _installation:

Installation
============

trlX is a pure Python library that supports two optional distributed backends: `Huggingface 🤗 Accelerate <https://github.com/huggingface/accelerate>`_ and `NVIDIA NeMo <https://nvidia.github.io/NeMo>`_, the latter is optional and can be installed separately.

Requirements
------------

* OS: Linux
* Python: 3.9-3.11

Install with pip
----------------

You can install trlX using pip:

.. code-block:: console

    $ pip install -U git+https://github.com/CarperAI/trlx.git

.. _build_from_source:

Install from source
-------------------

You can also install trlX from source:

.. code-block:: console

    $ git clone https://github.com/CarperAI/trlx.git
    $ cd trlx
    $ pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    $ pip install -e .

Install NeMo
____________

Install NeMo version v1.17.0:

.. code-block:: console

    $ git clone https://github.com/NVIDIA/NeMo/
    $ cd NeMo
    $ git checkout d3017e4
    $ pip install -e '.[all]'

Install Apex:

.. code-block:: console

   $ git clone https://github.com/NVIDIA/apex
   $ cd apex
   $ # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
   $ pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
