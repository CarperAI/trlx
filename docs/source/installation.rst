.. _installation:

Installation
============

trlX is a Python library that supports two optional distributed backends (Accelerate and NeMO) that can be installed separately

Requirements
------------

* OS: Linux
* Python: 3.9

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
