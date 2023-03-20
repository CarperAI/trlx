.. trlX documentation master file, created by
   sphinx-quickstart on Mon Oct  3 21:21:33 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to trlX's documentation!
================================
trlX is a library made for training large language models using reinforcement learning. It
currently supports training using PPO or ILQL for models up to 20B using Accelerate.

Installation
------------
.. code-block:: bash

   pip install "trlx"


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   README
   data
   configs
   pipeline
   trainer

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Resources

   faq
   glossary


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
