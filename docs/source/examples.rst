.. _examples:

Examples
************************

In the ``examples`` folder you can find several example training tasks. Check the configs folder for the associated configs
files. ``examples.ilql_randomwalks`` trains an ILQL model on random data, whereass ``examples.ilql_sentiments``
and ``examples.ppo_sentiments`` train ILQL and PPO respectively on the IMDB review dataset to generate
reviews with high sentiments. You can tweak these scripts to your liking if you wish to use trlX for
your own custom tasks. Note that this may require that you write your own data pipeline,
and potentially your own orchestrator for the task.
