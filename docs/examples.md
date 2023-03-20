# Examples

In the `examples` folder you can find several example training tasks.

Check the configs folder for the associated configs files.

## randomwalks

does offline reinforcement on a set of graph random walks to stitch shortest paths
to some destination.

## simulacra

optimizes prompts by using [prompts-ratings dataset](https://github.com/JD-P/simulacra-aesthetic-captions).

## architext

tries to optimize designs represented textually by minimizing number of rooms (pre-trained model is under a license on hf).

## ilql_sentiments and ppo_sentiments

train to generate movie reviews with a positive sentiment, in offline setting – by fitting to IMDB
dataset sentiment scores, and in online setting – by sampling finetuned on IMDB
model and rating samples with learned sentiment reward model, You can tweak
these scripts to your liking and tune hyperparameters to your problem if you
wish to use trlx for some custom task.
