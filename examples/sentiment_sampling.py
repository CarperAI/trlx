from framework.model.sentiment import SentimentILQLModel
from framework.data.configs import TRLConfig
from framework.utils.loading import get_model

if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/sentiment_config.yml")

    model : SentimentILQLModel = get_model(cfg.model.model_type)(cfg, train_mode = False)
    model.load("./")

    samples = model.sample(["This movie was so"] * 20)
    for sample in samples:
        print(sample)
