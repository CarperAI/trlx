from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.sentiment_ppo_orch import PPOSentimentOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.data.configs import TRLConfig
from trlx.utils.loading import get_model, get_pipeline, get_orchestrator

import wandb


if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/default_config.yml")

    sentiment_pipe = pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=torch.device(0))
    def reward_fn(samples: List[str]) -> List[float]:
		"""
		Batched scoring function taking text and generating scalar
		"""
		sent_kwargs = {
				"return_all_scores": True,
				"function_to_apply": None,
				"batch_size": cfg.chunk_size,
			}
		pipe_outputs = sentiment_pipe(samples, **sent_kwargs)
		scores = torch.tensor([output[1]["score"] for output in pipe_outputs])
		return scores


    model : AcceleratePPOModel = get_model(cfg.model.model_type)(cfg)
    wandb.watch(model.model)

    pipeline : PPOPipeline = get_pipeline(cfg.train.pipeline)(model.tokenizer, cfg)
    orch : PPOOrchestrator = get_orchestrator(cfg.train.orchestrator)(model, pipeline, cfg.method.chunk_size, reward_fn=reward_fn)
    orch.make_experience(cfg.method.num_rollouts)
    model.learn()

    print("DONE!")
