from abc import abstractmethod

from framework.pipeline import BasePipeline
from framework.model import BaseRLModel

class Orchestrator:
    def __init__(self, pipeline : BasePipeline, rl_model : BaseRLModel):
        self.pipeline = pipeline
        self.rl_model = rl_model

    @abstractmethod
    def make_experience(self):
        """
        Draw from pipeline, get action, generate reward
        Push to models RolloutStorage
        """
        pass