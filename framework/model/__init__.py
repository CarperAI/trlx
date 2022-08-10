from abc import abstractmethod

from framework.data import RolloutStore, RLElement

class BaseRLModel:
    def __init__(self):
        self.store : RolloutStore = None
    
    @abstractmethod
    def act(data : RLElement) -> RLElement:
        """
        Given RLElement with state, produce an action and add it to the RLElement.
        Orchestrator should call this, get reward and push subsequent RLElement to RolloutStore
        """
        pass
    
    @abstractmethod
    def learn():
        """
        Use experiences in RolloutStore to learn
        """
