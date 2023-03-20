# Glossary of Terms

```{glossary}
[Agent]()
  An agent in reinforcement learning is the entity that interacts with the {term}`Environment` to learn how to maximize its {term}`Reward`.

[Action]()
  An action in reinforcement learning is the signal that the {term}`Agent` provides to the {term}`Environment` to indicate what it wants to do.

  In other words, an action is a scalar value that the agent provides to the environment to indicate what it wants to do. The agent's goal is to maximize the total reward it receives over a sequence of {term}`Steps`.

[CPU](https://en.wikipedia.org/wiki/Central_processing_unit)
  Short for *Central Processing Unit*, CPUs are the standard computational architecture
  available in most computers. trlX can run computations on CPUs, but often can achieve
  much better performance on {term}`GPU` .


[Device](https://en.wikipedia.org/wiki/Device_computing)
  The generic name used to refer to the {term}`CPU`, {term}`GPU`, used
  by TRLX to perform computations.

[Environment]()
  An environment in reinforcement learning is the system that the agent interacts with. It is the source of {term}`State`, {term}`Action`, and {term}`Reward`.

  In other words, an environment is a system that defines the agent's observation space, action space, and reward function. It is the source of the agent's experience, and the goal of the agent is to maximize the total reward it receives over a sequence of {term}`Steps`.

[GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit)
  Short for *Graphical Processing Unit*, GPUs were originally specialized for operations
  related to rendering of images on screen, but now are much more general-purpose. TRLX is
  able to target GPUs for fast operations on arrays (see also {term}`CPU`).

[Policy]()
  A policy in reinforcement learning is a function that maps {term}`State` to {term}`Action`.

  In other words, a policy is a function that maps the agent's current state to the action it should take. The agent's goal is to maximize the total reward it receives over a sequence of {term}`Steps`.

[PPO](https://arxiv.org/abs/1707.06347)
  Short for *Proximal Policy Optimization*, PPO is a {term}`Policy Gradient` algorithm
  that is able to learn policies in high-dimensional, continuous action spaces.

[Policy Gradient](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#policy-gradients)
  Policy gradient methods are a class of reinforcement learning algorithms that are able to
  learn policies in high-dimensional, continuous action spaces.

[Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
  Reinforcement learning (RL) is a machine learning paradigm that trains an agent to maximize its
  {term}`Reward` by interacting with an {term}`Environment`.

[Reward]()
  A reward in reinforcement learning is the signal that the {term}`Environment` provides to the {term}`Agent` to indicate how well it is performing.

  In other words, a reward is a scalar value that the environment provides to the agent to indicate how well it is performing. The agent's goal is to maximize the total reward it receives over a sequence of {term}`Steps`.

[Rollout]()
  A rollout in reinforcement learning is the process of executing a {term}`Policy`, starting from a specific state in the {term}`Environment`, and following it to the end to obtain a complete trajectory of {term}`State`, {term}`Action`, and {term}`Reward`.

  In other words, a Rollout is a simulation of a policy's behavior in the environment over a fixed number of {term}`Steps` or until a terminal state is reached. It provides a means of evaluating the {term}`Policy`'s performance, as the total reward collected over the trajectory can be used as a measure of its effectiveness.

[State]()
  A state in reinforcement learning is the observation that the {term}`Environment` provides to the {term}`Agent`.

[Steps]()
  A step in reinforcement learning is the process of taking a single {term}`Action` in the {term}`Environment`, and observing the resulting {term}`State` and {term}`Reward`.

  In other words, a step is a single iteration of the environment's dynamics, where the agent takes an action and receives a reward and a new state. The agent's goal is to maximize the total reward it receives over a sequence of steps.

[Trajectory]

  In a {term}`PPO` (Proximal Policy Optimization) setup, a fixed-length trajectory
  segment refers to a fixed number of time steps in an episode of an
  environment.At each time step, the agent takes an action based on the current
  state and receives a reward from the environment. By using fixed-length
  trajectory segments, the agent's behavior is divided into chunks of a fixed
  length, and each chunk is used for a single PPO update. This allows for more
  efficient use of the {term}`Agent`'s experience by breaking it into smaller pieces, and
  it also helps to stabilize the learning process by making the training updates
  less sensitive to the length of the episode. Fixed-length trajectory segments
  are often used in Reinforcement Learning (RL) algorithms, including {term}`PPO`, to
  update the policy network.

```
