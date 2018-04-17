# pytorch-cartpole
Adapted from the PyTorch examples found [here](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).



## Notes from the file:

Reinforcement Learning (DQN) tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_


This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
on the CartPole-v0 task from the `OpenAI Gym <https://gym.openai.com/>`__.

**Task**

The agent has to decide between two actions - moving the cart left or
right - so that the pole attached to it stays upright. You can find an
official leaderboard with various algorithms and visualizations at the
`Gym website <https://gym.openai.com/envs/CartPole-v0>`__.

.. figure:: /_static/img/cartpole.gif
   :alt: cartpole

   cartpole

As the agent observes the current state of the environment and chooses
an action, the environment *transitions* to a new state, and also
returns a reward that indicates the consequences of the action. In this
task, the environment terminates if the pole falls over too far.

The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (position, velocity, etc.).
However, neural networks can solve the task purely by looking at the
scene, so we'll use a patch of the screen centered on the cart as an
input. Because of this, our results aren't directly comparable to the
ones from the official leaderboard - our task is much harder.
Unfortunately this does slow down the training, because we have to
render all the frames.

Strictly speaking, we will present the state as the difference between
the current screen patch and the previous one. This will allow the agent
to take the velocity of the pole into account from one image.

**Packages**

First, let's import needed packages. Firstly, we need
`gym <https://gym.openai.com/docs>`__ for the environment
(Install using `pip install gym`).
We'll also use the following from PyTorch:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)
-  utilities for vision tasks (``torchvision`` - `a separate
   package <https://github.com/pytorch/vision>`__).

Replay Memory
-------------

We'll be using experience replay memory for training our DQN. It stores
the transitions that the agent observes, allowing us to reuse this data
later. By sampling from it randomly, the transitions that build up a
batch are decorrelated. It has been shown that this greatly stabilizes
and improves the DQN training procedure.

For this, we're going to need two classses:

-  ``Transition`` - a named tuple representing a single transition in
our environment
-  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
transitions observed recently. It also implements a ``.sample()``
method for selecting a random batch of transitions for training.

Now, let's define our model. But first, let quickly recap what a DQN is.

DQN algorithm
-------------

Our environment is deterministic, so all equations presented here are
also formulated deterministically for the sake of simplicity. In the
reinforcement learning literature, they would also contain expectations
over stochastic transitions in the environment.

Our aim will be to train a policy that tries to maximize the discounted,
cumulative reward
:math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
:math:`R_{t_0}` is also known as the *return*. The discount,
:math:`\gamma`, should be a constant between :math:`0` and :math:`1`
that ensures the sum converges. It makes rewards from the uncertain far
future less important for our agent than the ones in the near future
that it can be fairly confident about.

The main idea behind Q-learning is that if we had a function
:math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
us what our return would be, if we were to take an action in a given
state, then we could easily construct a policy that maximizes our
rewards:

.. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)

However, we don't know everything about the world, so we don't have
access to :math:`Q^*`. But, since neural networks are universal function
approximators, we can simply create one and train it to resemble
:math:`Q^*`.

For our training update rule, we'll use a fact that every :math:`Q`
function for some policy obeys the Bellman equation:

.. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))

The difference between the two sides of the equality is known as the
temporal difference error, :math:`\delta`:

.. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))

To minimise this error, we will use the `Huber
loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
like the mean squared error when the error is small, but like the mean
absolute error when the error is large - this makes it more robust to
outliers when the estimates of :math:`Q` are very noisy. We calculate
this over a batch of transitions, :math:`B`, sampled from the replay
memory:

.. math::

 \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)

.. math::

 \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
   \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
   |\delta| - \frac{1}{2} & \text{otherwise.}
 \end{cases}

Q-network
--

Our model will be a convolutional neural network that takes in the
difference between the current and previous screen patches. It has two
outputs, representing :math:`Q(s, \mathrm{left})` and
:math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
network). In effect, the network is trying to predict the *quality* of
taking each action given the current input.

Input extraction
--

The code below are utilities for extracting and processing rendered
images from the environment. It uses the ``torchvision`` package, which
makes it easy to compose image transforms. Once you run the cell it will
display an example patch that it extracted.

Training
--------

Hyperparameters and utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This cell instantiates our model and its optimizer, and defines some
utilities:

-  ``Variable`` - this is a simple wrapper around
 ``torch.autograd.Variable`` that will automatically send the data to
 the GPU every time we construct a Variable.
-  ``select_action`` - will select an action accordingly to an epsilon
 greedy policy. Simply put, we'll sometimes use our model for choosing
 the action, and sometimes we'll just sample one uniformly. The
 probability of choosing a random action will start at ``EPS_START``
 and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
 controls the rate of the decay.
-  ``plot_durations`` - a helper for plotting the durations of episodes,
 along with an average over the last 100 episodes (the measure used in
 the official evaluations). The plot will be underneath the cell
 containing the main training loop, and will update after every
 episode.

Training loop
--

Finally, the code for training our model.

Here, you can find an ``optimize_model`` function that performs a
single step of the optimization. It first samples a batch, concatenates
all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
:math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
state. We also use a target network to compute :math:`V(s_{t+1})` for
added stability. The target network has its weights kept frozen most of
the time, but is updated with the policy network's weights every so often.
This is usually a set number of steps but we shall use episodes for
simplicity.


Below, you can find the main training loop. At the beginning we reset
the environment and initialize the ``state`` variable. Then, we sample
an action, execute it, observe the next screen and the reward (always
1), and optimize our model once. When the episode ends (our model
fails), we restart the loop.

Below, `num_episodes` is set small. You should download
the notebook and run lot more episodes.
