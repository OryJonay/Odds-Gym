*****************************
Theory behind the environment
*****************************

TL;DR (or, I had enough of math for a lifetime)
###############################################

The theory behind the environment is quite simple- placing a bet is choosing an
possible outcome and a stake size, multiply it by the winning odds and subtract
the initial bet and any losses.

The scenic route
################

After this (very) brief explanation, let's get down to business:

Definitions
***********

Let :math:`g` be a game with :math:`N\in\mathbb{N}` distinct possible outcomes, and let :math:`p_i` be the probability
of outcome :math:`g_i`, so that :math:`\sum\limits_{i \in N} p_i = 1`. Let :math:`G` be the set of games so that :math:`g_j` is the :math:`j^{th}` game.

Let :math:`o` be the decimal odds for a game :math:`g`, so that :math:`o_i` is the decimal odd for outcome :math:`g_i`.
Let :math:`O` be the set of odds so that :math:`o_j` is the odds for game :math:`g_j` in :math:`G`.

.. math::

    o \triangleq \{o_i| i \in I_N ,  1 \leq o_i \}

Let :math:`r` be the result of game :math:`g`, so that :math:`r_i` is 1 if the outcome was :math:`i` and 0 if not
(in other words, :math:`r` is the indicator of game :math:`g`).
Let :math:`R` be the set of results so that :math:`r_j` is the result for game :math:`g_j` in :math:`G`.

.. math::

    r_i \triangleq \begin{cases}
        1 & \text{if outcome i happened,}\\
        0 & \text{otherwise.}
    \end{cases}

.. math::

    r \triangleq \{r_i | i \in I_N\}

Let :math:`b` be the bet for game :math:`g`, so that :math:`b_i` is the amount of money to place on outcome :math:`g_i`
Let :math:`B` be the set of bets so that :math:`b_j` is the bets for game :math:`g_j` in :math:`G`, and let
:math:`BANK \in \mathbb{R}^+` be the limit of the bet, so that the sum of the money placed can not exceed :math:`BANK`.

.. math::

    b \triangleq \left\{b_i | b_i \in \mathbb{R}, \sum\limits_{i \in N} b_i \leq BANK\right\}

Let :math:`M_O` be a matrix of size :math:`|G| \times N` of the numerical values of the set :math:`O`. In a similar manner
we'll define :math:`M_R` for the numerical values of :math:`R` and :math:`M_B` for the numerical values of the :math:`B`.

Let :math:`W` be the winnings matrix, defined by the |Hadamard product link| of all the matrices defined above like this:

.. math::

    W \triangleq M_B \circ M_R \circ M_O

So, the winnings on the set of games :math:`G` is grand sum (the sum of all elements) of :math:`W`.

The Environments
****************

All the environments implemented are a subclass of an |openai gym environments|,
with an observation space that equals to the odds (:math:`O`) and an action space that equals to the bets (:math:`B`).
A step in the environments is simply getting the current subset of odds and placing a bet on them, calculating the reward
by using the grand sum of :math:`W` and subtracting the total amount of the bet placed.
An episode is reached when there are no more games or when the :math:`BANK` is depleted.

.. |Hadamard product link| raw:: html

   <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)" target="_blank">Hadamard product</a>

.. |openai gym environments| raw:: html

   <a href="http://gym.openai.com/docs/#environments" target="_blank">OpenAI Gym envrionment</a>