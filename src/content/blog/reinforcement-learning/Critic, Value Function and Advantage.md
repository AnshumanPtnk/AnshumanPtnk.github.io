---
title: Critic, Value Function and Advantage
subtitle: Why the critic learns the average return and why the advantage remains the learning signal
description: An intuitive account of why the critic learns the expected return rather than individual outcomes, why the advantage doesn't vanish as the critic converges, how the value-function baseline reduces variance without bias, and how this connects to GRPO's group-relative baseline.
author: Anshuman Patnaik
pubDatetime: 2026-07-08T20:05:00+05:30
category: "Reinforcement Learning"
readingTime: "38 min read"
tags:
  - Reinforcement Learning
  - Actor Critic
  - PPO
  - Value Function
  - Advantage
draft: false
last_updated: 2026-07-06
---

> [!abstract]
> **The Elevator Pitch**
>
> A critic does not try to reproduce every return observed on every trajectory. It learns the expected return from a state, while an individual rollout can finish above or below that expectation. Their difference is the advantage—the signal that tells the actor whether a sampled action performed better or worse than usual. This note resolves why the advantage does not vanish when the critic becomes accurate, explains how the baseline reduces variance, and connects the learned critic in actor–critic methods with GRPO's group-relative baseline.

---

# Contents

[[#1. Introduction: The Apparent Paradox]]
[[#2. What Is the Critic Trying to Learn?]]
[[#3. Why the Value Function is an Expectation]]
[[#4. Return is a Random Variable]]
[[#5. Why the Advantage Isn't Zero]]
[[#6. How the Critic Actually Learns]]
[[#7. Why the Critic Is Never Perfect]]
[[#8. Why Advantage Reduces Variance]]
[[#9. Connection with GRPO]]
[[#10. Summary]]
[[#Key Takeaways]]

---

# 1. Introduction: The Apparent Paradox

One of the first questions that arises when studying Actor-Critic methods is deceptively simple.

The critic is trained to predict the return $G_t.$The advantage is then computed as

$$
A_t = G_t - V(s_t).
$$

At first glance, these two equations appear contradictory. If the critic is explicitly trained to predict the return, then after sufficient training we might expect

$$
V(s_t)=G_t.
$$

Substituting this into the definition of the advantage immediately gives

$$
A_t=0.
$$

If the advantage always became zero, the policy gradient would vanish,

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta
\log\pi_\theta(a|s)
A(s,a)
\right]
=
0,
$$

and learning would stop completely.

Yet this clearly does not happen in practice. Actor-Critic algorithms continue to learn effectively even after the critic has become highly accurate.

This apparent contradiction often leads beginners to conclude that either the critic is never properly trained or that the definition of the advantage is somehow inconsistent.

Neither conclusion is correct.

The resolution lies in understanding **what the critic is actually trying to predict**. The critic is **not** attempting to memorize every observed return. Instead, it learns the **expected return** obtainable from a state. Individual trajectories almost never equal this expectation exactly, and it is precisely this difference between an observed return and its expectation that gives rise to the advantage.

This seemingly small distinction is one of the most important ideas in modern Reinforcement Learning. It explains not only why Actor-Critic methods continue to learn after the critic has converged, but also why subtracting the value function dramatically reduces the variance of policy gradient estimates without introducing bias.

The remainder of this note develops this idea from first principles.

---

# 2. What Is the Critic Trying to Learn?

To understand the behaviour of the critic, we must first clarify the meaning of the value function.

The state-value function of a policy $\pi$ is defined as

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s
\right].
$$

This definition is often read quickly, but every symbol is significant.

The expectation operator indicates that the critic is **not** predicting a single observed return. Instead, it predicts the average return that would be obtained if the agent were to begin from state $s$ and repeatedly follow the policy $\pi$ over an infinite number of hypothetical episodes.

The conditioning on the state,

$$
S_t=s,
$$

is equally important. The value function depends only on the current state. It does not depend on the particular trajectory that happened to occur in one episode. Many different trajectories may originate from exactly the same state, each producing a different return because of stochastic actions, stochastic environment transitions, or random rewards.

Consequently, the return associated with a state is not a fixed quantity but a random variable possessing an entire probability distribution. The critic compresses this distribution into a single representative statistic—its expected value.

This observation immediately explains why the critic cannot hope to reproduce every observed return. A single number cannot simultaneously equal every possible outcome of a random variable. Instead, it learns the value that best summarizes all of those outcomes in the least-squares sense.

The next section formalizes this idea by showing that minimizing the critic's mean squared error necessarily causes it to converge to the expected return.

---

# 3. Why the Value Function is an Expectation

The previous section established that the critic is trained to predict the value function

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s
\right].
$$

At this point, however, an obvious question remains.

> **Why should the critic learn the expected return rather than the most recent return, the largest return, or some other statistic?**

The answer lies in the objective function used to train the critic.

In almost every Actor-Critic algorithm, the critic is trained by minimizing the **Mean Squared Error (MSE)** between its prediction and the observed return. For a fixed state $s$, the loss function can be written as

$$
\mathcal{L}(v)
=
\mathbb{E}
\left[
(G-v)^2
\right],
$$

where

- $G$ denotes the observed return,
- $v$ is the critic's prediction for that state.

Notice that, for the purpose of this derivation, we treat $v$ as a single scalar representing the critic's prediction for one particular state.

The objective is therefore to find the value of $v$ that minimizes the average squared prediction error.

---

## Finding the Optimal Prediction

To determine the optimal prediction, we differentiate the loss with respect to $v$.

Expanding the loss,

$$
\mathcal{L}(v)
=
\mathbb{E}
\left[
G^2
-
2Gv
+
v^2
\right].
$$

Using the linearity of expectation,

$$
=
\mathbb{E}[G^2]
-
2v\,
\mathbb{E}[G]
+
v^2.
$$

Differentiating with respect to $v$ gives

$$
\frac{d\mathcal{L}}{dv}
=
-2\mathbb{E}[G]
+
2v.
$$

The minimum occurs when the derivative equals zero,

$$
-2\mathbb{E}[G]
+
2v
=
0.
$$

Rearranging,

$$
\boxed{
v
=
\mathbb{E}[G].
}
$$

This result is fundamental.

Whenever a model minimizes the mean squared prediction error, the optimal prediction is **the expected value of the target distribution**.

Consequently,

$$
\boxed{
V^\pi(s)
=
\mathbb{E}
[G_t|S_t=s].
}
$$

The critic is therefore not memorizing individual returns. It is estimating the conditional expectation of the return distribution.

---

## Why Not the Maximum Return?

A common misunderstanding is that the critic should predict the highest return ever observed from a state.

Suppose that, starting from a particular state, five different episodes produce the following returns:

$$
10,\;2,\;6,\;0,\;12.
$$

Should the critic predict

$$
12?
$$

If it did, the prediction errors would be

$$
2,\;10,\;6,\;12,\;0,
$$

which are clearly very large.

Instead, minimizing the squared error causes the critic to predict

$$
\frac{10+2+6+0+12}{5}
=
6.
$$

Although the value six is not equal to any particular return in this example, it minimizes the average squared prediction error across **all** observed trajectories.

This illustrates an important principle.

The critic is not attempting to explain one episode.

It is attempting to summarize the behaviour of **every possible future episode** beginning from the same state.

The expectation is therefore the statistically optimal prediction.

---

## The Meaning of the Value Function

The derivation above provides a deeper interpretation of the value function.

The value function is **not** the reward obtained in one trajectory.

It is **not** the largest reward that has ever been observed.

It is **not** the reward obtained by taking a particular action.

Instead, it is the **average long-term return** that would be obtained if the current policy were executed repeatedly from the same state over an infinite number of hypothetical episodes.

The critic therefore answers the following question:

> **"If I repeatedly start from this state and continue following the current policy, what return should I expect on average?"**

The word *expectation* is therefore not merely mathematical notation—it is the precise quantity the critic is designed to estimate.

---

# 4. Return is a Random Variable

The previous derivation raises another important question.

If the critic predicts only a single number for each state, why do we observe different returns every time we visit that state?

The answer is that the return is **not a deterministic quantity**.

Instead, $G_t$ is a random variable whose value depends on everything that happens after the current state.

To understand this, consider an agent standing at the entrance to a maze.

Although the starting state is identical in every episode, the future is uncertain. The agent may choose different actions because its policy is stochastic. Even if the same actions are selected, the environment itself may be stochastic, causing different state transitions.

The rewards obtained along the trajectory may also contain randomness.

Finally, the episode may terminate at different times. Each of these sources of uncertainty changes the accumulated return.

Consequently, beginning from exactly the same state may produce many different values of $G_t.$ The return should therefore be viewed not as a single number but as a probability distribution.

---

## Sources of Randomness

Several factors contribute to the variability of the return.

**Stochastic policies.** Many reinforcement learning algorithms deliberately sample actions from a probability distribution rather than selecting the highest-probability action. Two identical visits to the same state may therefore produce entirely different trajectories.

**Stochastic environments.** Real-world environments are rarely deterministic. Wind affects the motion of a drone, traffic changes the travel time of a vehicle, and human behaviour introduces uncertainty into interactive systems.

**Delayed rewards.** The return depends upon every future reward,

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+\cdots,
$$

so even small differences early in a trajectory may accumulate into substantially different total returns.

These factors combine to produce an entire distribution of possible returns associated with a single state.

---

## One State, Many Returns

Suppose the agent visits the same state five times during training and observes the following returns.

| Episode | Return |
|---------:|-------:|
| 1 | 10 |
| 2 | 2 |
| 3 | 6 |
| 4 | 0 |
| 5 | 12 |

Although the starting state is identical in every episode, the outcomes differ considerably.

Graphically,

```text
State s

├── Episode 1 → Return = 10
├── Episode 2 → Return = 2
├── Episode 3 → Return = 6
├── Episode 4 → Return = 0
└── Episode 5 → Return = 12
```

The critic cannot predict five different values for the same state.

Instead, it compresses the entire return distribution into its expectation,

$$
V(s)=6.
$$

This is exactly the quantity derived in the previous section by minimizing the mean squared error.

---

## An Important Consequence

We now arrive at the central insight of this note.

Since the critic predicts the **mean** of the return distribution, while each observed trajectory produces **one sample** from that distribution, the two quantities almost never coincide.

Formally,

$$
G_t
\neq
V(s_t)
\quad
\text{for most trajectories.}
$$

This difference is not a failure of the critic.

It is an unavoidable consequence of predicting the expectation of a random variable using a single deterministic number.

In fact, this difference between the observed return and its expected value is precisely what the advantage function measures.

The next section shows that the advantage is simply the deviation of one observed trajectory from the average behaviour predicted by the critic. This observation resolves the apparent paradox introduced at the beginning of the chapter and explains why the advantage continues to provide a meaningful learning signal even after the critic has converged.

---

# 5. Why the Advantage Isn't Zero

We are now in a position to resolve the paradox posed at the beginning of this note.

The critic is trained to predict the expected return,

$$
V(s)
=
\mathbb{E}[G|s].
$$

The advantage is defined as

$$
A(s,a)
=
G-V(s).
$$

At first glance, one might expect the critic to become so accurate after sufficient training that

$$
V(s)=G,
$$

causing

$$
A(s,a)=0.
$$

The previous chapters now allow us to see why this conclusion is incorrect.

The critic is **not** trained to predict the return observed in a particular episode. Instead, it predicts the average return over **all possible trajectories** originating from the same state under the current policy. Every observed return is merely one realization of a random variable whose expectation is given by the value function.

Consequently, the advantage compares two fundamentally different quantities.

The first, $G,$ is a **sample** drawn from the return distribution. The second, $V(s),$ is the **mean** of that distribution.

Subtracting the mean from one sample rarely produces zero. Instead, it measures how much better or worse that particular trajectory was than the average trajectory expected from the same state.

---

## The Advantage Measures Surprise

One of the most useful ways to interpret the advantage is as a measure of **surprise**.

Suppose that, from a particular state, the critic predicts

$$
V(s)=6.
$$

This prediction represents the average return expected from that state.

Now imagine observing several trajectories beginning from the same state.

| Episode | Return | Value Prediction | Advantage |
|---------:|-------:|-----------------:|----------:|
| 1 | 10 | 6 | +4 |
| 2 | 2 | 6 | -4 |
| 3 | 6 | 6 | 0 |
| 4 | 0 | 6 | -6 |
| 5 | 12 | 6 | +6 |

Notice that the critic remains unchanged throughout these episodes because the state is the same. What changes is the sampled trajectory and, consequently, the observed return.

Each advantage simply measures the deviation of one realisation from its expected value.

- Positive advantages indicate outcomes that were better than expected.

- Negative advantages indicate outcomes that were worse than expected.

- An advantage of zero indicates that the observed return exactly matched the critic's prediction.

The advantage is therefore **not** intended to vanish. On the contrary, it is designed to capture precisely the unpredictable component of the return that cannot be explained solely by knowing the current state.

---

## A Statistical Interpretation

The relationship between the return and the value function is identical to a familiar concept from elementary statistics.

Suppose a class of students scores

$$
\{72,\;81,\;65,\;90,\;82\}
$$

marks in an examination. The average score is $78.$

For each student, we may compute the deviation from the class average.

| Student | Score | Mean | Deviation |
|---------:|------:|-----:|----------:|
| A | 72 | 78 | -6 |
| B | 81 | 78 | +3 |
| C | 65 | 78 | -13 |
| D | 90 | 78 | +12 |
| E | 82 | 78 | +4 |

No one would expect every student's deviation to become zero merely because the class average has been computed correctly.

The average summarizes the distribution; it does not eliminate the variability within that distribution.

Exactly the same principle applies to the critic.

The value function summarizes the return distribution by predicting its mean.

The advantage measures how much an individual trajectory deviates from that mean.

Seen from this perspective, the existence of non-zero advantages is not surprising at all. It is an unavoidable consequence of the variability inherent in the return distribution.

---

## Why the Average Advantage Is Zero

Although individual advantages are generally non-zero, they possess an important statistical property.

Taking the expectation of the advantage conditioned on the state gives

$$
\mathbb{E}[A(s,a)\mid s]
=
\mathbb{E}[G\mid s]
-
V(s).
$$

Using the definition of the value function,

$$
V(s)
=
\mathbb{E}[G\mid s],
$$

we immediately obtain

$$
\boxed{
\mathbb{E}[A(s,a)\mid s]
=
0.
}
$$

This result is one of the cornerstones of policy gradient methods.

It does **not** imply that every advantage equals zero.

Instead, it states that **positive and negative advantages balance each other on average**.

Some trajectories perform better than expected.

Others perform worse.

Across many trajectories, these deviations cancel out.

This property allows the advantage to serve as a centered learning signal. Rather than rewarding every action simply because it occurs in a high-value state, the advantage rewards only those actions that perform better than what the critic already expected from that state.

---

## Why This Matters for the Actor

The actor's objective is not to identify states with large returns. The critic already performs that task.

Instead, the actor must answer a different question.

> **Given the current state, was the selected action better or worse than what I normally expect to happen?**

The advantage provides exactly this information.

Suppose the agent reaches a highly valuable state. The critic predicts $V(s)=100.$ One action produces a return of $105,$ while another produces $95.$

If the actor were trained directly on the return, both actions would appear highly desirable because both returns are large.

However, relative to the state's expected value, the first action exceeded expectations while the second fell short.

The corresponding advantages become

$$
105-100=+5,
$$

and

$$
95-100=-5.
$$

The actor therefore increases the probability of the first action and decreases the probability of the second.

The critic removes the component of the return that is attributable to the state itself, leaving behind only the contribution of the chosen action.

---

## The Central Insight

We can now resolve the apparent paradox completely.

The critic is trained to predict the **expected return**, not the return observed in an individual trajectory.

The return remains a random variable even after the critic has converged.

Consequently,

$$
G
-
V(s)
$$

does not measure prediction error in the usual sense. Instead, it measures how much one sampled trajectory differs from the average behaviour expected from that state.

This quantity is precisely the information required by the actor. Actions that consistently produce returns above expectation should become more probable, while actions that consistently produce returns below expectation should become less probable.

The advantage therefore remains the fundamental learning signal throughout actor-critic training. Rather than disappearing as the critic improves, it becomes increasingly meaningful because it isolates the contribution of the action from the intrinsic quality of the state itself.

> [!important]
> The critic learns the **average return** for a state, while the return observed during training is only **one sample** from the underlying return distribution. The advantage is the difference between these two quantities. It is therefore a measure of **deviation from expectation**, not **prediction error**, and it is this deviation that tells the actor whether the chosen action was better or worse than expected.


---

# 6. How the Critic Actually Learns

The previous sections explained **what** the critic is trying to learn. It estimates the expected return associated with each state by minimizing the mean squared error between its prediction and the observed returns. An equally important question, however, is **how** this learning actually takes place.

A common misconception is that every time a new return is observed, the critic simply replaces its previous prediction with that return. If this were true, the critic would exactly match the most recent trajectory, and the advantage would indeed become zero for that sample. Fortunately, this is not how learning occurs.

Instead, every observed return contributes only a **small correction** to the critic's current estimate. The critic therefore behaves like a running average that gradually incorporates information from many trajectories rather than memorizing any single episode.

---

## The Critic's Objective

Recall that the critic minimizes the mean squared error between its prediction and the observed return,

$$
\mathcal{L}(\theta_v)
=
\frac12
\left(
V_{\theta_v}(s)-G
\right)^2,
$$

where

- $V_{\theta_v}(s)$ is the critic's current prediction,
- $G$ is the sampled return,
- $\theta_v$ denotes the parameters of the value network.

The factor of one-half is included only for mathematical convenience, as it simplifies the derivative.

Unlike the actor, whose objective is to maximize expected return, the critic performs a supervised learning task. Given a state as input and a sampled return as the target, it adjusts its parameters to reduce the prediction error.

---

## A Single Gradient Update

To understand how the critic changes after observing one trajectory, let us first consider the simplest possible case in which the value estimate itself is the optimization variable.

The loss is

$$
\mathcal{L}(V)
=
\frac12
(V-G)^2.
$$

Differentiating with respect to the prediction,

$$
\frac{\partial\mathcal{L}}{\partial V}
=
V-G.
$$

Applying one step of gradient descent,

$$
V
\leftarrow
V
-
\alpha
(V-G),
$$

where $\alpha$ is the learning rate.

Rearranging,

$$
\boxed{
V
\leftarrow
V
+
\alpha
(G-V).
}
$$

This simple equation is one of the most important update rules in Reinforcement Learning.

Rather than replacing the current prediction with the newly observed return, the critic moves only a fraction of the distance toward it.

The quantity

$$
G-V
$$

is often called the **temporal prediction error** in the Monte Carlo setting because it measures how far the current prediction lies from the observed return.

---

## Why the Critic Never Equals the Current Return

The update equation immediately explains why the critic never becomes identical to the latest sampled return.

Suppose the critic currently predicts $V(s)=6,$ and the latest episode produces a return of $G=10.$ If the learning rate is $\alpha=0.1,$ then

$$
V
\leftarrow
6
+
0.1(10-6)
=
6.4.
$$

Notice that the critic does **not** jump directly to 10. Instead, it moves only ten percent of the way toward the new observation.

If the next episode produces a return of 2, the critic becomes

$$
V
\leftarrow
6.4
+
0.1(2-6.4)
=
5.96.
$$

Each trajectory therefore pulls the critic slightly in its own direction. No single trajectory dominates the learning process.

Over many episodes, these small adjustments accumulate, and the critic gradually converges toward the expected return rather than any individual sample.

---

## Learning the Average Through Repeated Updates

To see this process more clearly, consider the same sequence of returns introduced earlier,

$$
\{10,\;2,\;6,\;0,\;12\},
$$

with the critic initialized at $V(s)=0$ and a learning rate of $\alpha=0.1.$

| Episode | Return $G$ | Updated Value |
|---------:|-----------:|--------------:|
| 1 | 10 | 1.00 |
| 2 | 2 | 1.10 |
| 3 | 6 | 1.59 |
| 4 | 0 | 1.43 |
| 5 | 12 | 2.49 |

After only five episodes, the critic is still far from the true average return of six. However, each update moves the prediction in the appropriate direction. As more trajectories are observed, the critic continues refining its estimate until it stabilizes near the expected return. This gradual convergence illustrates why the critic should be viewed as a **running estimate** rather than an exact predictor of individual returns. lecture-04-critic-and-advantage.pdf

---

## A Neural Network Learns in Exactly the Same Way

The previous derivation treated the value estimate as though it were a single scalar. In practice, the critic is usually implemented as a neural network with millions of parameters.

Although the mathematics appears more complicated, the underlying idea is identical.

Instead of updating one scalar value, gradient descent updates the network parameters,

$$
\theta_v
\leftarrow
\theta_v
-
\alpha
\nabla_{\theta_v}
\mathcal{L}(\theta_v).
$$

Changing the parameters alters the predicted values for many states simultaneously because the network shares parameters across the entire state space. States with similar feature representations often experience similar changes in their value estimates.

Despite this increased complexity, the objective remains unchanged. Every gradient step nudges the network toward predicting the expected return associated with each state. No individual trajectory is memorized; rather, the critic gradually extracts the statistical regularities present across many episodes.

---

## The Critic Is an Incremental Estimator

The update rule

$$
V
\leftarrow
V
+
\alpha
(G-V)
$$

reveals an important conceptual insight.

The critic is not attempting to predict the future perfectly after every episode. Instead, it behaves as an **incremental estimator** that continuously refines its estimate as additional experience becomes available.

Every observed return contributes a small amount of information about the underlying return distribution. The critic integrates these observations over time, gradually improving its estimate of the expected return while filtering out the randomness present in individual trajectories.

This explains why the critic and the advantage can coexist without contradiction. At the moment the advantage is computed, the critic represents the accumulated knowledge obtained from many previous trajectories, whereas the return represents the outcome of one particular trajectory. Since these quantities describe different statistical objects, their difference remains meaningful throughout training.

> [!important]
> The critic is **not** updated by replacing its prediction with the latest return. Each observed return produces only a small gradient step toward itself. Over many episodes, these incremental updates cause the critic to converge to the **expected return**, while the difference between an individual return and this expectation continues to provide the learning signal used by the actor.


---

# 7. Why the Critic Is Never Perfect

The previous section described how the critic gradually learns the expected return by repeatedly adjusting its predictions toward observed returns. If enough trajectories are collected, it might appear that the critic should eventually become perfectly accurate, predicting the expected return exactly for every state.

In practice, however, this almost never happens.

Even after millions of training iterations, the critic remains an approximation. Its predictions continually improve, but they are rarely exact. This is not a failure of the learning algorithm; rather, it is an unavoidable consequence of the way reinforcement learning is formulated. Unlike supervised learning, where the target labels remain fixed throughout training, the target that the critic is attempting to estimate is itself continuously changing.

To understand why, we must examine three independent sources of error that affect every value function.

---

## Cold Start: Learning from Limited Experience

At the beginning of training, the critic has almost no information about the environment.

Its parameters are typically initialised randomly, so its predictions bear little resemblance to the true expected returns. The only information available comes from a small number of sampled trajectories, each of which represents a noisy realisation of the return distribution.

Suppose an agent visits a state for the very first time and observes a return of $G = 15.$

At this point, there is no reason to believe that 15 is the expected return from that state. It is simply one observation. The next visit may produce a return of 9, followed by 18, then 12. Only after many such observations does the critic begin to approximate the underlying expectation.

The early stages of training are therefore dominated by statistical uncertainty. The critic is attempting to estimate an expectation from a very small sample, and its predictions naturally exhibit high variance.

Fortunately, actor-critic methods do not require a perfect critic. They require only a baseline that is sufficiently accurate to reduce the variance of the policy gradient. Even an imperfect estimate often provides a substantial improvement over using the raw return directly.

---

## Function Approximation Error

A second source of imperfection arises because the critic is usually implemented as a neural network.

The value function,$V_\theta(s),$ is therefore only an approximation to the true value function,$V^\pi(s).$
Neural networks possess finite capacity. They cannot represent every possible mapping from states to values exactly, particularly when the state space is extremely large or continuous.

Furthermore, the network parameters are shared across many states. Updating the prediction for one state inevitably changes the predictions for neighbouring states as well. While this parameter sharing enables generalisation, it also introduces approximation error.

Consequently, even if the optimization algorithm converges perfectly, the learned value function may still differ slightly from the true expectation because of the representational limitations of the model itself.

---

## A Moving Target

Perhaps the most important reason the critic is never perfect is that the quantity it is trying to estimate is itself changing throughout training.

Recall that the value function is defined as

$$
V^\pi(s)
=
\mathbb{E}_\pi[G\mid s].
$$

Notice that the expectation depends explicitly on the policy $\pi.$ As the actor improves, the policy changes.

Since the policy determines which actions are selected and which trajectories are experienced, changing the policy also changes the distribution of returns.

Consequently,$V^\pi(s)$ is not a fixed function. It evolves continuously as the policy evolves.

This creates a moving target for the critic. By the time the critic has adapted to the value function associated with one policy, the actor has already updated the policy again, producing a new value function that must now be learned.

The critic is therefore engaged in a continual process of tracking rather than converging.

---

## Tracking Rather Than Memorising

An intuitive way to understand the critic is to compare it to a weather forecast.

A weather model does not attempt to memorize yesterday's temperature. Instead, it continually updates its predictions as new observations become available because the underlying system is itself changing.

The critic behaves similarly.

It does not memorize historical returns. Instead, it continually revises its estimate of the expected return under the **current** policy.

As the policy improves, the expected return from many states increases. The critic must therefore adjust its predictions accordingly.

For this reason, reinforcement learning practitioners often use a constant learning rate for the critic rather than a learning rate that decays to zero. A constant learning rate allows the value function to remain responsive to recent experience, enabling it to track the continually changing return distribution rather than converging to an outdated estimate. 

---

## Why an Imperfect Critic Is Still Useful

At first glance, the fact that the critic is always imperfect may seem undesirable. One might reasonably wonder how the actor can learn reliably from a baseline that is itself only an approximation.

The key observation is that the critic is not required to predict the return perfectly.

Its primary purpose is to provide a stable estimate of the expected return so that the actor can distinguish between actions that performed **better than expected** and those that performed **worse than expected**.

Suppose the true expected return from a state is 100, but the critic predicts 97.

An observed trajectory produces a return of 108.

The true advantage is

$$
108 - 100 = 8,
$$

while the estimated advantage becomes

$$
108 - 97 = 11.
$$

Although the magnitude differs slightly, the sign remains positive. The actor still receives the correct qualitative learning signal: the selected action performed better than expected and should therefore become more likely.

As the critic improves, these estimation errors gradually decrease, producing increasingly accurate advantages and lower-variance policy gradient estimates.

---

## The Critic Is Always One Step Behind

The interaction between the actor and critic can therefore be viewed as a feedback loop.

1. The actor improves the policy.

2. The improved policy changes the expected return.

3. The critic updates its estimate of the new expected return.

4. The improved critic produces more accurate advantages.

5. Those improved advantages enable the actor to make better policy updates.

6. The process then repeats.

Rather than converging independently, the actor and critic continually co-evolve throughout training. The critic is almost always slightly behind the actor because it is estimating a quantity that changes whenever the actor learns.

This behaviour is not a flaw of actor-critic methods. It is an inherent property of learning in a non-stationary environment where both the policy and its corresponding value function evolve together.

> [!important]
> The critic is not a perfect oracle that predicts the exact return of every trajectory. It is a continually evolving estimate of the **expected return under the current policy**. Because the policy itself changes throughout training, the critic is always tracking a moving target rather than converging to a fixed function. This is precisely why the advantage remains a meaningful learning signal even after extensive training.


---

# 8. Why Advantage Reduces Variance

Having established that the advantage measures the deviation of an observed return from its expected value, we can now address the fundamental motivation behind Actor-Critic methods.

If policy gradients can already be estimated using the return,

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta
\log\pi_\theta(a|s)
\,G
\right],
$$

why introduce a critic at all? Why not simply use the return as the learning signal?

The answer lies in one of the central challenges of Monte Carlo estimation: **variance**.

The return contains much more information than the actor actually needs. While part of the return reflects the quality of the chosen action, another large portion simply reflects the intrinsic quality of the state from which the action was taken. The actor should learn from the former but ignore the latter.

The critic serves precisely this purpose.

---

## The Return Contains Two Different Sources of Information

Consider an agent playing a game.

Suppose it reaches an extremely advantageous state from which almost every reasonable action eventually leads to victory.

Regardless of which action is selected, the observed return is likely to be very high.

For example,

| State | Action | Return |
|-------|--------|-------:|
| Winning | A | 98 |
| Winning | B | 102 |
| Winning | C | 100 |

Now consider another state that is already highly unfavourable. Even the best possible action produces only a modest return.

| State | Action | Return |
|-------|--------|-------:|
| Losing | A | 8 |
| Losing | B | 12 |
| Losing | C | 10 |

If the actor were trained directly on the return, it would conclude that every action taken in the first state is vastly superior to every action taken in the second state.

However, this conclusion is misleading.

The large returns are not necessarily the consequence of excellent decisions.

They arise primarily because the agent happened to begin from an exceptionally favourable state.

Similarly, the poor returns in the second state do not imply that every action was bad. They merely reflect the difficulty of the state itself.

The actor should therefore distinguish between

- **the quality of the state**, and
- **the quality of the chosen action within that state.**

The return alone cannot make this distinction.

---

## Removing the State Component

The critic estimates

$$
V(s),
$$

the expected return obtainable from the current state.

Subtracting this expectation from the observed return gives

$$
A(s,a)
=
G
-
V(s).
$$

Notice what this subtraction accomplishes.

The value function removes the portion of the return that is already explained by simply being in that state.

The remaining quantity reflects only whether the selected action performed better or worse than expected.

Suppose

$$
V(s)=100.
$$

Now consider three possible actions.

| Action | Return | Advantage |
|--------|-------:|----------:|
| A | 105 | +5 |
| B | 100 | 0 |
| C | 95 | -5 |

Although all three returns are large, only the first action exceeded expectations.

Similarly,

consider a poor state,

$$
V(s)=10.
$$

| Action | Return | Advantage |
|--------|-------:|----------:|
| A | 15 | +5 |
| B | 10 | 0 |
| C | 5 | -5 |

Even though the returns are much smaller, the advantages are identical.

This illustrates the key insight.

The actor is no longer comparing actions across different states.

Instead, it compares actions **relative to what is normally expected in the same state**.

---

## Centering the Learning Signal

From a statistical perspective,

subtracting the value function **centres** the learning signal.

Recall that

$$
V(s)
=
\mathbb E[G|s].
$$

Therefore,

$$
\mathbb E[A(s,a)|s]
=
0.
$$

Instead of learning from large absolute returns,

the actor now learns from deviations around zero.

Positive deviations indicate actions that performed better than expected.

Negative deviations indicate actions that performed worse than expected.

This centred signal is considerably easier to optimize because positive and negative updates naturally balance one another.

---

## Why Variance Is Reduced

To understand why variance decreases,

consider two different learning signals.

### Using the Return

Suppose the returns are

$$
98,\;
102,\;
100,\;
101,\;
99.
$$

The numerical values are large,

even though the actions differ only slightly in quality.

The policy gradient therefore receives large updates whose magnitudes are dominated by the value of the state itself.

---

### Using the Advantage

Suppose the critic predicts

$$
V(s)=100.
$$

The corresponding advantages become

$$
-2,\;
+2,\;
0,\;
+1,\;
-1.
$$

Notice what has changed.

The large common offset has disappeared. Only the relative differences between actions remain.

The learning signal is now centred near zero and exhibits much smaller variability.

Lower variance leads directly to more stable gradient estimates and smoother optimization.

---

## Why the Expected Gradient Does Not Change

At this point, one might wonder whether subtracting the value function changes the policy gradient itself. Remarkably, it does not. The policy gradient theorem states that

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\nabla_\theta
\log\pi(a|s)
\,G
\right].
$$

Now subtract an arbitrary baseline $b(s),$ that depends only on the state. 

The gradient becomes

$$
\mathbb E
\left[
\nabla_\theta
\log\pi(a|s)
(G-b(s))
\right].
$$

Expanding,

$$
=
\mathbb E
\left[
\nabla_\theta
\log\pi(a|s)
G
\right]
-
\mathbb E
\left[
\nabla_\theta
\log\pi(a|s)
b(s)
\right].
$$

Since $b(s)$ does not depend on the selected action, it can be moved outside the inner expectation. The remaining expectation is

$$
\mathbb E_a
\left[
\nabla_\theta
\log\pi(a|s)
\right].
$$

Using the identity

$$
\sum_a
\pi(a|s)
\nabla_\theta
\log\pi(a|s)
=
0,
$$

the second term disappears completely.

Consequently,

$$
\boxed{
\mathbb E
\left[
\nabla_\theta
\log\pi(a|s)
(G-b(s))
\right]
=
\mathbb E
\left[
\nabla_\theta
\log\pi(a|s)
G
\right].
}
$$

This remarkable result shows that **subtracting any state-dependent baseline leaves the expected policy gradient unchanged.**

The only thing that changes is the variance.

Among all possible baselines, the value function $V(s)$ is particularly attractive because it closely approximates the expected return, making it highly effective at reducing variance.

---

## The Role of the Critic

The discussion throughout this note now comes together.

The actor is responsible for improving the policy. The critic is responsible for estimating the expected return. The advantage combines these two quantities to determine whether a sampled action performed better or worse than expected.

Without the critic, the actor learns from noisy returns whose magnitude is strongly influenced by the intrinsic quality of the state. With the critic, the actor learns from deviations relative to the expected return, producing a centred learning signal with substantially lower variance.

The critic therefore does not make the policy gradient **correct**. The policy gradient was already correct. Instead, the critic makes the policy gradient **practical** by dramatically improving its statistical efficiency.

> [!important]
> The primary purpose of the critic is **not to increase the expected policy gradient but to reduce its variance**. By subtracting the expected return from the observed return, the advantage isolates the contribution of the chosen action while leaving the expected gradient unchanged. This single idea transforms the high-variance Monte Carlo policy gradient into the stable Actor-Critic algorithms used in modern Reinforcement Learning.

---

# 9. Connection with GRPO

Throughout this note, we have seen that the critic estimates the expected return from a state and that the advantage measures how much better or worse an observed trajectory performed relative to that expectation. This raises an interesting question.

> **Is learning a value function the only way to construct an advantage?**

The answer is no.

Recent reinforcement learning algorithms, particularly those developed for training Large Language Models, often eliminate the critic entirely. One such algorithm is **Group Relative Policy Optimization (GRPO)**.

Unlike Actor-Critic methods, GRPO does not train a neural network to estimate the value function. Instead, it estimates the baseline directly from a group of sampled trajectories.

Suppose a language model is asked a question and generates several candidate responses. Each response is evaluated by a reward model, producing the following rewards:

| Response | Reward |
|----------|-------:|
| 1 | 8 |
| 2 | 12 |
| 3 | 10 |
| 4 | 6 |

The average reward of the group is

$$
\bar{R}
=
\frac{8+12+10+6}{4}
=
9.
$$

Rather than subtracting a learned value function, GRPO subtracts this group average.

The resulting relative advantages become

| Response | Reward | Relative Advantage |
|----------|-------:|-------------------:|
| 1 | 8 | -1 |
| 2 | 12 | +3 |
| 3 | 10 | +1 |
| 4 | 6 | -3 |

Notice that the interpretation is identical to the Actor-Critic advantage.

Responses that perform better than the group average receive positive advantages, while those that perform worse receive negative advantages.

The only difference lies in **how the baseline is obtained**.

Actor-Critic methods estimate the baseline by learning the value function

$$
V(s)
=
\mathbb E[G|s].
$$

GRPO estimates the baseline directly from the sampled trajectories,

$$
\bar{R}
=
\frac1N
\sum_{i=1}^{N}
R_i.
$$

Although these approaches appear different, they are built upon the same statistical principle.

Both subtract an estimate of the expected return before computing the policy gradient.

In other words, both algorithms attempt to remove the component of the return that is common to all actions so that the policy learns only from relative differences in performance.

GRPO therefore demonstrates an important lesson.

The critic is **one possible mechanism** for constructing a baseline, but it is **not the only mechanism**. What truly matters is reducing the variance of the policy gradient by centering the learning signal around an estimate of the expected return.

---

# 10. Summary

This note began with a seemingly paradoxical question.

If the critic is trained to predict the return, why does the advantage not eventually become zero?

The answer lies in understanding the distinction between a **sample** and its **expectation**.

The return observed during an episode is one realisation of a random variable. Even when the agent begins from exactly the same state, different actions, stochastic transitions, and random rewards produce different trajectories and therefore different returns. The critic does not attempt to memorize these individual outcomes. Instead, it estimates their conditional expectation, known as the value function.

Because the value function represents the average return while each trajectory produces only one sample from the return distribution, the two quantities are rarely identical. Their difference is the advantage, which measures whether a particular action performed better or worse than expected.

This interpretation also explains why the critic is trained using mean squared error. Minimizing the squared prediction error naturally causes the critic to converge to the expected return rather than to any individual observation. During training, the critic continuously refines this estimate by taking small gradient steps toward newly observed returns, gradually incorporating information from many trajectories instead of memorizing single episodes.

The note further showed that the critic is never perfectly accurate. Limited experience, approximation error, and the continually changing policy ensure that the value function remains an approximation throughout training. Rather than predicting the exact return of every trajectory, the critic continually tracks the expected return associated with the current policy.

Finally, we examined the true purpose of the critic. Contrary to a common misconception, the critic is not introduced to make the policy gradient unbiased. The policy gradient is already unbiased when computed using the return directly. Instead, the critic provides a baseline that removes the component of the return attributable to the state itself. The resulting advantage becomes a centred learning signal whose expectation is zero but whose variance is substantially smaller than that of the raw return. This reduction in variance leads to faster, more stable policy optimization and forms the foundation of modern Actor-Critic algorithms.

The same statistical principle appears even in algorithms that do not explicitly learn a value function. Methods such as GRPO replace the learned baseline with a group average, demonstrating that the essential idea is not the critic itself but the use of an appropriate baseline to construct a low-variance learning signal.

> [!important]
> The critic does **not** learn individual returns—it learns their **expected value**. The advantage is therefore not a prediction error but a **deviation from expectation**. By centering the learning signal around this expectation, Actor-Critic methods dramatically reduce the variance of policy gradient estimates while preserving their correctness. This single idea explains why modern policy optimization algorithms are both statistically efficient and computationally practical.

---

# Key Takeaways

- The value function is the **expected return** conditioned on the current state.
- The return is a **random variable**, not a fixed quantity.
- Minimizing Mean Squared Error causes the critic to learn the conditional expectation of the return.
- The advantage compares a **sampled return** with its **expected value**, not with another sampled return.
- The average advantage for a state is zero, but individual advantages are generally non-zero.
- The critic is always tracking a moving target because the policy changes throughout training.
- The primary purpose of the critic is **variance reduction**, not bias reduction.
- GRPO and Actor-Critic methods implement the same statistical idea using different mechanisms for estimating the baseline.
