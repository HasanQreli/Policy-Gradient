import math
import random
import numpy as np
from policy import StochasticPolicy

from gridworld import GridWorld
from gridworld import OneDimensionalGridWorld
from policy_gradient import PolicyGradient
from logistic_regression_policy import LogisticRegressionPolicy

class PolicyGradient:
    def __init__(self, mdp, policy, alpha) -> None:
        super().__init__()
        self.alpha = alpha  # Learning rate (gradient update step-size)
        self.mdp = mdp
        self.policy = policy

    """ Generate and store an entire episode trajectory to use to update the policy """

    def execute(self, episodes=100):
        for _ in range(episodes):
            actions = []
            states = []
            rewards = []

            state = self.mdp.get_initial_state()
            episode_reward = 0
            while not self.mdp.is_terminal(state):
                action = self.policy.select_action(state)
                next_state, reward = self.mdp.execute(state, action)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            deltas = self.calculate_deltas(rewards)
            self.policy.update(states=states, actions=actions, deltas=deltas)

    def calculate_deltas(self, rewards):
        """
        Generate a list of the discounted future rewards at each step of an episode
        Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
        We can use that pattern to populate the discounted_rewards array.
        """
        T = len(rewards)
        discounted_future_rewards = [0 for _ in range(T)]
        # The final discounted reward is the reward you get at that step
        discounted_future_rewards[T - 1] = rewards[T - 1]
        for t in reversed(range(0, T - 1)):
            discounted_future_rewards[t] = (
                rewards[t]
                + discounted_future_rewards[t + 1] * self.mdp.get_discount_factor()
            )
        deltas = []
        for t in range(len(discounted_future_rewards)):
            deltas += [
                self.alpha
                * (self.mdp.get_discount_factor() ** t)
                * discounted_future_rewards[t]
            ]
        return deltas





""" A two-action policy implemented using logistic regression from first principles """


class LogisticRegressionPolicy(StochasticPolicy):

    """Create a new policy, with given parameters theta (randomly initialised if theta is None)"""

    def __init__(self, actions, num_params, theta=None):
        assert len(actions) == 4

        self.actions = actions

        if theta is None:
            theta = np.zeros([num_params,2])
        self.theta = theta

    """ Select one of the two actions using the logistic function for the given state """

    def select_action(self, state):
        # Get the probability of selecting the first action
        probability_left = self.get_probability(state, self.actions[0])
        probability_right = self.get_probability(state, self.actions[1])
        probability_up = self.get_probability(state, self.actions[2])

        # With a probability of 'probability' take the first action
        randprob = random.random()
        if randprob < probability_left:
            return self.actions[0]
        elif randprob < probability_right+probability_left:
            return self.actions[1]
        elif randprob < probability_up+probability_right+probability_left:
            return self.actions[2]
        else:
            return self.actions[3]

    def update(self, states, actions, deltas):
        for t in range(len(states)):
            print(states[t])
            gradient_log_pi = self.gradient_log_pi(states[t], actions[t])
            # Update each parameter
            for i in range(len(self.theta)):
                self.theta[i] += deltas[t] * gradient_log_pi[i]
    """ Get the probability of applying an action in a state """

    def get_probability(self, state, action):

        # Pass y through the logistic regression function to convert it to a probability
        probability = self.softmax(np.dot(self.theta, state))
        if action == self.actions[0]:
            return probability[0]
        elif action == self.actions[1]:
            return probability[1]
        elif action == self.actions[2]:
            return probability[2]
        return probability[3]

    """ Computes the gradient of the log of the policy (pi),
    which is needed to get the gradient of the objective (J).
    Because the policy is a logistic regression, using the policy parameters (theta).
        pi(actions[0] | state)= 1 / (1 + e^(-theta * state))
        pi(actions[1] | state) = 1 / (1 + e^(theta * state))
    When we apply a logarithmic transformation and take the gradient we end up with:
        grad_log_pi(left | state) = state - state * pi(left|state)
        grad_log_pi(right | state) = - state * pi(0|state)
    """

    def gradient_log_pi(self, state, action):
        scores = np.dot(self.theta, state)
        probs = -self.softmax(scores)
        probs[self.actions.index(action)] += 1
        probs = np.dot(np.array([probs]).T, np.array([state]))
        return probs

    """ Standard logistic function """

    def softmax(self, temp):
        return np.exp(temp) / np.sum(np.exp(temp))

    @staticmethod
    def logistic_function(y):
        return 1 / (1 + math.exp(-y))

    """ Compute the dot product between two vectors """

    @staticmethod
    def dot_product(vec1, vec2):
        return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])








gridworld = GridWorld(
    height=3, width=3, initial_state=(0, 0), goals=[((2, 1), 1), ((2, 0), -1)]
)
gridworld_image = gridworld.visualise()



policy = LogisticRegressionPolicy(
    actions=[GridWorld.LEFT, GridWorld.RIGHT, GridWorld.UP, GridWorld.DOWN],
    num_params= 4,
)
policy_gradient = PolicyGradient(gridworld, policy, alpha=0.1)
policy_gradient.execute(episodes=1000)
policy_image = gridworld.visualise_stochastic_policy(policy)
