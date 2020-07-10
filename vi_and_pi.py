### MDP Value Iteration and Policy Iteration
### Acknowledgement: start-up codes were adapted with permission from Prof. Emma Brunskill of Stanford University

# Vikas Virani (s3715555)

import numpy as np
import gym
import time
import rmit_rl_env

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)
	#initialise value_function_old to keep track of old values of value function
	value_function_old = value_function.copy()

	#Perform 100 iterations for to calculate value function for all states
	for iter in range(100):
		delta = 0
		for s in range(nS):
			value = 0
			#initialise actions with predefined policy of each state
			actions = [policy[s]]
			for a in actions:
				#iterate through each probable outcome state of an action
				for next_state in P[s][a]:
					probability = next_state[0]
					reward = next_state[2]
					next_reward = gamma * value_function[next_state[1]]
					value += probability * (reward + next_reward)
			diff = abs(value_function_old[s] - value)
			delta = max(delta,diff)
			#update value_function with value of an action
			value_function[s] = value
		if delta <= tol:
			#Conerge when delta is less than tolerance
			break
		#Otherwise replace old value_function with new & loop again
		value_function_old = value_function

	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

	#initialise new_policy to be returned
	new_policy = policy.copy()

	#initialise changed boolean variable to keep track of policy is changed or not
	changed = False

	for s in range(nS):
		old_action = policy[s]
		max_value = -1
		max_action = -1
		for a in range(nA):
			value = 0
			#iterate through each probable outcome state of an action
			for next_state in P[s][a]:
				probability = next_state[0]
				reward = next_state[2]
				next_reward = gamma * value_from_policy[next_state[1]]
				value += probability * (reward + next_reward)
			if value > max_value:
				max_value = value
				max_action = a
		if max_action != old_action:
			#if policy(action) is changed, set variable to True
			changed = True
		#update new_policy with max_action of a state
		new_policy[s] = max_action

	#return new_policy & boolean variable
	return new_policy, changed


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	#initialise policy
	policy = np.zeros(nS, dtype=int)

	#initialise changed boolean variable to keep track of policy changes
	changed = True
	iterations = 0

	#iterate through "policy_evaluation" & "policy_improvement" functions till
	#the  policy stops being changed (i.e. Until it Converges)
	while changed:
		#call "policy_evaluation" with predefined policy to get value_function
		value_function = policy_evaluation(P, nS, nA, policy)
		iterations += 1
		#call "policy_improvement" with obtained value_function & get updated policy
		policy, changed = policy_improvement(P, nS, nA, value_function, policy)

	print("\nValue Function\n"+str(value_function))
	print("\niterations\n"+str(iterations))
	print("\nPolicy\n"+str(policy))

	#return value_function & policy
	return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	#initialise value_function_old to keep track of old values of value function
	value_function_old = value_function.copy()
	#initialise policy
	policy = np.zeros(nS, dtype=int)

	#Perform 100 iterations for to calculate value function for all states
	for iter in range(100):
		delta = 0
		for s in range(nS):
			max_value = -1
			for a in range(nA):
				value = 0
				#iterate through each probable outcome state of an action
				for next_state in P[s][a]:
					probability = next_state[0]
					reward = next_state[2]
					if value_function[next_state[1]] == 0:
						next_reward = gamma * value_function_old[next_state[1]]
					else:
						next_reward = gamma * value_function[next_state[1]]
					value += probability * (reward + next_reward)
				if value > max_value:
					max_value = value
					policy[s] = a
			diff = abs(value_function_old[s] - max_value)
			delta = max(delta,diff)
			#update value_function with max value of each action
			value_function[s] = max_value
			#update policy with action of that max value

		if delta <= tol:
			#Conerge when delta is less than tolerance
			break
		#Otherwise replace old value_function with new & loop again
		value_function_old = value_function

	print("\nValue Function\n"+str(value_function))

	print("\nPolicy\n"+str(policy))

	return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")

	#print("\nDictionary\n")
	#for s in range(env.nS):
	#	for a in range(env.nA):
	#		print("State :"+ str(s+1) +" Action :"+ str(a+1)+" : "+ str(env.P[s][a]))

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)
