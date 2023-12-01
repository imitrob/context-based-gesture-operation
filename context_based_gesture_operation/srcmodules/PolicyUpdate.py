import numpy as np
from copy import deepcopy
import itertools

class PolicyUpdate():
    def __init__(self):
        pass

    @staticmethod
    def is_policy_in_history(policy_history, policy):
        array_equal = np.array([np.array_equal(policy['CM_est'], policy_['CM_est']) for policy_ in policy_history])
        for n,is_equal in enumerate(array_equal):
            if is_equal:
                return True
        return False

class PolicyUpdateQLearning(PolicyUpdate):
    @staticmethod
    def do(policy_history, reward, actions, observation, alpha, gamma):
        policy = policy_history[-1]
        gestures = observation['gesture_vec']
        g = np.argmax(gestures)
        a = np.argmax(actions)

        old_value = policy['CM_est'][g, a]
        next_max = np.max(policy['CM_est'][g])
        policy['CM_est'][g,a] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        print(f"=== {g} {a}, {next_max}: {old_value}::: {policy['CM_est'][g,a]}")
        policy['r'] = reward
        return policy

class PolicyUpdateByMutation(PolicyUpdate):
    @staticmethod
    def do(policy_history, reward, out=False, mode='compare_last_reward__no_memory'):
        return getattr(PolicyUpdateByMutation, mode)(policy_history, reward, out)

    @staticmethod
    def mutation(policy, n_actions):
        r1,r2 = np.random.choice(n_actions, 2, replace=False)
        policy['CM_est'][[r1,r2]] = policy['CM_est'][[r2,r1]]
        return policy

    @staticmethod
    def compare_last_reward__no_memory__random(policy_history, reward, out):
        # Compute reward difference
        if len(policy_history) > 1:
            reward_dif = reward - policy_history[-1]['r']
        else:
            reward_dif = 0.00000001

        policy = deepcopy(policy_history[-1]) # new policy
        policy['r'] = reward

        n_actions = len(policy_history[0]['CM_est'])
        if out: print(f"r: {reward}, diff: {reward_dif}")
        if reward_dif >= 0:
            policy = PolicyUpdateByMutation.mutation(policy, n_actions)
            policy['i'] = 'forward'
        else: # Rewrites the policy from history, it has its own reward tag saved
            policy = deepcopy(policy_history[-2])
            policy['i'] = 'revert'

        return policy


    @staticmethod
    def compare_last_reward__with_memory(policy_history, reward, out):
        # Compute reward difference
        if len(policy_history) > 1:
            reward_dif = reward - policy_history[-1]['r']
        else:
            reward_dif = 0.00000001

        policy = deepcopy(policy_history[-1]) # new policy
        policy['r'] = reward

        n_actions = len(policy_history[0]['CM_est'])
        if out: print(f"r: {reward}, diff: {reward_dif}")
        if reward_dif >= 0:
            policy = PolicyUpdateByMutation.mutation(policy, n_actions)
            while PolicyUpdateByMutation.is_policy_in_history(policy_history, policy):
                policy = PolicyUpdateByMutation.mutation(policy, n_actions)

            policy['i'] = 'forward'
        else: # Rewrites the policy from history, it has its own reward tag saved
            policy = deepcopy(policy_history[-2])
            policy['i'] = 'revert'

        return policy

    @staticmethod
    def choose_new_permutation_based_on_list(n_permutations, achieved_list):
        try:
            p = np.random.choice(list(set([x for x in range(0, n_permutations)]) - set(achieved_list)))
        except ValueError:
            return None
        return p

    @staticmethod
    def get_achieved_list(policy_history):
        list_permutations = []
        for policy in policy_history:
            list_permutations.append(policy['permutation'])
        return list_permutations

    @staticmethod
    def get_policy_from_permutation(permutation, n_actions):
        permutation_indexs = list(itertools.permutations(list(range(0,n_actions))))[permutation]

        policy_CM = np.zeros([n_actions, n_actions])
        for i in range(n_actions):
            policy_CM[i][permutation_indexs[i]] = 1
        policy = {'CM_est': policy_CM, 'permutation': permutation}
        return policy

    @staticmethod
    def compare_last_reward__by_selection(policy_history, reward, out):
        n_actions = len(policy_history[0]['CM_est'])
        n_permutations = len(list(itertools.permutations(list(range(0,n_actions)))))

        # Compute reward difference
        if len(policy_history) > 1:
            reward_dif = reward - policy_history[-1]['r']
        else:
            reward_dif = 0.00000001

        policy = deepcopy(policy_history[-1]) # new policy
        policy['r'] = reward


        if out: print(f"r: {reward}, diff: {reward_dif}")
        if reward_dif >= 0:
            achieved_list = PolicyUpdateByMutation.get_achieved_list(policy_history)
            new_permutation = PolicyUpdateByMutation.choose_new_permutation_based_on_list(n_permutations, achieved_list)
            if new_permutation is None:
                return False
            policy = PolicyUpdateByMutation.get_policy_from_permutation(new_permutation, n_actions)

            policy['i'] = 'forward'
        else: # Rewrites the policy from history, it has its own reward tag saved
            policy = deepcopy(policy_history[-2])
            policy['i'] = 'revert'
        policy['r'] = reward # no effect

        return policy


if __name__ == '__main__':
    if True:
        policy_history = [{'CM_est': np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]]),       'i': 'init',      'r': False,      'action': {'target_object': 'drawer', 'target_action': 'put'}},
         {'CM_est': np.array([[0, 0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]]),      'i': 'forward',      'r': True,      'action': {'target_object': 'drawer', 'target_action': 'put'}},
         {'CM_est': np.array([[0, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]]),      'i': 'forward',      'r': True,      'action': {'target_object': 'drawer', 'target_action': 'put'}}]

    PolicyUpdateByMutation.compare_last_reward__no_memory(policy_history, 0, out=True)
    PolicyUpdateByMutation.compare_last_reward__no_memory(policy_history, 2, out=True)
    PolicyUpdateByMutation.compare_last_reward__no_memory(policy_history, 2, out=True)
    PolicyUpdateByMutation.compare_last_reward__no_memory(policy_history, 2, out=True)
    PolicyUpdateByMutation.compare_last_reward__no_memory(policy_history, 2, out=True)

    def mark_policy_history(policy_history, n_actions):
        coverage_total = len(list(itertools.permutations(list(range(0,n_actions)))))
        coverags_sum = 0
        for i in range(coverage_total):
            policy = PolicyUpdateByMutation.get_policy_from_permutation(i, n_actions)
            if PolicyUpdate.is_policy_in_history(policy_history, policy):
                coverags_sum +=1

        return f"{(coverags_sum/coverage_total)*100}%"

    n_actions = 6
    ''' policy CM est array with size 6 x 6, that has only ones and zeros and only single one can be used for each row
        There is 720 permutations.
        When there is 720 mutations of policy CM array, there is desire to cover all permutations.
        Let's check each method and coverage of draws.
    '''
    policy_history = [{'CM_est': np.diag(np.ones(6)), 'i': 'init', 'r': False, 'permutation': 0}]
    for i in range(720):
        PolicyUpdateByMutation.compare_last_reward__no_memory__random(policy_history, 2, out=False)
    mark_policy_history(policy_history, n_actions)
    '''
    >>> 57.9%
    '''

    policy_history = [{'CM_est': np.diag(np.ones(6)), 'i': 'init', 'r': False, 'permutation': 0}]
    for i in range(720):
        PolicyUpdateByMutation.compare_last_reward__with_memory(policy_history, 2, out=False)
    mark_policy_history(policy_history, n_actions)
    '''
    >>> 64.2%
    '''

    policy_history = [{'CM_est': np.diag(np.ones(6)), 'i': 'init', 'r': False, 'permutation': 0}]
    for i in range(719):
        PolicyUpdateByMutation.compare_last_reward__by_selection(policy_history, 2, out=False)
    mark_policy_history(policy_history, n_actions)
    '''
    >>> 100.0%
    '''
