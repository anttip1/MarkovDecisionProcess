from operator import itemgetter
import random
import copy
import math
import matplotlib.pyplot as plt

NX = 4
NY = 3
INVALID_STATE = (1, 1)
TERMINATING_STATE = (3, 2)
GAMMA = 0.95
Q_LAMBDA = 0.01


def print_grid(grid, policy, iteration, title, precision=2):
    print("\n*** " + title + " ***")
    print("*** GRID STATE AFTER " + str(iteration) + " ITERATIONS: \n")
    for y in reversed(range(0, NY)):
        for x in range(0, NX):
            print(
                " | (" + str(x) + "," + str(y) + ")=" + str(round(grid[x + y * NX], precision)) + ": '" + policy[
                    x + y * NX] + "' |",
                end="")
        print("\n")
    print("*** END OF " + title)


def print_q_learning(Q, iteration):
    print("\n*** GRID STATE AFTER " + str(iteration) + " ITERATIONS: \n")
    actions = ["north", "east", "south", "west"]
    for y in reversed(range(0, NY)):
        for a in range(0, 4):
            for x in range(0, NX):
                print(
                    " | (" + str(x) + "," + str(y) + ")=" + ("%.2f" % Q[a + x * 4 + y * NX * 4]) + ": '" + str(
                        actions[a]) + "' |", end="")
            print("\n")
        print("\n")
    print("***")


def is_invalid_state(x, y):
    if x < 0 or x >= NX or y < 0 or y >= NY or (x, y) == INVALID_STATE:
        return True
    else:
        return False


def reward(index):
    x = index % NX
    y = int(index / NX)
    if (x, y) == (3, 2):
        return 1
    elif (x, y) == (3, 1):
        return -1
    else:
        return 0


def value_iteration(grid, policy):
    next_grid = [0 for i in range(NX * NY)]

    for i in range(NX * NY):
        x = i % NX
        y = int(i / NX)
        if not is_invalid_state(x, y):

            if not (x, y) == TERMINATING_STATE:
                east = i if is_invalid_state(x + 1, y) else i + 1
                west = i if is_invalid_state(x - 1, y) else i - 1
                north = i if is_invalid_state(x, y + 1) else i + NX
                south = i if is_invalid_state(x, y - 1) else i - NX
                west_action = 0.8 * (reward(west) + GAMMA * grid[west]) + 0.1 * (
                    reward(south) + GAMMA * grid[south]) + 0.1 * (reward(north) + GAMMA * grid[north])
                east_action = 0.8 * (reward(east) + GAMMA * grid[east]) + 0.1 * (
                    reward(south) + GAMMA * grid[south]) + 0.1 * (reward(north) + GAMMA * grid[north])
                north_action = 0.8 * (reward(north) + GAMMA * grid[north]) + 0.1 * (
                    reward(east) + GAMMA * grid[east]) + 0.1 * (reward(west) + GAMMA * grid[west])
                south_action = 0.8 * (reward(south) + GAMMA * grid[south]) + 0.1 * (
                    reward(east) + GAMMA * grid[east]) + 0.1 * (reward(west) + GAMMA * grid[west])
                best_action = max(
                    [(west_action, "west"), (east_action, "east"), (north_action, "north"), (south_action, "south")],
                    key=itemgetter(0))
            else:
                best_action = (1.0 * (reward(0) + GAMMA * grid[0]), "To start")

            next_grid[i] = best_action[0]
            policy[i] = best_action[1]

    return next_grid


def q_learning(grid):
    # q-learning grid has the dimensions of NX*NY*4, every state has four slots that
    # contain the values of each of the four actions (north, east, south, west)

    #q_grid, q_policy = extract_q_learning_policy(q)
    start_values = []
    q = [0 for i in range(NX * NY * 4)]
    #q_sum_of_rewards = [0 for i in range(NX * NY * 4)]
    #q_tries = [0 for i in range(NX * NY * 4)]


    iteration = 0
    i = 0

    while not q_learning_has_converged(grid, q) and iteration < 200000:
        iteration += 1
        x = i % NX
        y = int(i / NX)

        if not (x, y) == TERMINATING_STATE:

            north = i if is_invalid_state(x, y + 1) else i + NX
            east = i if is_invalid_state(x + 1, y) else i + 1
            south = i if is_invalid_state(x, y - 1) else i - NX
            west = i if is_invalid_state(x - 1, y) else i - 1

            # We need this because the size of q was NX*4*NY
            s = i * 4

            action_value_list = [(q[s + 0], 0), (q[s + 1], 1), (q[s + 2], 2), (q[s + 3], 3)]

            if random.uniform(0, 1) >= 0.7:
                # 5 % chance of taking the action randomly.
                chosen_action = random.choice(action_value_list)[1]
            else:
                # 95 % chance of taking the action with largest Q-value.
                random.shuffle(action_value_list)
                chosen_action = max(action_value_list, key=itemgetter(0))[1]

            '''
            if q_tries[s + 0] == 0:
                chosen_action = 0
            elif q_tries[s + 1] == 0:
                chosen_action = 1
            elif q_tries[s + 2] == 0:
                chosen_action = 2
            elif q_tries[s + 3] == 0:
                chosen_action = 3
            else:
                north_action_value = q_sum_of_rewards[s + 0] / q_tries[s + 0] + math.sqrt(
                    (math.log(2 * sum(q_tries[s:s + 3]))) / q_tries[s + 0])
                east_action_value = q_sum_of_rewards[s + 1] / q_tries[s + 1] + math.sqrt(
                    (math.log(2 * sum(q_tries[s:s + 3]))) / q_tries[s + 1])
                south_action_value = q_sum_of_rewards[s + 2] / q_tries[s + 2] + math.sqrt(
                    (math.log(2 * sum(q_tries[s:s + 3]))) / q_tries[s + 2])
                west_action_value = q_sum_of_rewards[s + 3] / q_tries[s + 3] + math.sqrt(
                    (math.log(2 * sum(q_tries[s:s + 3]))) / q_tries[s + 3])
                chosen_action = max([(north_action_value, 0), (east_action_value, 1), (south_action_value, 2), (west_action_value, 3)], key=itemgetter(0))[1]
            '''
            chance = random.random()

            if chance < 0.8:
                actual_action = chosen_action
            elif chance >= 0.8 < 0.9:
                actual_action = (chosen_action - 1 + 4) % 4
            else:
                actual_action = (chosen_action + 1) % 4

            if actual_action == 0:
                # North
                # The successor state s' = s1
                s1 = north * 4
                i1 = north

            elif actual_action == 1:
                # East
                # The successor state s' = s1
                s1 = east * 4
                i1 = east

            elif actual_action == 2:
                # South
                # The successor state s' = s1
                s1 = south * 4
                i1 = south

            else:
                # West
                # The successor state s' = s1
                s1 = west * 4
                i1 = west

            # Best action from the successor states:
            # max ( Q(s', north), Q(s', east), Q(s', south), Q(s', west))
            successor_state_values = [q[s1 + 0], q[s1 + 1], q[s1 + 2], q[s1 + 3]]
            random.shuffle(successor_state_values)
            s1_max_action = max(successor_state_values)

            # The q-learning update function:
            # Q(s, a) = (1-lambda)Q(s,a) + lambda(R(s,a,s') + gamma*max(Q(s',a)))
            q[s + chosen_action] = (1 - Q_LAMBDA) * q[s + chosen_action] + Q_LAMBDA * (
                reward(i1) + GAMMA * s1_max_action)

            temp = (1 - Q_LAMBDA) * q[s + chosen_action] + Q_LAMBDA * (reward(i1) + GAMMA * s1_max_action)
            q[s + chosen_action] = temp
            if s == 0:
                start_values.append(temp);

            #q_sum_of_rewards[s + chosen_action] += temp
            #q_tries[s + chosen_action] += 1

            #print("Chosen action: " + str(chosen_action) + ", actual action: " + str(actual_action))
            #print("Current: " + str(i) + ", next: " + str(i1))

            i = i1

        else:
            # This is for the terminal state
            s = i * 4
            i1 = 0

            s1_max_action = max([q[0 + 0], q[0 + 1], q[0 + 2], q[0 + 3]])
            chosen_action = random.choice([0, 1, 2, 3])

            temp = (1 - Q_LAMBDA) * q[s + chosen_action] + Q_LAMBDA * (reward(i1) + GAMMA * s1_max_action)
            q[s + chosen_action] = temp
            #q_sum_of_rewards[s + chosen_action] += temp
            #q_tries[s + chosen_action] += 1

            #q[s + chosen_action] = (1 - Q_LAMBDA) * q[s + chosen_action] + Q_LAMBDA * (reward(0) + GAMMA * s1_max_action)
            #q[s + 1] = (1 - Q_LAMBDA) * q[s + 1] + Q_LAMBDA * (reward(0) + GAMMA * s1_max_action)
            #q[s + 2] = (1 - Q_LAMBDA) * q[s + 2] + Q_LAMBDA * (reward(0) + GAMMA * s1_max_action)
            #q[s + 3] = (1 - Q_LAMBDA) * q[s + 3] + Q_LAMBDA * (reward(0) + GAMMA * s1_max_action)
            #print("Current: " + str(i) + ", next: 0")

            i = i1

    q_grid, q_policy = extract_q_learning_policy(q)
    print_grid(q_grid, q_policy, iteration, "Q-LEARNING")
    #print_q_learning(q_tries, iteration)

    start_x_values = [i for i in range(0, len(start_values))]

    truncated_start_values = start_values[::100]
    truncated_start_x_values = start_x_values[::100]
    value_iteration_values = [grid[0] for i in range(0, len(truncated_start_x_values))]
    #value_iteration_upper_bound = [grid[0]+0.1 for i in range(0, len(truncated_start_x_values))]
    #value_iteration_lower_bound = [grid[0]-0.1 for i in range(0, len(truncated_start_x_values))]


    plt.plot(truncated_start_x_values, truncated_start_values)
    plt.plot(truncated_start_x_values, value_iteration_values)
    #plt.plot(truncated_start_x_values, value_iteration_upper_bound)
    #plt.plot(truncated_start_x_values, value_iteration_lower_bound)

    plt.legend(['Q-value', 'Value iteration value'])
    plt.title('Start state Q-value (Every 100th datapoint)')
    #plt.ylabel('Start state Q-value (Every 100th value)')
    plt.show()




def extract_q_learning_policy(q):
    actions = ["north", "east", "south", "west"]
    grid = [0 for i in range(NX * NY)]
    policy = ["" for i in range(NX * NY)]

    for i in range(NX * NY):
        x = i % NX
        y = int(i / NX)

        if not is_invalid_state(x, y):

            s = i * 4
            best_action = max([(q[s + 0], 0), (q[s + 1], 1), (q[s + 2], 2), (q[s + 3], 3)], key=itemgetter(0))
            grid[i] = best_action[0]
            if not (x, y) == TERMINATING_STATE:
                policy[i] = actions[best_action[1]]
            else:
                policy[i] = "To start"

    return grid, policy


def q_learning_has_converged(grid, q):
    for i in range(NX * NY):
        if abs(grid[i] - q[i*4]) > 0.1:
            return False
    return True


def has_converged(grid, other_grid, precision):
    for i in range(NX * NY):
        if round(grid[i], precision) != round(other_grid[i], precision):
            return False
    return True


def policy_has_converged(policy, other_policy):
    for i in range(NX * NY):
        if policy[i] != other_policy[i]:
            return False
    return True


def main():
    grid = [0 for i in range(NX * NY)]
    next_grid = [0 for i in range(NX * NY)]
    policy = ["" for i in range(NX * NY)]
    convergence = False
    iteration = 0

    while not convergence:
        iteration += 1
        next_grid = value_iteration(grid, policy)

        if has_converged(grid, next_grid, 8):
            convergence = True
        else:
            grid = copy.deepcopy(next_grid)

    print_grid(grid, policy, iteration, "VALUE ITERATION", 3)


    q_learning(grid)

    # print_q_learning(q, iteration)


if __name__ == "__main__":
    main()
