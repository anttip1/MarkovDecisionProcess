from operator import itemgetter
import random
import copy

NX = 4
NY = 3
INVALID_STATE = (1, 1)
TERMINATING_STATE = (3, 2)
GAMMA = 0.95
Q_LAMBDA = 0.01


def print_grid(grid, policy, iteration, title):
    print("\n*** " + title + " ***")
    print("*** GRID STATE AFTER " + str(iteration) + " ITERATIONS: \n")
    for y in reversed(range(0, NY)):
        for x in range(0, NX):
            print(
                " |(" + str(x) + "," + str(y) + ")=" + ("%.2f" % grid[x + y * NX]) + ": '" + policy[x + y * NX] + "' |",
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
                    " | (" + str(x) + "," + str(y) + ")=" + ("%.2f" % Q[x + a + y * NX]) + ": '" + str(
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


def q_learning(q):

    # q-learning grid has the dimensions of NX*NY*4, every state has four slots that
    # contain the values of each of the four actions (north, east, south, west)

    next_q = copy.deepcopy(q)

    for i in range(NX * NY):
        x = i % NX
        y = int(i / NX)

        if not is_invalid_state(x, y):
            if not (x, y) == TERMINATING_STATE:

                north = i if is_invalid_state(x, y + 1) else i + NX
                east = i if is_invalid_state(x + 1, y) else i + 1
                south = i if is_invalid_state(x, y - 1) else i - NX
                west = i if is_invalid_state(x - 1, y) else i - 1

                # We need this because the size of q was NX*4*NY
                s = i * 4

                # North value Q(s, "north")
                north_action = q[s + 0]
                # East value Q(s, "east")
                east_action = q[s + 1]
                # South value Q(s, "south")
                south_action = q[s + 2]
                # West value Q(s, "west")
                west_action = q[s + 3]


                if (random.uniform(0, 1) >= 0.95):
                    # 5 % chance of taking the action randomly.
                    chosen_action = random.choice(
                        [(west_action, 3), (east_action, 1), (north_action, 0), (south_action, 2)])
                else:
                    # 95 % chance of taking the action with largest Q-value.
                    chosen_action = max([(west_action, 3), (east_action, 1), (north_action, 0), (south_action, 2)],
                                        key=itemgetter(0))

                if chosen_action[1] == 0:
                    # North
                    # The successor state s' = s1
                    s1 = north * 4
                    i1 = north

                elif chosen_action[1] == 1:
                    # East
                    # The successor state s' = s1
                    s1 = east * 4
                    i1 = east

                elif chosen_action[1] == 2:
                    # South
                    # The successor state s' = s1
                    s1 = south * 4
                    i1 = south

                elif chosen_action[1] == 3:
                    # West
                    # The successor state s' = s1
                    s1 = west * 4
                    i1 = west

                # Best action from the successor statet:
                # max ( Q(s', north), Q(s', east), Q(s', south), Q(s', west))
                s1_max_action = max([q[s1 + 0], q[s1 + 1], q[s1 + 2], q[s1 + 3]])

                # The q-learning update function:
                # Q(s, a) = (1-lambda)Q(s,a) + lambda(R(s,a,s') + gamma*max(Q(s',a)))
                next_q[s + chosen_action[1]] = (1 - Q_LAMBDA) * chosen_action[0] + Q_LAMBDA*(reward(i1) + GAMMA * s1_max_action)


            else:
                # This is for the terminal state
                s = i * 4
                s1_max_action = max([q[0 + 0], q[0 + 1], q[0 + 2], q[0 + 3]])
                next_q[s + 0] = (1 - Q_LAMBDA) * q[s + 0] + Q_LAMBDA*(reward(0) + GAMMA * s1_max_action)
                next_q[s + 1] = (1 - Q_LAMBDA) * q[s + 1] + Q_LAMBDA*(reward(0) + GAMMA * s1_max_action)
                next_q[s + 2] = (1 - Q_LAMBDA) * q[s + 2] + Q_LAMBDA*(reward(0) + GAMMA * s1_max_action)
                next_q[s + 3] = (1 - Q_LAMBDA) * q[s + 3] + Q_LAMBDA*(reward(0) + GAMMA * s1_max_action)

    return next_q


def extract_q_learning_policy(q):

    actions = ["north", "east", "south", "west"]
    grid = [0 for i in range(NX*NY)]
    policy = ["" for i in range(NX*NY)]

    for i in range(NX*NY):
        x = i % NX
        y = int(i / NX)

        if not is_invalid_state(x, y):

            s = i*4
            best_action = max([(q[s+0], 0), (q[s+1], 1), (q[s+2], 2), (q[s+3], 3)], key=itemgetter(0))
            grid[i] = best_action[0]
            if not (x, y) == TERMINATING_STATE:
                policy[i] = actions[best_action[1]]
            else:
                policy[i] = "To start"

    return grid, policy


def main():
    grid = [0 for i in range(NX * NY)]
    policy = ["" for i in range(NX * NY)]
    convergence = False
    iteration = 0

    while not convergence:
        iteration += 1
        grid = value_iteration(grid, policy)

        if iteration >= 1000:
            convergence = True

    print_grid(grid, policy, iteration, "VALUE ITERATION")

    q = [0 for i in range(NX * NY * 4)]

    convergence = False
    iteration = 0

    while not convergence:
        iteration += 1
        q = q_learning(q)

        #print_q_learning(q_grid, iteration)

        if iteration >= 10000:
            convergence = True

    q_grid, q_policy = extract_q_learning_policy(q)
    print_grid(q_grid, q_policy, iteration, "Q-LEARNING")


if __name__ == "__main__":
    main()
