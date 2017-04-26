from operator import itemgetter

NX = 4
NY = 3
INVALID_STATE = (1, 1)
TERMINATING_STATE = (3, 2)
DISCOUNT = 0.95


def print_grid(grid, policy, iteration):
    print("\n*** GRID STATE AFTER " + str(iteration) + " ITERATIONS: \n")
    for y in reversed(range(0, NY)):
        for x in range(0, NX):
            print(
                " |(" + str(x) + "," + str(y) + ")=" + ("%.2f" % grid[x + y * NX]) + ": '" + policy[x + y * NX] + "' |",
                end="")
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
                west_action = 0.8 * (reward(west) + DISCOUNT * grid[west]) + (
                    0.1 * (reward(south) + DISCOUNT * grid[south]) + 0.1 * (reward(north) + DISCOUNT * grid[north]))

                east_action = 0.8 * (reward(east) + DISCOUNT * grid[east]) + (
                    0.1 * (reward(south) + DISCOUNT * grid[south]) + 0.1 * (reward(north) + DISCOUNT * grid[north]))

                north_action = 0.8 * (reward(north) + DISCOUNT * grid[north]) + (
                    0.1 * (reward(east) + DISCOUNT * grid[east]) + 0.1 * (reward(west) + DISCOUNT * grid[west]))

                south_action = 0.8 * (reward(south) + DISCOUNT * grid[south]) + (
                    0.1 * (reward(east) + DISCOUNT * grid[east]) + 0.1 * (reward(west) + DISCOUNT * grid[west]))

                best_action = max(
                    [(west_action, "west"), (east_action, "east"), (north_action, "north"), (south_action, "south")],
                    key=itemgetter(0))

            else:
                best_action = (1.0 * (reward(0) + DISCOUNT * grid[0]), "To start")

            next_grid[i] = best_action[0]
            policy[i] = best_action[1]

    return next_grid


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

    print_grid(grid, policy, iteration)


if __name__ == "__main__":
    main()
