# schedule_sim.py
# Item 13: how large could the causal "tail" be, and how much could reversing the query-tile order
# recover? A greedy list-scheduling model of the block scheduler: programs are dispatched in launch
# order (program_id(0), the query tile, varies fastest; then the head) to whichever slot frees up
# first. A program's cost is the number of key tiles it visits plus a fixed overhead for loading Q
# and storing O. The model ignores that programs run faster when fewer share an SM, so it gives an
# upper bound on the tail effect. "Longest first" swaps the grid axes (heads vary fastest) and
# reverses the query tiles, so every head's longest tiles start before anyone's short ones.
#   python schedule_sim.py > schedule_sim.txt
import heapq

N, HEADS = 4096, 32


def makespan(costs, slots):
    free = [0.0] * slots
    heapq.heapify(free)
    for c in costs:
        heapq.heappush(free, heapq.heappop(free) + c)
    return max(free)


print(f"{'tiles Q x K':12s} {'slots':>5s} {'overhead':>8s} {'ideal':>9s} {'in order':>9s} {'reversed':>9s} "
      f"{'longest 1st':>11s} {'tail (in order)':>15s} {'reversal gain':>13s} {'longest-1st gain':>16s}")
for bq, bk, slots in [(64, 32, 224), (64, 64, 224), (64, 128, 112)]:
    for overhead in (0.0, 1.0, 2.0):
        def cost(i):  # key tiles visited by query tile i (both loops), in units of one 32-key tile
            q_start = i * bq
            unmasked = (q_start // bk) * bk
            masked_stop = min(q_start + bq, N)
            n_tiles = unmasked // bk + -(-(masked_stop - unmasked) // bk)
            return n_tiles * bk / 32 + overhead
        tiles = N // bq
        order = [cost(i) for h in range(HEADS) for i in range(tiles)]
        rev = [cost(tiles - 1 - i) for h in range(HEADS) for i in range(tiles)]
        lpt = [cost(tiles - 1 - i) for i in range(tiles) for h in range(HEADS)]
        ideal = sum(order) / slots
        a, b, c = makespan(order, slots), makespan(rev, slots), makespan(lpt, slots)
        print(f"{bq}x{bk:<9d} {slots:5d} {overhead:8.1f} {ideal:9.1f} {a:9.1f} {b:9.1f} {c:11.1f} "
              f"{100 * (a / ideal - 1):14.1f}% {100 * (1 - b / a):12.1f}% {100 * (1 - c / a):15.1f}%")
