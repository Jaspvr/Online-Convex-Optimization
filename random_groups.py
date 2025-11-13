import math
import random

def randomGroups(n):
    # min 3 elements in a group
    # groups span 50% to 75% of the set
    if n < 3:
        return []

    # Choose how many indices to include (k)
    k_min = max(3, math.ceil(0.5 * n))
    k_max = max(k_min, math.floor(0.75 * n))
    k_max = min(k_max, n)          # just to be safe
    k_min = min(k_min, k_max)

    k = random.randint(k_min, k_max)

    #  Choose which indices to include and shuffle them 
    indices = list(range(n))
    random.shuffle(indices)
    chosen = indices[:k]           # the ones that will be in some group

    # Decide how many groups (each has size >= 3) 
    max_groups = max(1, k // 3)    # at least 3 per group
    num_groups = random.randint(1, max_groups)

    # Create random group sizes, each >= 3, summing to k 
    # Start with 3 in each group, then distribute the remaining 'extra' randomly
    sizes = [3] * num_groups
    extra = k - 3 * num_groups

    for _ in range(extra):
        g = random.randrange(num_groups)
        sizes[g] += 1

    # Slice the shuffled indices according to those sizes
    groups = []
    start = 0
    for size in sizes:
        end = start + size
        groups.append(chosen[start:end])
        start = end

    return groups

def main():
    print(randomGroups(20))
    print(randomGroups(20))
    print(randomGroups(20))
    print(randomGroups(40))
    print(randomGroups(40))
    print(randomGroups(40))

if __name__ == "__main__":
    main()
