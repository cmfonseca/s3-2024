def parse_input():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    M = int(data[index])
    P = int(data[index + 1])
    index += 2
    
    d = [int(data[i]) for i in range(index, index + M)]
    index += M
    
    a = []
    for i in range(P):
        a.append([int(data[j]) for j in range(index, index + M)])
        index += M
    
    return M, P, d, a

def calculate_cost(u, M, P, d, a):
    T = sum(d)
    rp = [sum(a[p][m] * d[m] for m in range(M)) for p in range(P)]
    
    cost = 0
    for t in range(1, T + 1):
        for p in range(P):
            actual_demand = sum(a[p][u[i]] for i in range(t))
            target_demand = t * rp[p] / T
            cost += (target_demand - actual_demand) ** 2
    
    return cost

def main():
    M, P, d, a = parse_input()
    
    # Initialize with a naive solution (e.g., assembling in the given order of models)
    T = sum(d)
    u = []
    for m in range(M):
        u.extend([m] * d[m])
    
    # Calculate the initial cost
    initial_cost = calculate_cost(u, M, P, d, a)
    print(f"Initial cost: {initial_cost}")
    
    # TODO: Implement optimization to find the best sequence `u`
    
    # Print the sequence `u`
    for i in u:
        print(i)

if __name__ == "__main__":
    main()