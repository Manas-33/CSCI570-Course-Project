"""
Memory Efficient Sequence Alignment - Hirschberg's Algorithm
Authors: Manas & Sampriti
CSCI-570 Final Project

Space Complexity: O(m + n) instead of O(m * n)
Uses divide-and-conquer with linear space DP
"""

import sys
import time
import psutil

# Constants
DELTA = 30  # Gap penalty

ALPHA = {  # Mismatch cost matrix
    'A': {'A': 0,   'C': 110, 'G': 48,  'T': 94},
    'C': {'A': 110, 'C': 0,   'G': 118, 'T': 48},
    'G': {'A': 48,  'C': 118, 'G': 0,   'T': 110},
    'T': {'A': 94,  'C': 48,  'G': 110, 'T': 0}
}


def process_memory():
    """Return memory usage in KB."""
    return psutil.Process().memory_info().rss // 1024


def read_input(path):
    """Parse input file into two parts for string generation."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    s_part, t_part = [], []
    current = s_part
    found_first = False
    
    for line in lines:
        if line.isalpha():
            if not found_first:
                s_part.append(line)
                found_first = True
            else:
                current = t_part
                t_part.append(line)
        else:
            current.append(int(line))
    
    return s_part, t_part


def generate_string(part):
    """Generate string by inserting at indices."""
    s = part[0]
    for idx in part[1:]:
        s = s[:idx + 1] + s + s[idx + 1:]
    return s


def space_efficient_alignment(X, Y):
    """
    Compute last row of DP table using O(min(m,n)) space.
    Returns array of costs for aligning X with Y[0:j] for all j.
    """
    m, n = len(X), len(Y)
    prev = [j * DELTA for j in range(n + 1)]
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr[0] = i * DELTA
        for j in range(1, n + 1):
            match = prev[j - 1] + ALPHA[X[i - 1]][Y[j - 1]]
            delete = prev[j] + DELTA
            insert = curr[j - 1] + DELTA
            curr[j] = min(match, delete, insert)
        prev, curr = curr, prev
    
    return prev


def hirschberg(X, Y):
    """
    Hirschberg's divide-and-conquer algorithm.
    Returns aligned strings using O(m + n) space.
    """
    m, n = len(X), len(Y)
    
    if m == 0:
        return '_' * n, Y
    if n == 0:
        return X, '_' * m
    if m == 1 or n == 1:
        return needleman_wunsch(X, Y)
    
    # Divide X at midpoint
    mid = m // 2
    
    # Compute scores: forward and backward
    score_left = space_efficient_alignment(X[:mid], Y)
    score_right = space_efficient_alignment(X[mid:][::-1], Y[::-1])
    
    # Find optimal split in Y
    split = min(range(n + 1), key=lambda j: score_left[j] + score_right[n - j])
    
    # Conquer: recursively solve subproblems
    left_X, left_Y = hirschberg(X[:mid], Y[:split])
    right_X, right_Y = hirschberg(X[mid:], Y[split:])
    
    return left_X + right_X, left_Y + right_Y


def needleman_wunsch(X, Y):
    """Standard DP for small base cases."""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i * DELTA
    for j in range(n + 1):
        dp[0][j] = j * DELTA
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + ALPHA[X[i - 1]][Y[j - 1]]
            delete = dp[i - 1][j] + DELTA
            insert = dp[i][j - 1] + DELTA
            dp[i][j] = min(match, delete, insert)
    
    # Backtrack
    i, j = m, n
    aligned_X, aligned_Y = [], []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + ALPHA[X[i - 1]][Y[j - 1]]:
            aligned_X.append(X[i - 1])
            aligned_Y.append(Y[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + DELTA:
            aligned_X.append(X[i - 1])
            aligned_Y.append('_')
            i -= 1
        else:
            aligned_X.append('_')
            aligned_Y.append(Y[j - 1])
            j -= 1
    
    return ''.join(reversed(aligned_X)), ''.join(reversed(aligned_Y))


def calculate_cost(aligned_X, aligned_Y):
    """Calculate alignment cost."""
    cost = 0
    for x, y in zip(aligned_X, aligned_Y):
        if x == '_' or y == '_':
            cost += DELTA
        else:
            cost += ALPHA[x][y]
    return cost


def main():
    if len(sys.argv) != 3:
        print("Usage: python efficient.py <input_file> <output_file>")
        return
    
    input_path, output_path = sys.argv[1], sys.argv[2]
    
    # Generate strings
    s_part, t_part = read_input(input_path)
    X = generate_string(s_part)
    Y = generate_string(t_part)
    
    # Run alignment with time/memory tracking
    mem_before = process_memory()
    t_start = time.time()
    
    aligned_X, aligned_Y = hirschberg(X, Y)
    cost = calculate_cost(aligned_X, aligned_Y)
    
    t_end = time.time()
    mem_after = process_memory()
    
    time_ms = (t_end - t_start) * 1000
    mem_used = mem_after - mem_before
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(f"{cost}\n")
        f.write(f"{aligned_X}\n")
        f.write(f"{aligned_Y}\n")
        f.write(f"{time_ms}\n")
        f.write(f"{mem_used}\n")


if __name__ == "__main__":
    main()
