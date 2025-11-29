import sys
import time
import psutil

# Gap penalty
delta = 30

# Mismatch cost matrix
alpha = {
    'A': {'A': 0,   'C': 110, 'G': 48,  'T': 94},
    'C': {'A': 110, 'C': 0,   'G': 118, 'T': 48},
    'G': {'A': 48,  'C': 118, 'G': 0,   'T': 110},
    'T': {'A': 94,  'C': 48,  'G': 110, 'T': 0}
}

def process_memory():
    """Return memory usage of the current process in KB."""
    return psutil.Process().memory_info().rss // 1024

def read_input(path):
    """
    Read the input file and split it into two parts:
    s_part = [base_string_s0, n1, n2, ...]
    t_part = [base_string_t0, m1, m2, ...]
    """
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    s_part = []
    t_part = []
    current = s_part
    found_first_string = False

    for line in lines:
        # Lines containing only letters represent base strings
        if line.isalpha():
            if not found_first_string:
                s_part.append(line)
                found_first_string = True
            else:
                current = t_part
                t_part.append(line)
        else:
            # Numeric lines represent insertion indices
            current.append(int(line))

    return s_part, t_part

def generate_string(part):
    """
    Generate the expanded string using the insertion rules.
    """
    s = part[0]
    for idx in part[1:]:
        pos = idx + 1
        s = s[:pos] + s + s[pos:]
    return s

def sequence_alignment_basic(X, Y):
    """
    Standard dynamic programming implementation of sequence alignment.
    Returns the minimum cost and the aligned strings.
    """
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialization
    for i in range(1, m + 1):
        dp[i][0] = i * delta
    for j in range(1, n + 1):
        dp[0][j] = j * delta

    # Fill DP table
    for i in range(1, m + 1):
        xi = X[i - 1]
        for j in range(1, n + 1):
            yj = Y[j - 1]
            cost_match = dp[i - 1][j - 1] + alpha[xi][yj]
            cost_gap_y = dp[i - 1][j] + delta
            cost_gap_x = dp[i][j - 1] + delta
            dp[i][j] = min(cost_match, cost_gap_y, cost_gap_x)

    # Backtracking to construct aligned strings
    i, j = m, n
    aligned_X = []
    aligned_Y = []

    while i > 0 or j > 0:
        # Case 1: match/mismatch
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + alpha[X[i - 1]][Y[j - 1]]:
            aligned_X.append(X[i - 1])
            aligned_Y.append(Y[j - 1])
            i -= 1
            j -= 1
        # Case 2: gap in Y
        elif i > 0 and dp[i][j] == dp[i - 1][j] + delta:
            aligned_X.append(X[i - 1])
            aligned_Y.append('_')
            i -= 1
        # Case 3: gap in X
        else:
            aligned_X.append('_')
            aligned_Y.append(Y[j - 1])
            j -= 1

    aligned_X = ''.join(reversed(aligned_X))
    aligned_Y = ''.join(reversed(aligned_Y))

    return dp[m][n], aligned_X, aligned_Y

def main():
    """Main function: handle I/O, measure time/memory, and write output."""
    if len(sys.argv) != 3:
        print("Usage: python3 basic.py <input_file> <output_file>")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Read and generate input strings
    s_part, t_part = read_input(input_path)
    S = generate_string(s_part)
    T = generate_string(t_part)

    # Measure memory and time around DP
    mem_before = process_memory()
    t_start = time.time()

    cost, aligned_S, aligned_T = sequence_alignment_basic(S, T)

    t_end = time.time()
    mem_after = process_memory()

    time_ms = (t_end - t_start) * 1000.0
    mem_used = mem_after - mem_before

    # Write output file (5 required lines)
    with open(output_path, 'w') as f:
        f.write(str(cost) + "\n")
        f.write(aligned_S + "\n")
        f.write(aligned_T + "\n")
        f.write(str(time_ms) + "\n")
        f.write(str(mem_used) + "\n")

if __name__ == "__main__":
    main()
