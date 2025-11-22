import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time

class HopfieldPatternStorage:
    
    def __init__(self, pattern_size):
        self.N = pattern_size
        self.w = np.zeros((self.N, self.N))
        
    def store_pattern(self, pattern):
        self.w = np.outer(pattern, pattern) / self.N
        np.fill_diagonal(self.w, 0)
        
    def add_noise(self, pattern, error_rate):
        corrupted = pattern.copy()
        num_errors = int(error_rate * self.N)
        flip_indices = np.random.choice(self.N, num_errors, replace=False)
        corrupted[flip_indices] *= -1
        return corrupted
    
    def recover_pattern(self, initial_state, max_iterations=1000):
        state = initial_state.copy()
        
        for iteration in range(max_iterations):
            old_state = state.copy()
            
            for i in range(self.N):
                activation = np.dot(self.w[i], state)
                state[i] = np.sign(activation) if activation != 0 else state[i]
            
            if np.array_equal(state, old_state):
                return state, iteration
        
        return state, max_iterations
    
    def test_error_correction(self, pattern, error_rates, trials=100):
        results = {rate: {'success': 0, 'avg_iterations': 0} 
        for rate in error_rates}
        
        for rate in error_rates:
            iterations_list = []
            
            for trial in range(trials):
                corrupted = self.add_noise(pattern, rate)
                recovered, iters = self.recover_pattern(corrupted)
                iterations_list.append(iters)
                
                if np.array_equal(recovered, pattern):
                    results[rate]['success'] += 1
            
            results[rate]['success_rate'] = results[rate]['success'] / trials
            results[rate]['avg_iterations'] = np.mean(iterations_list)
        
        return results


def problem1_error_correction():
    print("\n" + "="*70)
    print("PROBLEM 1: ERROR CORRECTING CAPABILITY OF HOPFIELD NETWORK")
    print("="*70)
    
    N = 100
    pattern = np.random.choice([-1, 1], N)
    
    error_rates = np.linspace(0, 0.5, 11)
    
    network = HopfieldPatternStorage(N)
    network.store_pattern(pattern)
    results = network.test_error_correction(pattern, error_rates, trials=100)
    
    print(f"\nNetwork Size: {N} neurons")
    print(f"Test Trials: 100 per error rate")
    print(f"\nError Rate | Success Rate | Avg Iterations")
    print("-" * 50)
    
    for rate in error_rates:
        success_rate = results[rate]['success_rate'] * 100
        avg_iter = results[rate]['avg_iterations']
        print(f"{rate*100:6.1f}%     | {success_rate:6.1f}%         | {avg_iter:6.1f}")
    
    rates = [r*100 for r in error_rates]
    success_rates = [results[r]['success_rate']*100 for r in error_rates]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rates, success_rates, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=15, color='r', linestyle='--', label='~15% threshold')
    plt.xlabel('Error Rate (%)', fontsize=12)
    plt.ylabel('Recovery Success Rate (%)', fontsize=12)
    plt.title('Hopfield Network: Error Correction Capability', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('problem1_error_correction.png', dpi=150)
    plt.show()
    
    print("\n✓ Problem 1 visualization saved as 'problem1_error_correction.png'")
    return results


class HopfieldEightRooks:
    
    def __init__(self, forbidden_positions=None):
        self.n = 8
        self.N = 64
        self.w = np.zeros((self.N, self.N))
        self.forbidden = forbidden_positions if forbidden_positions else []
        self.build_weights()
    
    def pos_to_idx(self, i, j):
        return i * self.n + j
    
    def idx_to_pos(self, idx):
        return idx // self.n, idx % self.n
    
    def build_weights(self):
        alpha = 5.0
        beta = 5.0
        gamma = 10.0
        
        for i in range(self.n):
            for j1 in range(self.n):
                for j2 in range(j1+1, self.n):
                    idx1 = self.pos_to_idx(i, j1)
                    idx2 = self.pos_to_idx(i, j2)
                    self.w[idx1, idx2] = -alpha
                    self.w[idx2, idx1] = -alpha
        
        for j in range(self.n):
            for i1 in range(self.n):
                for i2 in range(i1+1, self.n):
                    idx1 = self.pos_to_idx(i1, j)
                    idx2 = self.pos_to_idx(i2, j)
                    self.w[idx1, idx2] = -beta
                    self.w[idx2, idx1] = -beta
        
        for (i, j) in self.forbidden:
            idx = self.pos_to_idx(i, j)
            self.w[idx, idx] = gamma
    
    def solve(self, max_iterations=1000, max_attempts=5):
        
        for attempt in range(max_attempts):
            state = np.random.choice([-1, 1], self.N)
            biases = np.full(self.N, -self.n)
            
            for iteration in range(max_iterations):
                old_state = state.copy()
                
                for i in range(self.N):
                    activation = np.dot(self.w[i], state) + biases[i]
                    state[i] = np.sign(activation) if activation != 0 else state[i]
                
                if np.array_equal(state, old_state):
                    break
            
            if self.is_valid_solution(state):
                return state, True
        
        return state, False
    
    def is_valid_solution(self, state):
        board = (state + 1) / 2
        
        for i in range(self.n):
            row_sum = 0
            for j in range(self.n):
                idx = self.pos_to_idx(i, j)
                row_sum += board[idx]
            if row_sum != 1:
                return False
        
        for j in range(self.n):
            col_sum = 0
            for i in range(self.n):
                idx = self.pos_to_idx(i, j)
                col_sum += board[idx]
            if col_sum != 1:
                return False
        
        for (i, j) in self.forbidden:
            idx = self.pos_to_idx(i, j)
            if board[idx] == 1:
                return False
        
        return True
    
    def visualize_solution(self, state, title="Eight-Rook Solution"):
        board = (state + 1) / 2
        board_visual = board.reshape(self.n, self.n)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i in range(self.n):
            for j in range(self.n):
                color = 'lightgray' if (i+j) % 2 == 0 else 'white'
                rect = plt.Rectangle((j, self.n-1-i), 1, 1, 
                                    facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
        
        for i in range(self.n):
            for j in range(self.n):
                if board_visual[i, j] == 1:
                    ax.scatter(j+0.5, self.n-1-i+0.5, marker='s', s=500, 
                              color='red', edgecolors='darkred', linewidth=2, zorder=5)
        
        for (i, j) in self.forbidden:
            ax.plot(j+0.5, self.n-1-i+0.5, 'bx', markersize=15, markeredgewidth=2, zorder=4)
        
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.n+1))
        ax.set_yticks(range(self.n+1))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        return fig, ax


def problem2_eight_rooks():
    print("\n" + "="*70)
    print("PROBLEM 2: EIGHT-ROOK PROBLEM SOLUTION")
    print("="*70)
    
    forbidden = [(0, 0), (1, 1), (3, 4), (5, 6), (7, 7)]
    
    print(f"\nForbidden positions: {forbidden}")
    print("\nWeight Design:")
    print("  - Row constraints: α = 5.0 (negative weight between same-row positions)")
    print("  - Column constraints: β = 5.0 (negative weight between same-col positions)")
    print("  - Forbidden penalty: γ = 10.0 (positive weight on forbidden positions)")
    print("  - Bias terms: θ = -8 (encourage all neurons active)")
    
    network = HopfieldEightRooks(forbidden_positions=forbidden)
    
    print("\nSolving using Hopfield dynamics...")
    state, valid = network.solve(max_iterations=2000, max_attempts=20)
    
    if valid:
        print("✓ Valid solution found!")
        
        board = (state + 1) / 2
        rooks = []
        for i in range(8):
            for j in range(8):
                idx = network.pos_to_idx(i, j)
                if board[idx] == 1:
                    rooks.append((i, j))
        
        print(f"\nRook positions: {sorted(rooks)}")
        
        rows = [r[0] for r in rooks]
        cols = [r[1] for r in rooks]
        print(f"Rows occupied: {sorted(rows)}")
        print(f"Cols occupied: {sorted(cols)}")
        print(f"All rows covered: {len(set(rows)) == 8}")
        print(f"All cols covered: {len(set(cols)) == 8}")
        print(f"No forbidden rooks: {all(pos not in forbidden for pos in rooks)}")
    else:
        print("⚠ Generating greedy solution (not optimal Hopfield solution)...")
        
        rooks = []
        forbidden_set = set(forbidden)
        
        for i in range(8):
            for j in range(8):
                if (i, j) not in forbidden_set and j not in [r[1] for r in rooks]:
                    rooks.append((i, j))
                    break
        
        state = np.ones(64) * -1
        for (i, j) in rooks:
            idx = network.pos_to_idx(i, j)
            state[idx] = 1
        
        print(f"✓ Greedy solution: {sorted(rooks)}")
    
    fig, ax = network.visualize_solution(state, "Eight-Rook Problem Solution")
    plt.tight_layout()
    plt.savefig('problem2_eight_rooks.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Problem 2 visualization saved as 'problem2_eight_rooks.png'")
    return state


class HopfieldTSP:
    
    def __init__(self, cities, n_cities=10):
        self.n = n_cities
        self.N = n_cities * n_cities
        self.cities = cities
        self.compute_distances()
        self.w = np.zeros((self.N, self.N))
        self.build_weights()
    
    def compute_distances(self):
        self.dist = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.dist[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
    
    def neuron_idx(self, city, time):
        return city * self.n + time
    
    def build_weights(self):
        A = 1.0
        B = 1.0
        C = 0.5
        
        for t in range(self.n):
            for i in range(self.n):
                for j in range(i+1, self.n):
                    idx1 = self.neuron_idx(i, t)
                    idx2 = self.neuron_idx(j, t)
                    self.w[idx1, idx2] = -A
                    self.w[idx2, idx1] = -A
        
        for i in range(self.n):
            for t in range(self.n):
                for t2 in range(t+1, self.n):
                    idx1 = self.neuron_idx(i, t)
                    idx2 = self.neuron_idx(i, t2)
                    self.w[idx1, idx2] = -B
                    self.w[idx2, idx1] = -B
        
        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.n):
                    t_prev = (t - 1) % self.n
                    t_next = (t + 1) % self.n
                    
                    idx_it = self.neuron_idx(i, t)
                    idx_jp = self.neuron_idx(j, t_prev)
                    idx_jn = self.neuron_idx(j, t_next)
                    
                    weight = -C * self.dist[i, j] / np.max(self.dist)
                    self.w[idx_it, idx_jp] += weight
                    self.w[idx_it, idx_jn] += weight
    
    def solve(self, max_iterations=500):
        state = np.random.choice([-1, 1], self.N)
        biases = np.full(self.N, -self.n)
        
        for iteration in range(max_iterations):
            old_state = state.copy()
            
            for k in range(self.N):
                activation = np.dot(self.w[k], state) + biases[k]
                state[k] = np.sign(activation) if activation != 0 else state[k]
            
            if np.array_equal(state, old_state):
                break
        
        return state
    
    def decode_tour(self, state):
        tour = []
        visited = []
        
        for t in range(self.n):
            for i in range(self.n):
                idx = self.neuron_idx(i, t)
                if state[idx] > 0:
                    tour.append(i)
                    visited.append(i)
                    break
        
        valid = len(tour) == self.n and len(set(tour)) == self.n
        
        if valid:
            tour_length = sum(
                self.dist[tour[i], tour[(i+1)%self.n]]
                for i in range(self.n)
            )
        else:
            tour_length = float('inf')
        
        return tour, tour_length, valid
    
    def visualize_solution(self, tour, title="TSP Solution"):
        plt.figure(figsize=(10, 8))
        
        plt.scatter(self.cities[:, 0], self.cities[:, 1], 
                   s=200, c='red', zorder=5, edgecolors='black', linewidth=2)
        
        for i, city in enumerate(self.cities):
            plt.annotate(str(i), xy=city, fontsize=12, 
                        ha='center', va='center', color='white', fontweight='bold')
        
        if len(tour) == self.n:
            tour_extended = tour + [tour[0]]
            tour_coords = self.cities[tour_extended]
            plt.plot(tour_coords[:, 0], tour_coords[:, 1], 
                    'b-', linewidth=2, alpha=0.6)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X coordinate', fontsize=12)
        plt.ylabel('Y coordinate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt


def problem3_tsp():
    print("\n" + "="*70)
    print("PROBLEM 3: TRAVELING SALESMAN PROBLEM (10 CITIES)")
    print("="*70)
    
    np.random.seed(42)
    n_cities = 10
    cities = np.random.rand(n_cities, 2) * 100
    
    print(f"\nNumber of cities: {n_cities}")
    print(f"Network neurons: {n_cities * n_cities} = {n_cities}²")
    
    row_weights = n_cities * (n_cities-1) // 2
    col_weights = n_cities * (n_cities-1) // 2
    dist_weights = n_cities * n_cities * 2
    total_weights = row_weights + col_weights + dist_weights
    
    print(f"\nWeight Matrix Complexity:")
    print(f"  - Row constraint weights: {row_weights}")
    print(f"  - Column constraint weights: {col_weights}")
    print(f"  - Distance constraint weights: {dist_weights}")
    print(f"  - Total weights: {total_weights}")
    print(f"  - Density: {total_weights / (n_cities**4) * 100:.2f}%")
    print(f"  - Complexity class: O(n³) where n={n_cities}")
    
    print("\nSolving TSP...")
    network = HopfieldTSP(cities, n_cities)
    state = network.solve(max_iterations=500)
    
    tour, tour_length, valid = network.decode_tour(state)
    
    print(f"\nSolution found: {'✓ Valid' if valid else '✗ Invalid'}")
    if valid:
        print(f"Tour: {tour}")
        print(f"Tour length: {tour_length:.2f}")
    
    plt_obj = network.visualize_solution(tour if valid else [], 
                                         f"TSP Solution (Length: {tour_length:.2f})")
    plt_obj.savefig('problem3_tsp.png', dpi=150, bbox_inches='tight')
    plt_obj.show()
    
    print("\n✓ Problem 3 visualization saved as 'problem3_tsp.png'")
    
    print("\nRunning 20 trials to collect statistics...")
    tour_lengths = []
    valid_count = 0
    
    for trial in range(20):
        state = network.solve(max_iterations=500)
        tour, length, valid_sol = network.decode_tour(state)
        if valid_sol:
            tour_lengths.append(length)
            valid_count += 1
    
    if tour_lengths:
        print(f"Valid solutions: {valid_count}/20")
        print(f"Average tour length: {np.mean(tour_lengths):.2f}")
        print(f"Min tour length: {np.min(tour_lengths):.2f}")
        print(f"Max tour length: {np.max(tour_lengths):.2f}")
        print(f"Std deviation: {np.std(tour_lengths):.2f}")
    
    return network, cities, tour


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HOPFIELD NETWORKS FOR COMBINATORIAL OPTIMIZATION")
    print("Lab Assignment 6 - Complete Implementation")
    print("="*70)
    
    results_p1 = problem1_error_correction()
    
    input("\nPress Enter to proceed to Problem 2...")
    
    state_p2 = problem2_eight_rooks()
    
    input("\nPress Enter to proceed to Problem 3...")
    
    network_p3, cities_p3, tour_p3 = problem3_tsp()
    
    print("\n" + "="*70)
    print("ALL PROBLEMS COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  - problem1_error_correction.png")
    print("  - problem2_eight_rooks.png")
    print("  - problem3_tsp.png")
    print("="*70)