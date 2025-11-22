import numpy as np
import random
import math

# Raag Bhairav scale 
notes = [0, 1, 2, 3, 4, 5, 6, 7]
note_names = ['Sa', 'Re♭', 'Ga', 'Ma', 'Pa', 'Dha♭', 'Ni', "Sa'"]

# Pakad (signature phrase) as note sequence
pakad = [2, 3, 1, 0, 6, 5, 6, 7]

# Designing energy function 
def energy(melody):
    E = 0

    for n in melody:
        if n not in notes:
            E += 50

    for i in range(len(melody) - 1):
        jump = abs(melody[i+1] - melody[i])
        if jump > 4:
            E += 10 * (jump - 4)

    if melody[-1] not in [0, 7]:
        E += 20

    if contains_subsequence(melody, pakad):
        E -= 40
        
    E += np.var(np.diff(melody))

    from itertools import groupby
    for _, group in groupby(melody):
        run_length = len(list(group))
        if run_length > 2:
            E += (run_length - 2) * 5  # or tweak weight
    # 7. Encourage use of vadi (Dha♭ = 5) and samvadi (Re♭ = 1)
    E -= melody.count(5) * 0.5
    E -= melody.count(1) * 0.3

    return E

def contains_subsequence(seq, subseq):
    """Check if subsequence appears within a sequence."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return True
    return False

# Simulated Annealing
def simulated_annealing(max_iters=5000, melody_length=16, T0=100.0, Tend=0.01):
    # Initialize random melody
    melody = [random.choice(notes) for _ in range(melody_length)]
    best = melody.copy()
    E_curr = energy(melody)
    E_best = E_curr

    def temperature(t):
        return T0 * (Tend / T0) ** (t / max_iters)

    for t in range(max_iters):
        T = temperature(t)
        # Generate neighbor: change one note
        new_melody = melody.copy()
        idx = random.randint(0, melody_length - 1)
        new_melody[idx] = random.choice(notes)

        E_new = energy(new_melody)
        deltaE = E_new - E_curr

        # Acceptance probability
        if deltaE < 0 or random.random() < math.exp(-deltaE / (T + 1e-12)):
            melody, E_curr = new_melody, E_new

        if E_curr < E_best:
            best, E_best = melody.copy(), E_curr

        # Optional: print progress
        if (t + 1) % (max_iters // 5) == 0:
            print(f"Iter {t+1}, Temp={T:.4f}, CurrE={E_curr:.2f}, BestE={E_best:.2f}")

    return best, E_best

# Run experiment
if __name__ == "__main__":
    best_melody, best_energy = simulated_annealing()
    print("\nBest melody (numeric):", best_melody)
    print("Best energy:", best_energy)
    print("Melody (notes):", ' - '.join(note_names[n] for n in best_melody))
