#!/usr/bin/env python3
"""
Visualization of mutation distributions.
Compares the old "power" distribution vs the new "gaussian" distribution.
"""
import math
import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
COLLISION_FRAME = 100
MUTATION_WINDOW_SIZE = 20
MUTATION_CENTER_OFFSET = 8
MUTATION_GAUSSIAN_STD = 3.5
MUTATION_DISTANCE_POWER = 0.8

# Create candidate frames
window_start = COLLISION_FRAME - MUTATION_WINDOW_SIZE
window_end = COLLISION_FRAME
candidate_frames = list(range(window_start, window_end))

# Calculate weights for POWER distribution
power_weights = [1.0 / ((COLLISION_FRAME - frame) ** MUTATION_DISTANCE_POWER)
                 for frame in candidate_frames]
# Normalize to probabilities
power_total = sum(power_weights)
power_probs = [w / power_total for w in power_weights]

# Calculate weights for GAUSSIAN distribution
center = COLLISION_FRAME - MUTATION_CENTER_OFFSET
std = MUTATION_GAUSSIAN_STD
gaussian_weights = [math.exp(-0.5 * ((frame - center) / std) ** 2)
                    for frame in candidate_frames]
# Normalize to probabilities
gaussian_total = sum(gaussian_weights)
gaussian_probs = [w / gaussian_total for w in gaussian_weights]

# Create the visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Power distribution
ax1.bar(candidate_frames, power_probs, width=0.8, alpha=0.7, color='steelblue')
ax1.axvline(COLLISION_FRAME, color='red', linestyle='--', linewidth=2, label='Collision point')
ax1.set_xlabel('Frame number', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title(f'Power Distribution (power={MUTATION_DISTANCE_POWER})\nBiased toward collision', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Gaussian distribution
ax2.bar(candidate_frames, gaussian_probs, width=0.8, alpha=0.7, color='forestgreen')
ax2.axvline(COLLISION_FRAME, color='red', linestyle='--', linewidth=2, label='Collision point')
ax2.axvline(center, color='orange', linestyle='--', linewidth=2, label=f'Center (collision - {MUTATION_CENTER_OFFSET})')
ax2.set_xlabel('Frame number', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title(f'Gaussian Distribution (center offset={MUTATION_CENTER_OFFSET}, std={MUTATION_GAUSSIAN_STD})\nCentered 8 frames before collision', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mutation_distribution_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to: mutation_distribution_comparison.png")

# Print statistics
print("\nStatistics for POWER distribution:")
print(f"  Mean frame: {sum(f * p for f, p in zip(candidate_frames, power_probs)):.1f}")
print(f"  Frames before collision: {COLLISION_FRAME - sum(f * p for f, p in zip(candidate_frames, power_probs)):.1f}")
max_power_prob = max(power_probs)
max_power_frame = candidate_frames[power_probs.index(max_power_prob)]
print(f"  Peak probability at frame: {max_power_frame} ({COLLISION_FRAME - max_power_frame} frames before collision)")

print("\nStatistics for GAUSSIAN distribution:")
print(f"  Mean frame: {sum(f * p for f, p in zip(candidate_frames, gaussian_probs)):.1f}")
print(f"  Frames before collision: {COLLISION_FRAME - sum(f * p for f, p in zip(candidate_frames, gaussian_probs)):.1f}")
max_gaussian_prob = max(gaussian_probs)
max_gaussian_frame = candidate_frames[gaussian_probs.index(max_gaussian_prob)]
print(f"  Peak probability at frame: {max_gaussian_frame} ({COLLISION_FRAME - max_gaussian_frame} frames before collision)")

# Calculate cumulative probabilities
print("\nCumulative probabilities (gaussian):")
cumsum = 0
for i, (frame, prob) in enumerate(zip(candidate_frames, gaussian_probs)):
    cumsum += prob
    frames_before = COLLISION_FRAME - frame
    if frames_before in [4, 6, 8, 10, 12, 15]:
        print(f"  {cumsum*100:.1f}% of mutations occur {frames_before}+ frames before collision")