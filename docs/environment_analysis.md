# Lunar Lander Environment Analysis

## Observation Space

-   **Dimensions**: 8 values
-   **State Vector**: [x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]
-   **Ranges**:
    -   Position: -2.5 to 2.5 (x), -2.5 to 2.5 (y)
    -   Velocity: -10 to 10 (both directions)
    -   Angle: -2π to 2π radians
    -   Leg contact: 0 (no contact) or 1 (contact)

## Action Space

-   **Type**: Discrete(4)
-   **Actions**:
    -   0: Do nothing
    -   1: Fire left orientation engine (rotate right)
    -   2: Fire main engine (thrust up)
    -   3: Fire right orientation engine (rotate left)

## Reward Structure

-   **Positive rewards**:
    -   Moving closer to landing pad
    -   Reducing velocity
    -   Reducing tilt angle
    -   +10 for each leg touching ground
    -   +100 for safe landing
-   **Negative rewards**:
    -   Moving away from pad
    -   High velocity
    -   High tilt
    -   -0.03 per frame for side engines
    -   -0.3 per frame for main engine
    -   -100 for crashing

## Success Criteria

Episode score ≥ 200 points

## Challenges Observed

From my random-action episodes in `src/explore_environment.py`:

### Highly Negative Rewards (Common for Random Agents)

All episodes scored between –380 and –67, which is expected because:

-   Random actions often fire engines unnecessarily (fuel penalties).
-   They frequently crash (–100 penalty).
-   They fail to stabilize angle or velocity.
-   They rarely achieve leg contact bonuses (+10 each).

### High Episodic Variance

Across the random observations:

-   Episode steps range from 70 to 134.
-   Rewards vary from –380 to –67.

This shows the environment is stochastic, with randomness coming from:

-   Initial forces
-   Orientation and velocity
-   Contact events
-   Terminal conditions (crash vs. drifting out of bounds)

### Difficult Control Requirements

-   The lander must manage angle, x/y velocity, and position simultaneously. A random policy cannot do this.
-   Slight horizontal drift quickly becomes unrecoverable without precise counter-thrust.
-   Excessive rotation makes it impossible to land upright.
-   The landing pad is small target.

### Engine Penalties Accumulate Quickly

Randomly firing engines—especially the main engine—adds:

-   –0.3 per frame for the main engine
-   –0.03 per frame for each side engine

This creates additional negative reward on top of uncontrolled motion.

### Crashes Dominate

Most episodes terminate due to:

-   Excess vertical velocity
-   Excess tilt
-   Hitting terrain before stabilizing

This explains your consistent negative triple-digit rewards.
