"""The latent servo: at t = 0, before the policy acts, move the robot until the policy's own view looks nominal.

limits   what the VR-H3.1 base and waist accept, from the robot team's measurements and deploy.yaml
prior    the per-task low-rank Gaussian over nominal start features, and its energy
motion   how a waypoint is executed: the waist turns in place (default) or the legs walk
search   the star search over the motion's axes
libero   the LIBERO side: placing the base in the simulator and reading the policy server's features
"""
