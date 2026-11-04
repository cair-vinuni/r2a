"""The checkpoint contract every trained head in this package is written and read through.

CheckpointedModule (base.py) is a mixin over torch.nn.Module: it owns the path resolution, the
meta/state split and the missing-file error (the .npz mechanics themselves live in robot_ood.core.npz, which
the artifact side shares), and leaves each head only the two things that are
really its own -- which shape constants it stamps, and how it rebuilds itself from them. It WRAPS
the .npz layout each head already had; no file on disk changes by a byte.
"""
