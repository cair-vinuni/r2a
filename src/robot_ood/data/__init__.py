"""Where this package reads its inputs from disk.

Everything that opens a demonstration file goes through robot_ood.data.libero, so the demo order a
builder sees, the order a calibration measures, and the order a cache indexes are one decision in
one place rather than five copies that agree by luck.
"""
