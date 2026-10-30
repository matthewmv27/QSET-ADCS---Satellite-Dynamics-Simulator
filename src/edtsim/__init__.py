"""edtsim: electrodynamic-tether CubeSat simulator.

Subpackages (imports may only point "down" this list; enforced by tests/test_architecture.py):

    core          rotations, frames, state contracts, constants, parameters
    inertia       rigid-body inertia building blocks and configurations      (Stage 1)
    environment   orbit + geomagnetic field tables                            (Stage 2)
    edt_current   bare-tether OML current profile                             (Stage 2)
    tether        lumped-mass tether chain dynamics                           (Stage 3)
    viz           plotting / animation of results files                       (Stage 4)
    attitude      future rigid-body attitude propagator
    coupling      future tether <-> bus two-way simulator
"""

__version__ = "0.1.0"
