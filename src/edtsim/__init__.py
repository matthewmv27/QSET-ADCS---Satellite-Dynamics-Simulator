"""edtsim: electrodynamic-tether CubeSat simulator.

Subpackages (imports may only point "down" this list; enforced by tests/test_architecture.py):

    core          rotations, constants, parameter containers
    inertia       rigid-body inertia building blocks and configurations      (Stage 1)
    environment   orbit + geomagnetic field tables                            (Stage 2)
    edt_current   bare-tether OML current profile                             (Stage 2)
    tether        lumped-mass tether chain dynamics                           (Stage 3)
    viz           plotting / animation of results files                       (Stage 4)
    coupled       tether <-> attitude coupled simulation                      (Stage 6)
"""

__version__ = "0.1.0"
