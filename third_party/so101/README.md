# SO-101 URDF assets

This directory contains the SO-101 URDF used by the project's CV+IK path.

- `so101.urdf`: calibrated robot description loaded by `control.ik.TopDownIK`
- `assets/`: STL meshes referenced by the URDF with relative `assets/...` paths

The files were imported from the project's previous
`SO-ARM100/Simulation/SO101` layout. Only the URDF and referenced STL meshes
are retained; the unused `.part` metadata files were removed.

Treat these as third-party assets. Verify their upstream source and licence
before redistributing them outside this project.
