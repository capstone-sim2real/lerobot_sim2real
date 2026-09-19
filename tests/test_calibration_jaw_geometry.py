import numpy as np
from session.calibration_jaw_geometry import triangles_intersect_box


def test_diagonal_face_does_not_fill_its_bounding_box():
    # AABB overlaps origin, but every vertex/edge of this triangle has x+y>=3.
    triangle=np.array([[[0,3,0],[3,0,0],[3,3,0]]],dtype=float)
    assert not triangles_intersect_box(triangle,[1,1,1])[0]


def test_real_crossing_and_top_contact_are_detected():
    triangles=np.array([[[-3,0,0],[3,0,0],[0,3,0]],
                        [[0,0,1],[3,0,1],[0,3,1]],
                        [[0,0,2],[3,0,2],[0,3,2]]],dtype=float)
    assert triangles_intersect_box(triangles,[1,1,1]).tolist()==[True,True,False]


def test_mount_rotation_changes_which_neighbour_the_jaw_reaches():
    from config import CalibrationClearanceConfig
    from session.calibration_jaw_geometry import JawGeometry
    geometry=object.__new__(JawGeometry)
    geometry.limits=(0.,1.)
    geometry.to_link=np.eye(4)
    geometry.joint_origin=np.eye(4)
    triangle=np.array([[.049,-.001,.01,1],[.051,-.001,.01,1],[.05,.001,.01,1]])
    geometry.boxes=[("gripper_link",triangle,triangle)]
    cfg=CalibrationClearanceConfig(uncertainty_mm=0,block_radius_mm=5,jaw_mount_yaw_deg=0)
    obstacle={"yellow":(0.,50.)}
    assert geometry.check([np.eye(4)],obstacle,cfg)["clear"]
    cfg.jaw_mount_yaw_deg=90.
    result=geometry.check([np.eye(4)],obstacle,cfg)
    assert not result["clear"] and result["conflicts"]==["yellow"]
