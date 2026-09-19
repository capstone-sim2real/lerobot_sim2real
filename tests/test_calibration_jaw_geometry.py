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
