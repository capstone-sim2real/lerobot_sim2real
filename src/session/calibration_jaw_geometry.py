"""URDF collision-mesh bounds for experimental neighbour screening.

Each mesh keeps its own box. Moving jaw covers its entire URDF joint range,
so no unverified servo-percent to jaw-angle conversion is needed. Mesh boxes
are conservative, and this is not a full-arm collision checker.
"""
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


def triangles_intersect_box(triangles, half_extents):
    """Exact separating-axis test of triangles against an axis-aligned box.

    Coordinates are relative to the box centre. Unlike per-triangle AABBs,
    slanted faces do not fill their entire rectangular projection.
    """
    import numpy as np
    t=np.asarray(triangles,dtype=float)
    half=np.asarray(half_extents,dtype=float)
    hit=np.all(t.min(axis=1)<=half,axis=1)&np.all(t.max(axis=1)>=-half,axis=1)
    edges=np.roll(t,-1,axis=1)-t
    axes=[np.cross(edges[:,0],edges[:,1])]
    for edge in range(3):
        for basis in np.eye(3):
            axes.append(np.cross(edges[:,edge],basis))
    for axis in axes:
        projection=np.einsum('nvi,ni->nv',t,axis)
        radius=np.abs(axis)@half
        hit &= (projection.min(axis=1)<=radius)&(projection.max(axis=1)>=-radius)
    return hit


def transform(node):
    import numpy as np
    xyz = [float(v) for v in node.get("xyz", "0 0 0").split()]
    r,p,y = [float(v) for v in node.get("rpy", "0 0 0").split()]
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    t=np.eye(4)
    t[:3,:3]=[[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
              [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]
    t[:3,3]=xyz
    return t


class JawGeometry:
    def __init__(self, urdf_path):
        import numpy as np
        path=Path(urdf_path); root=ET.parse(path).getroot()
        tcp=transform(root.find("./joint[@name='gripper_frame_joint']/origin"))
        self.to_link=np.linalg.inv(tcp)
        joint=root.find("./joint[@name='gripper']")
        self.joint_origin=transform(joint.find("origin"))
        self.limits=tuple(float(joint.find("limit").get(k)) for k in ("lower","upper"))
        self.boxes=[]
        for name in ("gripper_link","moving_jaw_so101_v1_link"):
            for part in root.findall("./link[@name='%s']/collision"%name):
                mesh=part.find("geometry/mesh"); data=(path.parent/mesh.get("filename")).read_bytes()
                count=struct.unpack_from("<I",data,80)[0]
                if len(data)!=84+50*count: raise ValueError("Expected binary STL")
                triangles=np.frombuffer(data, dtype=np.dtype([("normal","<f4",3),("v","<f4",(3,3)),("attr","<u2")]),offset=84)
                vertices=triangles["v"].reshape(-1,3).astype(float)
                vertices*=np.array([float(v) for v in mesh.get("scale","1 1 1").split()])
                lo,hi=vertices.min(axis=0),vertices.max(axis=0)
                corners=np.array([[x,y,z,1] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
                origin=transform(part.find("origin"))
                homogeneous=np.column_stack((vertices,np.ones(len(vertices))))
                self.boxes.append((name,(origin@corners.T).T,(origin@homogeneous.T).T))

    def check(self, tcp_poses, obstacles, cfg):
        import numpy as np
        step=math.radians(cfg.jaw_angle_step_deg)
        if not math.isfinite(step) or step<=0: raise ValueError("Invalid jaw angle step")
        angles=np.linspace(*self.limits,max(2,math.ceil((self.limits[1]-self.limits[0])/step)+1))
        local=[]
        for name,box,triangles in self.boxes:
            for angle in (angles if name.startswith("moving") else [None]):
                t=self.to_link
                padding=0.0
                if angle is not None:
                    r=np.eye(4);c,s=math.cos(angle),math.sin(angle);r[:2,:2]=[[c,-s],[s,c]]
                    t=t@self.joint_origin@r
                    padding=float(np.linalg.norm(box[:,:2],axis=1).max())*step*1000
                local.append((name,(t@box.T).T,padding,(t@triangles.T).T))
        obstacle_frames={}
        for color,value in obstacles.items():
            if isinstance(value,dict):
                corners=np.asarray(value["box"],dtype=float)
                if corners.shape!=(4,2) or not np.isfinite(corners).all():raise ValueError("Invalid obstacle box")
                edges=[corners[1]-corners[0],corners[3]-corners[0]]
                lengths=[np.linalg.norm(e) for e in edges]
                if min(lengths)<=0:raise ValueError("Empty obstacle box")
                axes=np.stack([edges[i]/lengths[i] for i in (0,1)],axis=1)
                center=corners.mean(axis=0)
                half=np.asarray(lengths)/2
            else:
                center=np.asarray(value);axes=np.eye(2);half=np.array([cfg.block_radius_mm]*2)
            obstacle_frames[color]=(center,axes,half)
        conflicts=set();pieces=set()
        for pose in tcp_poses:
            for name,box,padding,triangles in local:
                xyz=(pose@box.T).T[:,:3]*1000
                lo,hi=xyz.min(axis=0),xyz.max(axis=0)
                margin=cfg.uncertainty_mm+padding
                if lo[2]-margin>cfg.obstacle_height_mm:continue
                for color,(center,axes,half) in obstacle_frames.items():
                    if color in conflicts:continue
                    projected=(xyz[:,:2]-center)@axes
                    radius=half+margin
                    if np.any(projected.min(axis=0)>radius) or np.any(projected.max(axis=0)<-radius):continue
                    verts=(pose@triangles.T).T[:,:3].reshape(-1,3,3)*1000
                    triangle_local=np.empty_like(verts)
                    triangle_local[:,:,:2]=(verts[:,:,:2]-center)@axes
                    triangle_local[:,:,2]=verts[:,:,2]-cfg.obstacle_height_mm/2
                    half3=np.array([*radius,cfg.obstacle_height_mm/2+margin])
                    broad=np.all(triangle_local.min(axis=1)<=half3,axis=1)&np.all(triangle_local.max(axis=1)>=-half3,axis=1)
                    if np.any(broad) and np.any(triangles_intersect_box(triangle_local[broad],half3)):
                        conflicts.add(color);pieces.add(name)
        return {"clear":not conflicts,"conflicts":sorted(conflicts),"colliding_links":sorted(pieces),
                "geometry":"URDF triangle-box SAT, full moving-jaw sweep","physical_geometry_verified":False}
