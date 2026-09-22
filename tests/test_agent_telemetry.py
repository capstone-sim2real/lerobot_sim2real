from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace as NS
import threading

from agent.telemetry import collect
from agent.service import AgentService
from agent.worker import RobotWorker
from config import AppConfig

@dataclass
class Cal:
    id: int = 1
    drive_mode: int = 0
    homing_offset: int = 12
    range_min: int = 1024
    range_max: int = 3071

class Bus:
    def __init__(self):
        self.calibration={'shoulder_pan':Cal()}
        self.motors={'shoulder_pan':NS(id=1,model='test',norm_mode=NS(name='DEGREES'))}
        self.model_resolution_table={'test':4096}
        self.threads=[]
    def sync_read(self, register, normalize=False):
        self.threads.append(threading.current_thread().name)
        if register=='Present_Temperature': raise OSError('offline')
        return {'shoulder_pan': 10 if register=='Goal_Position' else 5}
    def read_calibration(self): return self.calibration

def test_read_only_partial_failure_units_and_worker_ownership():
    bus=Bus()
    s=NS(_inner_robot=NS(robot=NS(bus=bus,calibration_fpath='cal.json')),
         ik=NS(forward_position_mm=lambda x:(x['shoulder_pan'],0,0)),cfg=AppConfig(),held=None)
    worker=RobotWorker(lambda:NS(s=s))
    worker.start()
    try: result=worker.run(collect)
    finally: worker.stop()
    m=result['motors'][0]
    assert m['unit']=='°' and m['error']==5
    assert m['temperature'] is None and result['errors']
    assert result['fk_distance']==5 and result['calibration_match']
    assert set(bus.threads)=={'so101-robot'}
    assert m['minimum']<0<m['maximum']

def test_polling_has_at_most_one_queued_read():
    pending=Future()
    calls=[]
    service=NS(_telemetry_future=None,_telemetry_cache={},cfg=AppConfig(),
               _worker=NS(submit=lambda fn:(calls.append(fn) or pending)),gate=NS(snapshot=lambda:{'state':'busy'}))
    for _ in range(20):
        assert AgentService.telemetry(service)['pending']
    assert len(calls)==1
    pending.set_exception(OSError('disconnected'))
    d=AgentService.telemetry(service)
    assert d['error']=='OSError'
