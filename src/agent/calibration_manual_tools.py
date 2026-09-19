"""Legacy manual adapters for the fake-provider calibration server only."""
from .tools import ToolDef, _obj, _mm, _cell_span
from .provider.types import ToolSpec


def build_calibration_manual_tools(cfg):
    rel=cfg.agent.relative
    labels=list(cfg.agent.zone_slots.labels)
    span=_cell_span(cfg)
    cell={k:{'type':'integer','minimum':-span,'maximum':span} for k in ('x','y')}
    offsets={k:_mm('Bounded relative mm',rel.max_shift_mm) for k in ('forward_mm','left_mm')}
    definitions=[]
    def add(name,fields,required,run):
        definitions.append(ToolDef(ToolSpec(name,'Calibration manual compatibility adapter',_obj(fields,required)),run))
    add('move_arm',{k:_mm('Bounded jog mm',rel.max_jog_mm) for k in ('forward_mm','left_mm','up_mm')},[],lambda sk,a:sk.move_arm(**a))
    add('move_to_cell',cell,['x','y'],lambda sk,a:sk.move_to_cell(**a))
    add('place_at_cell',cell,['x','y'],lambda sk,a:sk.place_at_cell(**a))
    add('place_at_slot',{'slot':{'type':'string','enum':labels},**offsets},['slot'],lambda sk,a:sk.place_at_slot(labels.index(a['slot']),a.get('forward_mm',0),a.get('left_mm',0)))
    add('place_on_table',{'column':{'type':'string','enum':list(cfg.agent.table_regions.columns_deg)},'row':{'type':'string','enum':list(cfg.agent.table_regions.rows_fraction)}},[],lambda sk,a:sk.place_on_table(**a))
    add('place_here',{},[],lambda sk,a:sk.place_here())
    add('place_at_pixel',{k:{'type':'integer','minimum':0} for k in ('u','v')},['u','v'],lambda sk,a:sk.place_at_pixel(**a))
    return definitions
