"""Click the block centre in each calibration image to fill points.csv.

The calibration pairs a pixel with a robot-frame coordinate (AGENTS.md §6):
so101_record_calibration_point.py records the FK side (x_m, y_m, z_m), this
tool records the pixel side (u_px, v_px).

Serve the images, click each one in a browser, and the pixels are written
back to points.csv:

    python -m tools.pick_pixels
    # then open http://<host>:8091/

A browser UI rather than cv2.imshow because the team runs opencv-headless
(no GUI) and works on the Orin remotely.

Click the centre of the *block*, on an image taken with the arm moved away —
the gripper hides the block in the frame captured at record time. Zoom with
the mouse wheel; the click maps back to full-resolution pixels either way.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CSV_FIELDS = ["name", "image", "u_px", "v_px", "x_m", "y_m", "z_m", "notes"]

PAGE = """<!doctype html><meta charset=utf-8><title>calibration pixel picker</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#eee}
 header{padding:8px 12px;background:#1c1c1c;position:sticky;top:0;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 button{font:inherit;padding:4px 10px;border-radius:5px;border:1px solid #555;background:#2a2a2a;color:#eee;cursor:pointer}
 button:hover{background:#383838}
 #wrap{overflow:auto;max-height:calc(100vh - 96px)}
 canvas{cursor:crosshair;display:block}
 td,th{padding:2px 10px;text-align:left;border-bottom:1px solid #333}
 .done{color:#6c6} .todo{color:#e94}
</style>
<header>
 <span id=name></span>
 <button onclick=prev()>&larr; prev</button><button onclick=next()>next &rarr;</button>
 <button onclick=zoom(1.25)>zoom +</button><button onclick=zoom(0.8)>zoom -</button><button onclick=zoom(0)>fit</button>
 <span id=msg></span>
</header>
<div id=wrap><canvas id=c></canvas></div>
<table id=tbl></table>
<script>
let pts=[],i=0,img=new Image(),sc=1,cur=null;
const c=document.getElementById('c'),g=c.getContext('2d');
async function load(){pts=await (await fetch('/points')).json();if(!pts.length){document.getElementById('msg').textContent='points.csv is empty';return}show()}
function show(){const p=pts[i];document.getElementById('name').textContent=`[${i+1}/${pts.length}] ${p.name} — ${p.image}`;
 cur=(p.u_px!==''&&p.v_px!=='')?{x:+p.u_px,y:+p.v_px}:null;img.onload=()=>{sc=Math.min(1200/img.width,1);draw()};img.src='/img/'+encodeURIComponent(p.image)+'?t='+Date.now();table()}
function draw(){c.width=img.width*sc;c.height=img.height*sc;g.imageSmoothingEnabled=false;g.drawImage(img,0,0,c.width,c.height);
 if(cur){const x=cur.x*sc,y=cur.y*sc;g.strokeStyle='#0f0';g.lineWidth=1;g.beginPath();g.moveTo(x-14,y);g.lineTo(x+14,y);g.moveTo(x,y-14);g.lineTo(x,y+14);g.stroke();
 g.strokeStyle='#0f08';g.beginPath();g.arc(x,y,9,0,7);g.stroke();
 document.getElementById('msg').textContent=`u=${cur.x.toFixed(1)} v=${cur.y.toFixed(1)}`}}
c.onclick=async e=>{const r=c.getBoundingClientRect();cur={x:(e.clientX-r.left)/sc,y:(e.clientY-r.top)/sc};draw();
 await fetch('/set?name='+encodeURIComponent(pts[i].name)+'&u='+cur.x.toFixed(2)+'&v='+cur.y.toFixed(2));
 pts[i].u_px=cur.x.toFixed(2);pts[i].v_px=cur.y.toFixed(2);table()}
function next(){i=(i+1)%pts.length;show()} function prev(){i=(i-1+pts.length)%pts.length;show()}
function zoom(f){sc=f?sc*f:Math.min(1200/img.width,1);draw()}
function table(){document.getElementById('tbl').innerHTML='<tr><th>point<th>u_px<th>v_px<th>x_m<th>y_m'+
 pts.map((p,k)=>`<tr class=${p.u_px?'done':'todo'} style="${k==i?'background:#222':''}"><td>${p.name}<td>${p.u_px||'—'}<td>${p.v_px||'—'}<td>${(+p.x_m).toFixed(3)}<td>${(+p.y_m).toFixed(3)}`).join('')}
onkeydown=e=>{if(e.key=='ArrowRight')next();if(e.key=='ArrowLeft')prev()};load();
</script>"""


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def make_handler(csv_path: Path, image_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/points":
                self._send(json.dumps(read_rows(csv_path)).encode(), "application/json")
            elif u.path.startswith("/img/"):
                name = Path(html.unescape(u.path[len("/img/"):])).name
                p = image_dir / name
                if not p.is_file():
                    self.send_error(404, f"no such image: {name}")
                    return
                self._send(p.read_bytes(), "image/jpeg")
            elif u.path == "/set":
                q = parse_qs(u.query)
                name, uu, vv = q["name"][0], q["u"][0], q["v"][0]
                rows = read_rows(csv_path)
                for r in rows:
                    if r["name"] == name:
                        r["u_px"], r["v_px"] = uu, vv
                write_rows(csv_path, rows)
                print(f"  {name}: u={uu} v={vv}")
                self._send(b"ok", "text/plain")
            else:
                self.send_error(404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, default=Path("docs/calibration/points.csv"))
    ap.add_argument("--image-dir", type=Path, default=None, help="default: the points.csv directory")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args(argv)

    if not args.points.exists():
        print(f"ERROR: {args.points} not found — record calibration points first.")
        return 2
    image_dir = args.image_dir or args.points.parent
    rows = read_rows(args.points)
    todo = [r["name"] for r in rows if not r["u_px"]]
    print(f"{len(rows)} points, {len(todo)} still need a pixel: {todo or '(none)'}")
    print(f"open http://<this-host>:{args.port}/   (Ctrl-C to stop)")
    HTTPServer((args.host, args.port), make_handler(args.points, image_dir)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
