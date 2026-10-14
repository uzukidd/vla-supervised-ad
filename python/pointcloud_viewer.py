"""Offline semantic LiDAR point-cloud views in sensor-local coordinates (z up).

Only HTML display points are deterministically sampled; source PLY/NPZ and
caller-provided metrics remain complete. Colors always represent labels or
coverage, never RGB photographs. Matplotlib is imported only for PNG output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


UNCOVERED_COLOR = (95, 105, 120)
UNLABELED_COLOR = (190, 190, 190)
IGNORED_COLOR = (65, 70, 80)
CORRECT_COLOR = (45, 190, 95)
ERROR_COLOR = (235, 65, 65)
COVERED_COLOR = (56, 189, 248)


def _validate(xyz, prediction, target, covered, class_names, palette, summary):
    for name, value in (("xyz", xyz), ("prediction", prediction),
                        ("covered", covered), ("palette", palette)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if not (np.issubdtype(xyz.dtype, np.integer) or np.issubdtype(xyz.dtype, np.floating)):
        raise TypeError("xyz must have a real numeric dtype")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz must contain finite coordinates; points are never silently removed")
    count = len(xyz)
    if prediction.shape != (count,) or covered.shape != (count,):
        raise ValueError("prediction and covered must have shape (N,) matching xyz")
    if prediction.dtype != np.uint8:
        raise TypeError("prediction must have dtype uint8")
    if covered.dtype != np.bool_:
        raise TypeError("covered must have dtype bool")
    if target is not None:
        if not isinstance(target, np.ndarray):
            raise TypeError("target must be a numpy.ndarray or None")
        if target.shape != (count,):
            raise ValueError("target must have shape (N,) matching xyz")
        if not np.issubdtype(target.dtype, np.integer):
            raise TypeError("target must have an integer dtype")
        if np.any((target < 0) | (target > 255)):
            raise ValueError("target IDs must be in [0, 255]")
    if palette.shape != (256, 3):
        raise ValueError("palette must have shape (256, 3)")
    if palette.dtype != np.uint8:
        raise TypeError("palette must have dtype uint8")
    if not isinstance(class_names, dict):
        raise TypeError("class_names must be a dict")
    for key, name in class_names.items():
        if (not isinstance(key, (int, np.integer)) or isinstance(key, (bool, np.bool_))
                or not 0 <= key <= 255):
            raise ValueError("class IDs must be integers in [0, 255]")
        if not isinstance(name, str):
            raise TypeError("class names must be strings")
    if not isinstance(summary, dict):
        raise TypeError("summary must be a dict")
    # Validate before creating any output; reject NaN/Infinity rather than emit
    # non-standard JSON. NumPy scalars and arrays are common in metric reports.
    _json(summary)


def _json(value):
    def numpy_value(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), default=numpy_value)


def _display_colors(prediction, target, covered, palette):
    """Return label-derived RGB: full GT; covered-only error comparisons."""
    prediction_colors = palette[prediction].copy()
    prediction_colors[prediction == 0] = UNLABELED_COLOR
    prediction_colors[~covered] = UNCOVERED_COLOR
    coverage_colors = np.empty((len(prediction), 3), dtype=np.uint8)
    coverage_colors[:] = COVERED_COLOR
    coverage_colors[~covered] = UNCOVERED_COLOR
    colors = {"prediction": prediction_colors, "coverage": coverage_colors}
    if target is not None:
        gt_colors = palette[target].copy()
        gt_colors[target == 0] = IGNORED_COLOR
        error_colors = np.empty_like(gt_colors)
        error_colors[:] = IGNORED_COLOR
        valid = covered & (target != 0)
        error_colors[valid & (prediction == target)] = CORRECT_COLOR
        error_colors[valid & (prediction != target)] = ERROR_COLOR
        colors.update(gt=gt_colors, error=error_colors)
    return colors


def _sample_indices(count: int, max_points: int) -> np.ndarray:
    if (not isinstance(max_points, (int, np.integer))
            or isinstance(max_points, (bool, np.bool_))):
        raise TypeError("max_points must be a positive integer")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    # Evenly spaced, unique, ordered indices including both ends (unless only
    # one display point is requested). No random state or class-based bias.
    return np.linspace(0, count - 1, int(max_points), dtype=np.int64)


def save_pointcloud_html(
    path: Path, xyz: np.ndarray, prediction: np.ndarray,
    target: np.ndarray | None, covered: np.ndarray,
    class_names: dict[int, str], palette: np.ndarray, summary: dict,
    max_points: int = 60000,
) -> None:
    """Write self-contained interactive WebGL HTML; require no network or CDN.

    Left drag rotates, right/Shift drag pans, wheel/pinch zooms. GT and error
    controls are disabled without target. Sampling only affects visualization.
    All input points determine the framing; there is no automatic range crop.
    Non-finite coordinates raise ValueError instead of being silently dropped.
    """
    _validate(xyz, prediction, target, covered, class_names, palette, summary)
    indices = _sample_indices(len(xyz), max_points)
    selected_target = None if target is None else target[indices]
    colors = _display_colors(prediction[indices], selected_target, covered[indices], palette)
    bounds = ([xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]
              if len(xyz) else [[0, 0, 0], [0, 0, 0]])
    data = {
        "total_points": len(xyz), "rendered_points": len(indices),
        "covered_points": int(np.count_nonzero(covered)),
        "sampled_indices": indices.tolist(), "xyz": xyz[indices].tolist(),
        "prediction": prediction[indices].tolist(),
        "target": None if selected_target is None else selected_target.tolist(),
        "covered": covered[indices].tolist(), "bounds": bounds,
        "class_names": {int(key): name for key, name in class_names.items()},
        "palette": palette.tolist(), "summary": summary,
        "colors": {key: value.tolist() for key, value in colors.items()},
    }
    # A literal '<' can terminate even an application/json script element.
    # Dynamic names/summary only enter the DOM through textContent below.
    encoded = (_json(data).replace("<", "\\u003c").replace(">", "\\u003e")
               .replace("&", "\\u0026").replace("\u2028", "\\u2028")
               .replace("\u2029", "\\u2029"))
    document = _HTML.replace("__POINTCLOUD_JSON__", encoded)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def save_pointcloud_preview(
    path: Path, xyz: np.ndarray, prediction: np.ndarray,
    target: np.ndarray | None, covered: np.ndarray,
    class_names: dict[int, str], palette: np.ndarray, summary: dict,
) -> None:
    """Save all points in top/oblique PNG panels for each available color mode."""
    _validate(xyz, prediction, target, covered, class_names, palette, summary)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    colors = _display_colors(prediction, target, covered, palette)
    modes = [mode for mode in ("prediction", "gt", "error", "coverage") if mode in colors]
    titles = {"prediction": "Prediction", "gt": "Ground truth (full scan)",
              "error": "Errors (covered, GT nonzero)", "coverage": "Camera coverage"}
    figure = Figure(figsize=(5 * len(modes), 10), facecolor="#111827")
    FigureCanvasAgg(figure)
    if len(xyz):
        lower, upper = xyz.min(axis=0).astype(float), xyz.max(axis=0).astype(float)
        center = (lower + upper) / 2
        # Full coordinate ranges with a small margin and equal metric scales.
        # Retain every outlier; short/constant axes get a visible finite span.
        span = np.maximum((upper - lower) * 1.04, 1.0)
    else:
        center, span = np.zeros(3), np.ones(3)
    limits = [(float(center[i] - span[i] / 2), float(center[i] + span[i] / 2))
              for i in range(3)]
    for row in range(2):
        for col, mode in enumerate(modes):
            top = row == 0
            axes = figure.add_subplot(2, len(modes), row * len(modes) + col + 1,
                                      **({} if top else {"projection": "3d"}))
            axes.set_facecolor("#111827")
            if len(xyz):
                if top:
                    axes.scatter(xyz[:, 0], xyz[:, 1], c=colors[mode] / 255,
                                 s=1.2, linewidths=0)
                else:
                    axes.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors[mode] / 255,
                                 s=1.2, linewidths=0, depthshade=False)
            axes.set(xlim=limits[0], ylim=limits[1], xlabel="x (m)", ylabel="y (m)",
                     title=f"{titles[mode]} — {'Top (z up)' if top else 'Oblique'}")
            if top:
                axes.set_aspect("equal", adjustable="box")
                for spine in axes.spines.values():
                    spine.set_color("#42536e")
                    spine.set_linewidth(0.5)
            else:
                axes.set(zlim=limits[2], zlabel="z up (m)")
                axes.view_init(elev=30, azim=-60)
                axes.set_proj_type("ortho")
                axes.set_box_aspect(span)
                axes.grid(False)
                for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
                    axis.pane.set_facecolor((0.067, 0.105, 0.172, 1))
                    axis.pane.set_edgecolor("#42536e")
                    axis.pane.set_alpha(1)
            axes.tick_params(colors="#94a3b8", labelsize=7, width=0.5, length=2)
            axes.title.set_color("#f8fafc")
            for axis in ((axes.xaxis, axes.yaxis) if top else (axes.xaxis, axes.yaxis, axes.zaxis)):
                axis.label.set_color("#cbd5e1")
                axis.set_major_locator(MaxNLocator(nbins=4))
    handles = [Line2D([], [], linestyle="", marker="o", color=np.array(UNCOVERED_COLOR) / 255,
                      label="Uncovered (prediction / coverage)"),
               Line2D([], [], linestyle="", marker="o", color=np.array(UNLABELED_COLOR) / 255,
                      label="Prediction 0 / unlabeled")]
    for key, name in sorted(class_names.items()):
        if key:
            handles.append(Line2D([], [], linestyle="", marker="o", color=palette[key] / 255,
                                  label=f"{key}: {name}"))
    if target is not None:
        for label, color in (("Correct (covered)", CORRECT_COLOR), ("Error (covered)", ERROR_COLOR),
                             ("GT 0 / ignored comparison", IGNORED_COLOR)):
            handles.append(Line2D([], [], linestyle="", marker="o", color=np.array(color) / 255,
                                  label=label))
    handles.append(Line2D([], [], linestyle="", marker="o", color=np.array(COVERED_COLOR) / 255,
                          label="Covered"))
    legend = figure.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
                           facecolor="#111827", labelcolor="#e2e8f0", fontsize=8)
    for text in legend.get_texts():
        text.set_parse_math(False)
    metrics_text = _summary_text(summary)
    figure.suptitle(f"LiDAR sensor-local coordinates, z up | {len(xyz):,} / {len(xyz):,} points\n"
                   f"Full points; no range crop. PLY/NPZ remain complete; metrics are not sampled.\n"
                   f"{metrics_text}", color="#f8fafc", fontsize=11, parse_math=False)
    figure.subplots_adjust(left=0.04, right=0.975, bottom=0.14, top=0.85, wspace=0.18, hspace=0.15)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=130, facecolor=figure.get_facecolor(),
                   metadata={"Description": "Full LiDAR sensor-local cloud; z up; no point sampling.",
                             "Summary": _json(summary)})
    figure.clear()


def _summary_text(summary):
    """Compact PNG numeric report; the full summary is retained as metadata."""
    values = []
    def collect(mapping, prefix=""):
        for key, value in mapping.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                collect(value, name)
            elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                key_lower = str(key).lower()
                score = (0 if "iou" in key_lower else 1 if "accuracy" in key_lower
                         else 2 if "coverage" in key_lower else 3 if "points" in key_lower else 4)
                number = float(value)
                formatted = f"{number:,.6g}"
                if any(word in key_lower for word in ("iou", "accuracy", "coverage", "ratio", "fraction")) and 0 <= number <= 1:
                    formatted = f"{number * 100:.2f}%"
                values.append((score, name, formatted))
    collect(summary)
    selected = sorted(values, key=lambda item: item[0])[:8]
    labels = [f"{name.replace('mean_iou', 'mIoU')}: {value}" for _, name, value in selected]
    return "\n".join(" | ".join(labels[i:i + 4]) for i in range(0, len(labels), 4)) or "No numeric summary supplied"


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Offline LiDAR point cloud</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e2e8f0;font:14px system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 320px;height:100vh}section{position:relative;min-width:0}
canvas{display:block;width:100%;height:100%;touch-action:none}aside{overflow:auto;padding:20px;background:#111b2c}
h1{font-size:20px;margin:0 0 15px}h2{font-size:15px;margin:22px 0 10px}label{display:block;margin:14px 0}
select,button{background:#24334c;color:#e2e8f0;border:1px solid #42536e;border-radius:5px;padding:7px}
select{width:100%;margin-top:6px}button{cursor:pointer;margin-right:6px}input[type=range]{width:100%}
.note{color:#a6b4c8;font-size:12px;line-height:1.6}#status{position:absolute;left:15px;top:15px;padding:8px;background:#111b2ccc;pointer-events:none}
.legend-row{display:flex;gap:8px;align-items:center;margin:7px 0}.swatch{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.summary-row{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid #24334c;font-size:12px;overflow-wrap:anywhere}.summary-row strong{font-family:ui-monospace,monospace;text-align:right}
@media(max-width:700px){main{grid-template-columns:1fr;grid-template-rows:62vh auto;height:auto}aside{overflow:visible}}
</style></head><body><main><section><canvas id="cloud" aria-label="Interactive semantic LiDAR point cloud"></canvas><div id="status"></div></section>
<aside><h1>LiDAR point cloud</h1><div id="counts"></div>
<p class="note">Sensor-local coordinates; z points up. Colors represent semantic classes, never camera RGB.</p>
<label>Color mode<select id="mode"><option value="prediction">Prediction classes</option><option value="gt">Ground truth</option><option value="error">Errors</option><option value="coverage">Camera coverage</option></select></label>
<label><input id="coveredOnly" type="checkbox"> Only camera-covered points</label>
<label>Point size: <span id="sizeValue">3</span> px<input id="size" type="range" min="1" max="12" value="3" step="0.5"></label>
<button id="reset">Reset view</button><button id="top">Top view</button>
<p class="note">Drag to rotate; right-drag or Shift-drag to pan; wheel to zoom. On touch screens: one finger rotates; two fingers pan and pinch to zoom.</p>
<p class="note">Only this HTML display may be uniformly sampled. PLY/NPZ preserve all points; metrics use complete data and are not sampled. No automatic range crop or point removal.</p>
<h2>Legend</h2><div id="legend"></div><h2>Summary (complete data)</h2><div id="summary"></div></aside></main>
<script id="pointcloud-data" type="application/json">__POINTCLOUD_JSON__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('pointcloud-data').textContent);
const el=id=>document.getElementById(id),canvas=el('cloud'),status=el('status');
el('counts').textContent=`Total: ${data.total_points.toLocaleString()} · Drawn sample: ${data.rendered_points.toLocaleString()} · Covered (full): ${data.covered_points.toLocaleString()}`;
function summaryRows(value,path=[]){for(const [key,item] of Object.entries(value)){const parts=[...path,key];if(item!==null&&typeof item==='object'&&!Array.isArray(item)){summaryRows(item,parts);continue;}if(Array.isArray(item))continue;const row=document.createElement('div');row.className='summary-row';const name=document.createElement('span'),number=document.createElement('strong');name.textContent=parts.join(' / ').replaceAll('mean_iou','mIoU').replaceAll('_',' ');let text=item===null?'n/a':String(item);if(typeof item==='number'){text=item.toLocaleString(undefined,{maximumSignificantDigits:8});if(/iou|accuracy|coverage|dice|precision|recall|ratio|fraction/i.test(key)&&item>=0&&item<=1)text=`${(item*100).toFixed(2)}% (${text})`;}number.textContent=text;row.append(name,number);el('summary').append(row);}}
summaryRows(data.summary);
if(data.target===null){for(const option of el('mode').options){if(option.value==='gt'||option.value==='error')option.disabled=true;}}
function legendRow(color,label){const row=document.createElement('div');row.className='legend-row';const swatch=document.createElement('span');swatch.className='swatch';swatch.style.backgroundColor=`rgb(${color.join(',')})`;const text=document.createElement('span');text.textContent=label;row.append(swatch,text);el('legend').append(row);}
function updateLegend(){el('legend').replaceChildren(); const mode=el('mode').value;
 if(mode==='error'){legendRow([65,70,80],'Uncovered or GT 0: ignored');legendRow([45,190,95],'Correct (covered)');legendRow([235,65,65],'Error (covered)');}
 else if(mode==='coverage'){legendRow([95,105,120],'Uncovered');legendRow([56,189,248],'Camera covered');}
 else{if(mode==='prediction')legendRow([95,105,120],'Uncovered');legendRow(mode==='gt'?[65,70,80]:[190,190,190],mode==='gt'?'GT 0: ignored (full-scan GT)':'0: unlabeled');for(const [key,name] of Object.entries(data.class_names)){if(Number(key)!==0)legendRow(data.palette[Number(key)],`${key}: ${name}`);}}
}
updateLegend();
const gl=canvas.getContext('webgl',{antialias:true,alpha:false});
if(!gl){status.textContent='WebGL is unavailable. Open this file in a WebGL-capable browser; the PNG preview is also available.';}
else{try{
 const vs=`attribute vec3 aPosition;attribute vec3 aColor;uniform vec3 uRight,uUp,uForward;uniform float uAspect,uZoom,uSize;uniform vec2 uPan;varying vec3 vColor;void main(){gl_Position=vec4(dot(aPosition,uRight)*uZoom/uAspect+uPan.x,dot(aPosition,uUp)*uZoom+uPan.y,-dot(aPosition,uForward)*0.25,1.0);gl_PointSize=uSize;vColor=aColor;}`;
 const fs=`precision mediump float;varying vec3 vColor;void main(){vec2 p=gl_PointCoord-vec2(0.5);if(dot(p,p)>0.25)discard;gl_FragColor=vec4(vColor,1.0);}`;
 function shader(type,source){const result=gl.createShader(type);gl.shaderSource(result,source);gl.compileShader(result);if(!gl.getShaderParameter(result,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(result));return result;}
 const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 const positionBuffer=gl.createBuffer(),colorBuffer=gl.createBuffer();
 const positionLocation=gl.getAttribLocation(program,'aPosition'),colorLocation=gl.getAttribLocation(program,'aColor');
 const uniforms={};for(const name of ['uRight','uUp','uForward','uAspect','uZoom','uSize','uPan'])uniforms[name]=gl.getUniformLocation(program,name);
 const center=data.bounds[0].map((value,i)=>(value+data.bounds[1][i])/2);
 const scale=Math.max(Math.hypot(...data.bounds[0].map((value,i)=>data.bounds[1][i]-value))/2,0.001);
 let yaw=-Math.PI/3,elevation=Math.PI/6,zoom=0.85,pan=[0,0],drawn=0;
 function updateBuffers(){const positions=[],colors=[],mode=el('mode').value,only=el('coveredOnly').checked;
  for(let i=0;i<data.xyz.length;i++){if(only&&!data.covered[i])continue;for(let j=0;j<3;j++)positions.push((data.xyz[i][j]-center[j])/scale);colors.push(...data.colors[mode][i]);}
  drawn=positions.length/3;gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.STATIC_DRAW);gl.enableVertexAttribArray(positionLocation);gl.vertexAttribPointer(positionLocation,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,colorBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(colors),gl.STATIC_DRAW);gl.enableVertexAttribArray(colorLocation);gl.vertexAttribPointer(colorLocation,3,gl.UNSIGNED_BYTE,true,0,0);
  status.textContent=`Visible: ${drawn.toLocaleString()} / ${data.rendered_points.toLocaleString()} sampled / ${data.total_points.toLocaleString()} total`;
 }
 function render(){const ratio=window.devicePixelRatio||1,width=Math.max(1,Math.round(canvas.clientWidth*ratio)),height=Math.max(1,Math.round(canvas.clientHeight*ratio));if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}gl.viewport(0,0,width,height);gl.clearColor(0.043,0.071,0.125,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
  const c=Math.cos(yaw),s=Math.sin(yaw),ce=Math.cos(elevation),se=Math.sin(elevation);
  gl.uniform3f(uniforms.uRight,-s,c,0);gl.uniform3f(uniforms.uUp,-se*c,-se*s,ce);gl.uniform3f(uniforms.uForward,ce*c,ce*s,se);gl.uniform1f(uniforms.uAspect,width/height);gl.uniform1f(uniforms.uZoom,zoom*Math.min(width/height,1));gl.uniform1f(uniforms.uSize,Number(el('size').value)*ratio);gl.uniform2f(uniforms.uPan,pan[0],pan[1]);gl.drawArrays(gl.POINTS,0,drawn);
 }
 function reset(top=false){yaw=-Math.PI/3;elevation=top?Math.PI/2:Math.PI/6;zoom=0.85;pan=[0,0];render();}
 el('mode').addEventListener('change',()=>{updateLegend();updateBuffers();render();});el('coveredOnly').addEventListener('change',()=>{updateBuffers();render();});el('size').addEventListener('input',()=>{el('sizeValue').textContent=el('size').value;render();});el('reset').addEventListener('click',()=>reset());el('top').addEventListener('click',()=>reset(true));
 canvas.addEventListener('contextmenu',event=>event.preventDefault());
 const pointers=new Map();let pinch=null;
 canvas.addEventListener('pointerdown',event=>{canvas.setPointerCapture(event.pointerId);pointers.set(event.pointerId,{x:event.clientX,y:event.clientY,pan:event.button===2||event.shiftKey});pinch=null;});
 canvas.addEventListener('pointermove',event=>{if(!pointers.has(event.pointerId))return;const prev=pointers.get(event.pointerId),next={x:event.clientX,y:event.clientY,pan:prev.pan||event.shiftKey};pointers.set(event.pointerId,next);
  if(pointers.size===2){const [a,b]=[...pointers.values()],now={x:(a.x+b.x)/2,y:(a.y+b.y)/2,d:Math.hypot(a.x-b.x,a.y-b.y)};if(pinch){pan[0]+=2*(now.x-pinch.x)/canvas.clientWidth;pan[1]-=2*(now.y-pinch.y)/canvas.clientHeight;zoom=Math.min(10000,Math.max(0.01,zoom*now.d/Math.max(pinch.d,1)));}pinch=now;}
  else{const dx=next.x-prev.x,dy=next.y-prev.y;if(next.pan){pan[0]+=2*dx/canvas.clientWidth;pan[1]-=2*dy/canvas.clientHeight;}else{yaw-=dx*0.007;elevation=Math.min(Math.PI/2,Math.max(-Math.PI/2,elevation+dy*0.007));}}render();});
 function release(event){pointers.delete(event.pointerId);pinch=null;}canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
 canvas.addEventListener('wheel',event=>{event.preventDefault();zoom=Math.min(10000,Math.max(0.01,zoom*Math.exp(-event.deltaY*0.001)));render();},{passive:false});
 window.addEventListener('resize',render);if(typeof ResizeObserver!=='undefined')new ResizeObserver(render).observe(canvas);updateBuffers();render();
}catch(error){status.textContent=`WebGL initialization failed: ${error.message}`;}}
</script></body></html>
'''
