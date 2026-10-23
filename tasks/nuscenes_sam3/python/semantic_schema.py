"""Runtime semantic taxonomy and configurable plane classes, with legacy output support."""
from pathlib import Path
import json
import re
from shared.python.sam3 import Category,parse_class_file,make_palette

LEGACY_NAMES={0:'unassigned',1:'sidewalk',2:'road',3:'pedestrian',4:'vehicle',5:'bike',6:'rider'}
DEFAULT_PLANES=('sidewalk','road')
COLORS_BY_NAME={'unassigned':(170,177,188),'sidewalk':(210,70,180),'road':(210,155,20),
                'crosswalk':(35,190,205),'pedestrian':(35,110,225),'vehicle':(200,55,45),
                'bike':(40,160,85),'rider':(125,65,210)}
ZH={'unassigned':'未分配','sidewalk':'人行道','road':'道路','crosswalk':'人行横道','pedestrian':'行人',
    'vehicle':'车辆','bike':'两轮车','rider':'骑乘者'}


def categories_from_txt(path):
    return [Category(i,name,prompts) for i,(name,prompts) in enumerate(parse_class_file(Path(path)),1)]


def validate_names(names):
    names={int(cid):str(name) for cid,name in names.items()}
    if sorted(names)!=list(range(len(names))) or len(names)<2 or len(names)>255:
        raise ValueError('Saved taxonomy must contain contiguous IDs0..N, with1..254 semantic classes')
    if names[0] not in ('unassigned','ignore','unlabeled'):raise ValueError('Class0 must be unassigned/ignore')
    if len(set(names.values()))!=len(names):raise ValueError('Class names must be unique')
    return names


def taxonomy_from_categories(categories):
    return validate_names({0:'unassigned',**{category.id:category.name for category in categories}})


def normalize_colors(colors, names=None):
    """Validate internal-name overrides and normalize #RRGGBB to #rrggbb."""
    if not isinstance(colors,dict) or any(
            not isinstance(name,str) or not name.strip() or
            not isinstance(color,str) or re.fullmatch(r'#[0-9a-fA-F]{6}',color) is None
            for name,color in colors.items()):
        raise ValueError('Colors must map internal class names to #rrggbb strings')
    if names is not None:
        unknown=set(colors)-set(names.values())
        if unknown:raise ValueError(f'Unknown color class names: {sorted(unknown)}')
    return {name:color.lower() for name,color in colors.items()}


def load_colors(classes_file):
    """Load optional sibling color overrides; absent sidecars preserve defaults."""
    path=Path(classes_file).with_suffix('.colors.json')
    return normalize_colors(json.loads(path.read_text(encoding='utf-8'))) if path.is_file() else {}


def palette_for(names,colors=None):
    overrides=normalize_colors({} if colors is None else colors,names)
    palette=make_palette()
    for cid,name in names.items():
        if name in COLORS_BY_NAME:palette[cid]=COLORS_BY_NAME[name]
        if name in overrides:
            color=overrides[name]
            palette[cid]=tuple(int(color[i:i+2],16) for i in (1,3,5))
    return palette


def _display_names(mapping):
    if not isinstance(mapping,dict) or any(not isinstance(name,str) or not isinstance(label,str) or not label.strip()
                                          for name,label in mapping.items()):
        raise ValueError('Display names must map internal class names to nonempty strings')
    return dict(mapping)


def load_display_names(classes_file):
    """Load optional sibling labels without changing class names, prompts or IDs."""
    path=Path(classes_file).with_suffix('.display_names.json')
    return _display_names(json.loads(path.read_text(encoding='utf-8'))) if path.is_file() else {}


def saved_display_names(directory):
    """Recover labels from the same saved taxonomy selected by saved_schema."""
    directory=Path(directory)
    if directory.is_file():directory=directory.parent
    for candidate in (directory,directory.parent,directory.parent.parent):
        path=candidate/'classes.json'
        if path.is_file():
            raw=json.loads(path.read_text(encoding='utf-8'))
            return _display_names({row['name']:row['display_name'] for row in raw
                                   if row.get('display_name') is not None}) if isinstance(raw,list) else {}
        config_path=candidate/'run_config.json'
        if config_path.is_file():
            config=json.loads(config_path.read_text(encoding='utf-8'))
            if isinstance(config.get('taxonomy'),dict):return {}
    return {}


def saved_colors(directory):
    """Recover RGB colors from the same nearest taxonomy selected by saved_schema."""
    directory=Path(directory)
    if directory.is_file():directory=directory.parent
    for candidate in (directory,directory.parent,directory.parent.parent):
        path=candidate/'classes.json'
        if path.is_file():
            raw=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(raw,list):return {}
            colors={}
            for row in raw:
                rgb=row.get('color_rgb')
                if rgb is None:continue
                if not isinstance(rgb,list) or len(rgb)!=3 or any(type(v) is not int or not 0<=v<=255 for v in rgb):
                    raise ValueError('Saved color_rgb must contain three integer channels in [0,255]')
                colors[row['name']]='#'+''.join(f'{v:02x}' for v in rgb)
            return normalize_colors(colors)
        config_path=candidate/'run_config.json'
        if config_path.is_file():
            config=json.loads(config_path.read_text(encoding='utf-8'))
            if isinstance(config.get('taxonomy'),dict):return {}
    return {}


def class_rows(names,display_names=None,colors=None):
    palette=palette_for(names,colors);display_names=display_names or {}
    return [{'id':cid,'name':name,'label':display_names.get(name,f'{ZH.get(name,name)} / {name}'),
             **({'display_name':display_names[name]} if name in display_names else {}),
             'color':palette[cid].tolist()} for cid,name in names.items()]


def resolve_planes(names,requested=None,default_ids=None):
    names=validate_names(names)
    if requested is None:
        if default_ids is not None:
            ids=tuple(sorted(set(int(cid) for cid in default_ids)))
            if not ids or any(cid not in names or cid==0 for cid in ids):raise ValueError('Saved plane IDs are invalid for the saved taxonomy')
            return ids
        requested=[name for name in DEFAULT_PLANES if name in names.values()]
    if not requested:raise ValueError('Select at least one --plane-classes name')
    reverse={name:cid for cid,name in names.items() if cid}
    unknown=set(requested)-reverse.keys()
    if unknown:raise ValueError(f'Unknown plane classes: {sorted(unknown)}; available: {list(reverse)}')
    return tuple(sorted({reverse[name] for name in requested}))


def saved_schema(directory,allow_legacy=False):
    """Read the saved taxonomy; never interpret old IDs using today's prompt file."""
    directory=Path(directory)
    if directory.is_file():directory=directory.parent
    for candidate in (directory,directory.parent,directory.parent.parent):
        path=candidate/'classes.json'
        config_path=candidate/'run_config.json'
        if path.is_file():
            raw=json.loads(path.read_text(encoding='utf-8'))
            if isinstance(raw,list):names={int(row['id']):row['name'] for row in raw}
            elif isinstance(raw,dict):names={int(cid):name for cid,name in raw.items()}
            else:raise ValueError('Invalid classes.json')
            config=json.loads(config_path.read_text()) if config_path.is_file() else {}
            return validate_names(names),config.get('plane_class_ids')
        if config_path.is_file():
            config=json.loads(config_path.read_text())
            if isinstance(config.get('taxonomy'),dict):return validate_names(config['taxonomy']),config.get('plane_class_ids')
    if allow_legacy:return dict(LEGACY_NAMES),None
    raise ValueError(f'No saved taxonomy found near {directory}; provide the source classes.json to disambiguate IDs')
