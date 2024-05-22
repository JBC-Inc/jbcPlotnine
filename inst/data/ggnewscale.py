from plotnine import *

# SCALES ======================================================================

def standardise_aes_names(x):
  import re
  # convert US to UK spelling of colour
  x = x.replace("color", "colour")

  # convert old-style aesthetics names to ggplot version
  base_to_ggplot = {
      # Add the appropriate mappings here
      # Example: 'old_name': 'new_name'
  }
  x = base_to_ggplot.get(x, x)
  return x

def new_scale(new_aes):
  return standardise_aes_names(new_aes)

def new_scale_fill():
  return new_scale("fill")

def new_scale_color():
  return new_scale("colour")

def new_scale_colour():
  return new_scale("colour")

# HELPER ======================================================================

def remove_new(aes):
  import re
  return re.sub(r"(_new)*", "", aes, flags=re.IGNORECASE)

def change_name(obj, old, new):
    return obj.__class__.__name__.__class__.__call__(obj, old, new)

def change_name_character(lst, old, new):
    import copy
    lst_copy = copy.deepcopy(lst)
    lst_copy = [new if x in old else x for x in lst_copy]
    return lst_copy

def change_name_default(lst, old, new):
    names = list(lst.keys())
    names = [new if name in old else name for name in names]
    lst = dict(zip(names, lst.values()))
    return lst

def change_name_null(lst, old, new):
    return None

# BUMP_AES_GUIDES =============================================================

def bump_aes_guides(guides, new_aes):
    original_aes = new_aes

    if isinstance(guides, dict) and "guides" in guides:
        to_change = [remove_new([name])[0] == original_aes for name in guides["guides"].keys()]

        if any(to_change):
            guides["guides"] = {f"{name}_new" if change else name: value 
                                for name, value, change in zip(guides["guides"].keys(), guides["guides"].values(), to_change)}
    else:
        to_change = [remove_new([name])[0] == original_aes for name in guides.keys()]

        if any(to_change):
            guides = {f"{name}_new" if change else name: value 
                      for name, value, change in zip(guides.keys(), guides.values(), to_change)}

    return guides

# BUMP_AES_LABELS =============================================================

def bump_aes_labels(labels, new_aes):

    old_aes = [key for key in labels.keys() if remove_new([key])[0] in new_aes]
    new_aes = [f"{aes}_new" for aes in old_aes]

    for old, new in zip(old_aes, new_aes):
        labels[new] = labels.pop(old)
    
    return labels

# BUMP_AES_LAYER ==============================================================

def bump_aes_layer(layer, new_aes):
    original_aes = new_aes

    new_layer = layer.copy()

    old_aes = [key for key in new_layer.mapping.keys() if remove_new([key])[0] in new_aes]

    if not old_aes:
        old_aes = [key for key in new_layer.stat.default_aes.keys() if remove_new([key])[0] in new_aes]
        if not old_aes:
            old_aes = [key for key in new_layer.geom.default_aes.keys() if remove_new([key])[0] in new_aes]

    if not old_aes:
        return new_layer

    new_aes = [f"{ae}_new" for ae in old_aes]

    old_geom = new_layer.geom
    old_handle_na = old_geom.handle_na
    
# BUMP_AES_LAYERS =============================================================

def bump_aes_layers(layers, new_aes):
    return [bump_aes_layer(layer, new_aes) for layer in layers]
  
# BUMP_AES_SCALE ==============================================================

def bump_aes_scale(scale, new_aes):

    old_aes = [aes for aes in scale['aesthetics'] if aes in new_aes and aes not in remove_new(scale['aesthetics'])]
    if len(old_aes) != 0:
        new_aes = [f"{aes}_new" for aes in old_aes]

        scale['aesthetics'] = [new if aes in old_aes else aes for aes in scale['aesthetics']]

        if isinstance(scale['guide'], str):
            no_guide = scale['guide'] == "none"
        else:
            no_guide = not scale['guide'] or isinstance(scale['guide'], (GuideNone,))

        if not no_guide:
            if isinstance(scale['guide'], str):
                scale['guide'] = globals()[f"guide_{scale['guide']}"]()
            if isinstance(scale['guide'], Guide):
                old = scale['guide']
                new = ggplot2.ggproto(None, old)

                new.available_aes = change_name(new.available_aes, old_aes, new_aes)
                new.available_aes = [new if aes in old_aes else aes for aes in new.available_aes]

                if new.params.get('override.aes') is not None:
                    new.params['override.aes'] = change_name(new.params['override.aes'], old_aes, new_aes)

                scale['guide'] = new
            else:
                scale['guide'].available_aes = [new if aes in old_aes else aes for aes in scale['guide'].available_aes]

                if scale['guide'].get('override.aes') is not None:
                    scale['guide']['override.aes'] = {new if k == old_aes else k: v for k, v in scale['guide']['override.aes'].items()}

    return scale

# BUMP_AES_SCALES =============================================================

def bump_aes_scales(scales, new_aes):
    return [bump_aes_scale(scale, new_aes=new_aes) for scale in scales]

# GGPLOT_ADD_NEW_AES ==========================================================

def ggplot_add_new_aes(object, plot, object_name):
    if object not in plot['scales']:
        plot['scales'] = build_plot_scales(plot)
    
    # Global aes
    old_aes = [key for key in plot['mapping'] if remove_new(key) in object]
    new_aes = [f"{aes}_new" for aes in old_aes]
    
    for old, new in zip(old_aes, new_aes):
        plot['mapping'][new] = plot['mapping'].pop(old)
    
    plot['layers'] = bump_aes_layers(plot['layers'], new_aes=object)
    plot['scales']['scales'] = bump_aes_scales(plot['scales']['scales'], new_aes=object)
    plot['labels'] = bump_aes_labels(plot['labels'], new_aes=object)
    plot['guides'] = bump_aes_guides(plot['guides'], new_aes=object)
    
    return plot

























