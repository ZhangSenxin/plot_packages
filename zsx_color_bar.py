#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# cmod=1 color name, cmod=2 colormap name, cmod=3 colorlist
def colorbar(plt_, color_, cmod=2, height=1, width=10, x0=0, y0=0, cr=True, density=100, outer=0.5,
             ticks=None, labels=None, fs=12, fc='black', rotation=0, lc=None, lw=1, zorder=10):
    
    if cmod == 1:
        colors_ = [color_] * density
        alpha = [d_ / (density - 1) for d_ in range(density)]
        
    elif cmod == 2:
        cmap_ = cm.get_cmap(color_)
        colors_ = [cmap_(d_ / (density - 1)) if d_ < (density - 1) else cmap_(1 - 0.01 / (density - 1))
                   for d_ in range(density)]
        alpha = [None] * density
        
    elif cmod == 3:
        if (type(color_) == list) or (type(color_) == np.ndarray):
            density = len(color_)
            colors_ = color_
            alpha = [None] * density
        else:
            colors_ = [color_] * density
            alpha = [d_ / (density - 1) for d_ in range(density)]
    
    if cr:
        for i_ in range(density):
            plt_.fill_between([(i_ / density) * width + x0, ((i_ + 1) / density) * width + x0], height + y0, 0 + y0,
                              color=colors_[i_], alpha=alpha[i_], zorder=zorder)
        if ticks is not None:
            if not labels:
                labels = ticks
            for t_, l_ in zip(ticks, labels):
                plt_.text(t_ * width + x0, height * (1 + outer) + y0, l_, fontsize=fs, color=fc,
                          rotation=rotation, verticalalignment="bottom", horizontalalignment="center", zorder=zorder)
            
    else:
        for i_ in range(density):
            plt_.fill_betweenx([(i_ / density) * height + y0, ((i_ + 1) / density) * height + y0], width + x0, 0 + x0,
                               color=colors_[i_], alpha=alpha[i_], zorder=zorder)
        if ticks is not None:
            if not labels:
                labels = ticks
            for t_, l_ in zip(ticks, labels):
                plt_.text(width * (1 + outer) + x0, t_ * height + y0, l_, fontsize=fs, color=fc,
                          rotation=rotation, verticalalignment="center", horizontalalignment="left", zorder=zorder)
    
    if lc is not None:
        plt_.plot([x0, x0 + width, x0 + width, x0, x0], [y0, y0, y0 + height, y0 + height, y0],
                  lw=lw, c=lc, zorder=zorder + 1)
