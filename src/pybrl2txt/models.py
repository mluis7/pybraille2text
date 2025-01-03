'''
Created on Dec 31, 2024

@author: lmc
'''
from dataclasses import dataclass


@dataclass
class Area(object):
    xmin: int = 0
    xmax: int = 0
    ymin: int = 0
    ymax: int = 0

@dataclass
class CellParams(object):
    xdot = 0
    xsep = 0
    csize = 0
    ydot = 0
    
@dataclass
class Page(Area):
    '''
    Page attributes
    '''
    cell_params  = CellParams()
    lines = []

@dataclass
class Line(Area):
    # Get y possible values if parent object is a Line.
    # A Line should have only 3 possible values.
    # Name suffix represents cell braille indexes
    ydot14 = 0
    ydot25 = 0
    ydot36 = 0
    # Useful for debugging
    line_num = -1
    # CellParams
    cell_params  = CellParams()

    