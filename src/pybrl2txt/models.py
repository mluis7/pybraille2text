'''
Created on Dec 31, 2024

@author: lmc
'''
from dataclasses import dataclass


@dataclass
class Area(object):
    """Area coordinates of a Page or a Line object."""
    xmin: int = 0
    xmax: int = 0
    ymin: int = 0
    ymax: int = 0

@dataclass
class CellParams(object):
    """Cell dimension parameters.
    xdot: x distance between cell dots
    ydot: y distance between cell dots
    xsep: x separation between cells
    csize: total cell x size.
    See: get_area_parameters() on how they are calculated.
    
    |<-- csize  -->|
    |<xdot>|
    *      *        *      * -
    *      *        *      * - ydot
    *      *        *      *
    """
    xdot = 0
    xsep = 0
    csize = 0
    ydot = 0
    normalized = True
    
@dataclass
class Page(Area):
    '''
    Page attributes
    '''
    cell_params  = CellParams()
    lines = []
    lang = 'en'

@dataclass
class Line(Area):
    # Get y possible values if parent object is a Line.
    # A Line should have only 3 possible values.
    # Name suffix represents cell braille indexes, e.g. 14 means cell 1 or 4
    ydot14 = 0
    ydot25 = 0
    ydot36 = 0
    # Useful for debugging
    line_num = -1
    cell_count = 0
    # CellParams
    cell_params  = CellParams()
    
    def __repr__(self):
        return f"Line: {self.line_num}, ydot14: {self.ydot14}, ydot25: {self.ydot25}, ydot36: {self.ydot36}"
    def __str__(self):
        return self.__repr__()

    