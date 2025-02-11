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
    csize: total cell width (xdot + xsep).
    dot_size_spec: float 
        Expected xdot value for current page dimensions and BANA specifications.
        Calculated as
        ((xmax-xmin)/(max cell spec: 40)/(max spec cell size/max spec dot size)
        See: pybrl2txt.braille_maps.dots_rels
     
    See: get_area_parameters() on how they are calculated.
    
    |<-- csize  -->|
    |<xdot>|<xsep >|
    *      *        *      * -
    *      *        *      * - ydot
    *      *        *      *
    
    """
    xdot = 0
    xsep = 0
    csize = 0
    ydot = 0
    blob_sizes = None
    normalized = True
    # Expected cell size according to page size and specifications
    # Calculated as
    # ((xmax-xmin)/(max cell spec: 40)/(max spec cell size/max spec dot size)
    # See: pybrl2txt.braille_maps.dots_rels
    dot_size_spec = 0
    
    def __repr__(self):
        return f"CellParams: xdot: {self.xdot}, ydot: {self.ydot}, xsep: {self.xsep}, csize: {self.csize:.3f}, csize/xdot: {self.csize/self.xdot:3f}"
    def __str__(self):
        return self.__repr__()
    
@dataclass
class Page(Area):
    '''
    Page attributes
    '''
    cp  = CellParams()
    lines_params = {}
    ref_cells = []
    ref_dims = {
        'xdot': [],
        'xsep': [],
        'csize': []}
    lang = 'en'

@dataclass
class Line(Area):
    # Dots vertical separation possible values if parent object is a Line.
    # A Line should have only 3 possible values.
    # Name suffix represents cell braille indexes, e.g. 14 means cell 1 or 4
    ydot14 = 0
    ydot25 = 0
    ydot36 = 0
    # Useful for debugging
    line_num = -1
    cell_count = 0
    word_count = 0
    # CellParams
    cp  = CellParams()
    
    def __repr__(self):
        return f"Ln: {self.line_num:>2}, p0: ({self.xmin}, {self.ymin:4.1f}), p_nth: ({self.xmax:5.1f}, {self.ymax:4.1f}), csize: {self.cp.csize:.2f}, xsep/xdot: {self.cp.xsep/self.cp.xdot:.2f}, csize/xdot: {self.cp.csize/self.cp.xdot:.2f}, blob_sizes: {self.cp.blob_sizes}"
    def __str__(self):
        return self.__repr__()

    