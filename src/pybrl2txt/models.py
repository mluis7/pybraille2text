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
        return f"CellParams: xdot: {self.xdot}, ydot: {self.ydot}, xsep: {self.xsep}, csize: {self.csize:.3f}, blob_sizes: {self.blob_sizes}"
    def __str__(self):
        return self.__repr__()
    
@dataclass
class Page(Area):
    '''
    Page attributes
    '''
    cell_params  = CellParams()
    lines_params = {}
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
        return f"Line: {self.line_num}, xmin/xmax: {self.xmin}/{self.xmax}, ymin/ymax: {self.ymin}/{self.ymax}, csize: {self.cell_params.csize:.2f}, blob_sizes: {self.cell_params.blob_sizes}"
    def __str__(self):
        return self.__repr__()

    