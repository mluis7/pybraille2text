'''
Created on Dec 31, 2024

@author: lmc
'''

import sys
import yaml
from collections import deque
import cv2
import numpy as np
from pybrl2txt.models import Page, Line, CellParams, Area
from pybrl2txt.braille_maps import BLANK, abbr, cell_map, numbers, prefixes, rev_abbr

#abbr = { tuple([t for t in v]) : k for k,v in rev_abbr.items()}
#abbr = dict(map(reversed, rev_abbr.items()))

def get_build_detector_params(cv2_params):
    min_area = cv2_params['cv2_cfg']['detect']['min_area']
    max_area = cv2_params['cv2_cfg']['detect']['max_area']
    min_inertia_ratio = cv2_params['cv2_cfg']['detect']['min_inertia_ratio']
    min_circularity = cv2_params['cv2_cfg']['detect'].get('min_circularity', 0.9)
    min_convexity = cv2_params['cv2_cfg']['detect'].get('min_circularity', 0.75)
    # Set up SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
# Filter by area (size of the blob)
    params.filterByArea = True
    params.minArea = min_area # 10 Adjust based on dot size
    params.maxArea = max_area # 1000 Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = min_circularity # Adjust for shape of the dots
# Filter by convexity
    params.filterByConvexity = False
    params.minConvexity = min_convexity
# Filter by inertia (roundness)
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia_ratio
    #params.minDistBetweenBlobs = 12
    return params

def show_detection(image, detected_lines, xcell, xsep, xmin, ymax):
    """Help to visually debug if lines are correctly detected since dots would be colored by line.
    Black dots represent not correctly detected cells/lines.
    Color will repeat every for lines."""
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (178,102,255)]
    while len(colors) < len(detected_lines):
        colors.extend(colors)
    # Draw detected blobs as red circles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
#    x = int(xmin)
#    for i in range(1,7):
#        output_image = cv2.line(image, ( x * i + int(xsep), 50), (x * i + int(xsep), ymax), (0, 255, 0), thickness=2)

    for i, line in enumerate(detected_lines):
        output_image = cv2.drawKeypoints(output_image, line, np.array([]), colors[i], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print("Showing detection result")
    cv2.imshow("detected", output_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 

def group_by_lines(kp_map, blob_coords, xydiff, page_params):
    """Group coordinates by lines."""
    ycell = page_params.cell_params.ydot
    lines_coord = []
    detected_lines = [[kp_map[blob_coords[0][0], blob_coords[0][1]]]]
    print(f"new line {'0':>2} at: {int(blob_coords[0][0])},{int(blob_coords[0][1])}")
    # split coordinates by lines
    line_cnt = 1
    for i, d in enumerate(xydiff):
        curr_pt = blob_coords[i + 1]
        prev_pt = blob_coords[i]
        if curr_pt[0] >= 714 and curr_pt[1] >= 820 and curr_pt[1] < 851:
            pass
    #print(d, curr_pt, blob_coords[i+1], f"xdiff {d}, ydiff: {blob_coords[i+1][1] - blob_coords[i][1]}")
        # FIXME: line detect because a word may be split in 2 lines
        current_keypoint = kp_map[curr_pt[0], curr_pt[1]]
        #if (d[0] < page_params.cell_params.csize * -1 and d[1] >= ycell * 2):
        #727, 823
        #print(f"new line {line_cnt:>2} at: {curr_pt}, curr xdiff: {np.round(d)}, {page_params.xmax * -0.3:.0f}, previous: {blob_coords[i]}")

        #if (d[0] < 0 and d[1] >= ycell * 2.5) or (d[0] < 0 and prev_diff[1] >= ycell * 1.8):
        if (d[0] < 0 and d[1] >= ycell * 2.5) :# or (curr_pt[0] < prev_pt[0] and curr_pt[1] > prev_pt[1]):
            print(f"new line {line_cnt:>2} at: {curr_pt}, curr xdiff: {np.round(d)}, {page_params.xmax * -0.3:.0f}, previous: {blob_coords[i]}")
            detected_lines.append([current_keypoint])
            line_cnt += 1
        else:
            detected_lines[-1].append(current_keypoint)
    p0 = 0
    p1 = len(detected_lines[0])
    for j, lc in enumerate(detected_lines):
        if j > 0:
            p0 = p0 + len(detected_lines[j - 1])
        p1 = p0 + len(detected_lines[j])
        lines_coord.append(blob_coords[p0:p1])
        print(f"Line: {j + 1}; points: {len(detected_lines[j])}, last: {blob_coords[p0:p1].max()}")
    
    return detected_lines, lines_coord

def get_area_parameters(coords, area_obj: Area):
    """Parameters to help find cells from detected coordinates.
    """
    # x,y differences between contiguous dots. Negative values mean second/third rows in a cell and the start of a line.
    # e.g.: previous point p0=(520,69), current point =(69, 140). xydiff = (-451, 71).
    # xdiff is negative, ydiff is greater than vertical cell size --> current dot is starting a line.
    xydiff = np.array([ (kp[0] - coords[i-1][0], kp[1] - coords[i-1][1]) for i,kp in enumerate(coords) if i > 0 ])
    
    xcoords = coords[coords[:,0] > 1][:,0]
    ycoords = coords[coords[:,1] > 1][:,1]
    # minimum x in the whole image
    xmin = np.array(xcoords).min()
    # max x in the whole image. Represents last dot in a line.
    xmax = np.array(xcoords).max()
    # minimum y in the whole image
    ymin = np.array(ycoords).min()
    ymax = np.array(ycoords).max()
    
    area_obj.xmin = xmin
    area_obj.xmax = xmax
    area_obj.ymin = ymin
    area_obj.ymax = ymax
    
    xuniq = np.unique(np.round(xydiff[(xydiff[:,0] > 1)][:,0]))
    # x separation between dots in a cell
    xcell = xuniq.min()
    # y separation between dots in a cell
    yuniq = np.unique(np.round(xydiff[(xydiff[:,1] > 1)][:,1]))
    ycell = yuniq.min()
    # x separation between cells
    xsep = np.unique(xydiff[(xydiff[:,0] > xcell + 1)][:,0]).min()
    
    area_obj.cell_params.xdot = xcell
    area_obj.cell_params.ydot = ycell
    area_obj.cell_params.xsep = xsep
    area_obj.cell_params.csize = round(xcell + xsep)

    return area_obj, xydiff

def get_keypoints(img_path, cv2_params):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    params = get_build_detector_params(cv2_params)
# Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params) # Detect blobs
    keypoints = detector.detect(image)
    return keypoints, image

def coordinate_to_braille_indexes(cell, line_params, idx):
    """Return a sorted tuple representing dot indexes in the cell.
    The tuple should map to a text character in cell_map dict.
    Indexes are
    1 4
    2 5
    3 6
    
    Cell    Indexes       Text
    .
    .
    . . --> (1,2,3,6) --> 'v'
    
    """
    is_cell_error = False
    cell_start = get_cell_start(line_params, idx)
    #print(f"idx: {idx}, cell_start: {cell_start:.0f}, len: {len(cell)}")
    y_all_line = [round(line_params.ymin), round(line_params.ymin + line_params.cell_params.ydot * 1.20), round(line_params.ymin + line_params.cell_params.ydot * 2.3)]
    #y_all_line = [round(line_params.ymin), round(line_params.ymin + line_params.cell_params.ydot * 1.20), round(line_params.ymax + line_params.cell_params.ydot * 0.5)]
    cell_idxs = []
    cell_idx = -1
    # FIXME: detect x in first column. Cells with single dot may fail to detect 1 or 3 dot
    if idx == 0 and line_params.ymax > 800:
        pass
    for cc in cell:
        cell_middle = round(cell_start + line_params.cell_params.xdot * 0.8)
        if (cc[0] < cell_middle):
                if cc[1] <= y_all_line[0]:
                    cell_idx = 1
                elif cc[1] <= y_all_line[1] and cc[1] < y_all_line[2]:
                    cell_idx = 2
                elif cc[1] > y_all_line[1] and cc[1] <= y_all_line[2]:
                    cell_idx = 3
        elif cc[0] > cell_middle:
            if cc[1] <= y_all_line[0]:
                cell_idx = 4
            elif cc[1] <= y_all_line[1] and cc[1] < y_all_line[2]:
                cell_idx = 5
            elif cc[1] > y_all_line[1] + 1 and cc[1] <= y_all_line[2]:
                cell_idx = 6
        if cell_idx == -1:
            is_cell_error = True
            #print(f"WARNING. cell_idx not found: {cell_idx}, {cc}", "x_boundary:", cell_middle, "y_all_line", y_all_line, file=sys.stderr)
        elif cell_idx in cell_idxs:
            is_cell_error = True
            #print(f"WARNING. cell_idx duplicate: {cell_idx}, {cc}", "x_boundary:", cell_middle, "y_all_line", y_all_line, file=sys.stderr)
        # if len(cell) == 0 and cell_idx == 3:
        #     print("ERROR. First cell_idx can't be 3")
        cell_idxs.append(cell_idx)
        cell_idx = -1
    return tuple(sorted(cell_idxs)), is_cell_error

def translate_cell(cell, line_params, idx):
    """Translate cell coordinates to braille cell indexes."""
    
    if cell is BLANK:
        #print(f"SPACE: {idx}, {BLANK}")
        return BLANK
    brl_idx, is_cell_error = coordinate_to_braille_indexes(cell, line_params, idx)
#    for mm in [abbr, cell_map, numbers]:
#        if brl_idx in mm:
#            print(f"Cell: {idx}, {brl_idx}, {mm[brl_idx]}")
#            break
#    if brl_idx in prefixes:
#        print(f"prefix brl_idx: {prefixes[prefixes.index(brl_idx)]}")
    return brl_idx

def translate_cells(cells, line_params):
    word_tuples = []
    word_tpl = []
    for idx, cell in enumerate(cells):
        #print(cell)
        tcell = translate_cell(cell, line_params, idx)
    
        if tcell is not BLANK:
            word_tpl.append(tcell)
        else:
            word_tuples.append(tuple(word_tpl))
            word_tpl = []
    return word_tuples


def get_cell_start(line_params, idx):
    cell_start = line_params.xmin
    if idx == 1:
        cell_start += line_params.cell_params.csize
    elif idx > 1:
        cell_start += (line_params.cell_params.csize) * idx - line_params.cell_params.xdot * 0.3
    return cell_start

def translate_line(line_coor, ln, page):
    """Return array of cells in a line with BLANK inserted."""
    
    line_params, line_diff = get_area_parameters(line_coor, Line())
    if line_params.xmax < page.xmax:
            line_params.xmax = page.xmax
    if line_params.cell_params.csize > page.cell_params.csize:
        line_params.cell_params.csize = page.cell_params.csize
    
    cell_count_expected = int((line_params.xmax - line_params.xmin) / line_params.cell_params.csize)
    cells = []
    
    idx = 0
    cell_start = get_cell_start(line_params, idx)
    
    cell_end = line_params.xmin + line_params.cell_params.csize #  line_params.cell_params.xdot + line_params.cell_params.xsep / 2
    line_end = 0
    
    if ln == 2:
        pass
    
    while line_end <= line_params.xmax:
        #print("line, cell_start, cell_end", ln, cell_start, cell_end)
        # FIXME: cell_start, cell_end fail to find right cells
        cell = line_coor[(line_coor[:, 0] >= cell_start) & (line_coor[:, 0] <= cell_end)]
        if len(cell) == 0 or (len(cell) > 0 and cell[0][0] > cell_end):
            cells.append(BLANK)
            line_end += line_params.cell_params.csize
        else:
            #cell -= (line_params.xmin, line_params.ymin)
            cells.append(cell)
            line_end = cells[-1][0][0]
        idx += 1
        cell_start = get_cell_start(line_params, idx)
        cell_end += line_params.cell_params.csize
    
    #print(f"cell width: {line_params.cell_params.csize}, pt count: {len(line_coor)}, Cells found/expected: {len(cells)}/{cell_count_expected}")
    return cells, line_params


def translate_word_text_by_indexes(word_tuples, line_num):
    line_text = ''
    total_errors = 0
    for w, wrd in enumerate(word_tuples):
        if wrd in abbr:
            line_text += f"{abbr[wrd]} "
        elif len(wrd) >=1:
            for char in wrd:
                if char in prefixes:
                    line_text += f"~{''.join([str(n) for n in char])}"
                    continue
                else:
                    for schar_map in [numbers, cell_map]:
                        if char in schar_map:
                            line_text += cell_map[char]
                            break
            line_text += '_ '
        else:
            if wrd == ():
                continue
            else:
                print(f"line {line_num+1} word {w + 1} not found: {wrd}", file=sys.stderr)
                line_text += "XXXXX "
            total_errors += 1
    
    return line_text, total_errors

def main():
    base_dir = '/home/lmc/projects/eclipse-workspace/SOPython/lmc/braille_to_text_poc'
    cfg_path = '../resources/abbreviations.yml'
    #image_path = "braille-poem2.png"
    #image_path = "braille.jpg"
    #image_path = "technical.jpg"
    #image_path = "contracted_braille_example.webp"
    #image_path = "result2.brf.png"
    #image_path = "abbreviations_brl_abc.png"
    image_path = "abbreviations_brl_1line.png"
    
    grade = 2
    round_to = 2
    
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        grade = config['grade']
        round_to = config['parse']['round_to']
        show_detect = config['cv2_cfg']['detect']['show_detect']
    
    img_path = f"{base_dir}/{image_path}"
    keypoints, image = get_keypoints(img_path, config)
    # map of keypoints coordinates to keypoints
    kp_map = { (round(kp.pt[0], round_to), round(kp.pt[1], round_to)): kp for kp in keypoints}
    
    areas = set(round(kp.size) for kp in keypoints)
    # all dots coordinates, sorted to help find lines.
    blob_coords = np.array(list(kp_map.keys()))
    #blob_coords = np.array([kp.pt for kp in keypoints])
    blob_coords = blob_coords[np.lexsort((blob_coords[:,0], blob_coords[:,1]))]
    #blob_coords = np.round(blob_coords, decimals=2)
    
    page, xydiff = get_area_parameters(blob_coords, Page())
    cp = page.cell_params
    
    print(f"blob_coords: {len(blob_coords)}, xydiff: {len(xydiff)}")
    print(f"x params: xcell {cp.xdot :.0f}, xmin {page.xmin:.0f}, xsep {cp.xsep:.0f}, csize {cp.csize:.0f}, xmax {page.xmax:.0f}")
    print(f"y params: ycell {cp.ydot:.0f}, ymin {page.ymin:.0f}")
    print(f"areas {areas}")
    #print(f"max cells per line_params: {round((xmax)/(xcell + xsep))}")
    # List of list of cells by line_params
    detected_lines, lines_coord = group_by_lines(kp_map, blob_coords, xydiff, page)
   
    if show_detect: 
        show_detection(image, detected_lines, cp.xdot, cp.xsep, page.xmin, 400)
    
    text = ''
    total = 0
    total_errors = 0
    word_tuples = []
    for ln, line_coor in enumerate(lines_coord):
        #print(line_coor)
        cells, line_params = translate_line(line_coor, ln, page)
        cp = line_params.cell_params
        #print(f"x params: xdot {cp.xdot :.0f}, xsep {cp.xsep:.0f}, csize {cp.csize:.0f}, xmax {line_params.xmax:.0f}")
        #print(f"y params: ydot {cp.ydot:.0f}, ymin {line_params.ymin:.0f}")
        
        word_tuples.append((ln, translate_cells(cells, line_params)))
    
    for ln_wrd_tpl in word_tuples:
        print(f" ---------------> words in line {ln_wrd_tpl[0]}: {len(ln_wrd_tpl[1])}", file=sys.stderr)
        total += len(ln_wrd_tpl[1])
        wrd_text, wrd_errors = translate_word_text_by_indexes(ln_wrd_tpl[1], ln_wrd_tpl[1])
        text += f" {wrd_text}"
        total_errors += wrd_errors
                
        text += '\n'
    #print(text)
    found_count = 0
    not_found_count = 0
    not_found_text = ''
    found_text = ''
    for wr in [q for q in text.split(' ') if q not in ['', '\n']]:
        if wr in rev_abbr:
            found_text += wr + ' '
            found_count += 1
        else:
            not_found_text += wr + ' '
            not_found_count += 1
            
    print(total, total_errors)
    print(f"total translated words: {not_found_count + found_count}")
    print(f"total found: {found_count}\nnot_found_count: {not_found_count}\n")
    print(found_text)
    print(f'{"-" * 80}')
    print(not_found_text)
    print(f'{"-" * 80}')
    print(text)
if __name__ == '__main__':
    main()
    
