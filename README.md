# pybraille2text
Translate braille images to text using pyhon-opencv to detect blobs.  
Status: Alpha/PoC

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Image considerations](#image-considerations)
- [Braille considerations](#braille-considerations)
- [Translation strategy](#translation-strategy)
- [Group coordinates by line](#group-coordinates-by-line)
- [Find cells representing a character](#find-cells-representing-a-character)
- [Translate cell coordinates to Braille character indexes](#translate-cell-coordinates-to-braille-character-indexes)
- [Translate dot indexes to text](#translate-dot-indexes-to-text)
- [Tunning suggestions](#tunning-suggestions)
- [Compiling liblouis20](#compiling-liblouis20)

<!-- TOC end -->

## Image considerations
Blob detection is very sensitive to image quality so currently it's tuned for the following specs:

- Width: around 1100 px
- Must be a grayscale image
- white margin around dots area has an impact and should be at least 1.5 times the width/height of a character (empirical).
- If small dots are detected then the contrast should be increased.

Tuning the detection parameters can be done by setting `cv2_cfg.detect.show_detect: true` in the configuration file.
Detected dots in a line should all have the same color. Black dots without colored countour or small dots with color mean blob 
detection errors that will lead to traslation errors. Some times a lot of them so it's important.

![X difference between points](/src/resources/line-detection.png)

## Braille considerations
It aims to translate Unified English Braille (Grade 2). Testing samples have been generated using [ABC Braille site](https://abcbraille.com/text) 
for now but that text to Braille translation is also automated so it may contain inaccuracies leading to wrong detected text.
  
Sample images of real Braille pages will be mostly welcomed. :-)

## Translation strategy
In general terms

```None
Braille image --> blob coordinates --> Braille cell indexes --> Unicode Braille --> text
```

- Get a sorted numpy array of blob coordinates from opencv detected keypoints.
- Group coordinates by line.
- Find cells representing a character
- Translate cell coordinates to Braille character indexes (1,2,3 for the first column, 4,5,6 for the second).
- Translate dot indexes to Unicode Braille characters.
- Translate Unicode Braille to English text with `liblouis20`.

Unicode Braille character names are of the form `BRAILLE PATTERN DOTS-<cell indexes>`  
e.g.: `BRAILLE PATTERN DOTS-1236  --> letter 'v'`

## Group coordinates by line
Lines are detected using the coordinates differences between contiguous coordinates.  
Negative values are the key ones since they represent a row change (green line) or the start of a line (red line).

e.g.: previous point `p[i-1]=(520,69)`, current point `p[i] =(69, 140)` -> `xydiff = (-451, 71).`

```
if (x-diff is negative) and (y-diff is greater than vertical cell size):
    "current dot is starting a line"
```
    
![X difference between points](/src/resources/kp-differences.png)

## Find cells representing a character
Cell dimensions are extracted from calculated differences too. It's a key step.  
Values are stored on Page, Line data classes and are re-calculated for each line.  
See: `pybrl2txt.braille_to_text.get_area_parameters`

Errors may impact just the corresponding word or the whole line.

```None
WARNING [ pybrl2txt ] WARN. cell_idx not found. Line: 0, ydot14: 35.0, ydot25: 50.0, ydot36: 66.0, idx: 24, cell_middle: 954, kp: [954.  66.]
ERROR   [ pybrl2txt ] index translation error
[[954.  35.]
 [954.  50.]
 [954.  66.]] -> (-1, -1, -1)
 ```

## Translate cell coordinates to Braille character indexes
Translate cell coordinates to dot indexes tuples like `((4, 5, 6), (3, 4, 6),)`.
Those tuples are the keys or the values on `pybrl2txt.braille_maps` dictionaries.

There are 2 available methods for translation:
- Cell normalization (current). See: `pybrl2txt.braille_to_text.cell_keypoints_to_braille_indexes` docstrings.
- Cell coordinates computation: See: `pybrl2txt.braille_to_text.cell_to_braille_indexes_no_magic` docstrings.

## Translate dot indexes to text
Probably the toughest part because Braille reading rules must be applied.
It was decided not to write a new algorithm but to use `liblouis20` library instead
since it provides python bindings.
The result is very promising for English language.  
Spanish was tested but translation contained errors (~ 80% successful).
See texts and images for those tests in `tests/resources`.
Language selection is done with `parse.lang` configuration property

## Tunning suggestions

- Start with simple Braille images having more than one line.
- If images are digitally generated with online tools, save also the text.
- Have a different yaml configuration file for each image or type of images.
- If you see too many errors then blob detection probably needs tuning. Check found blob areas  
`INFO    [ pybrl2txt ] keypoint sizes [10 11]`  
If values are not close to each other then blob detection needs tuning, e.g.: `[4, 10, 11]`.

## Compiling liblouis20
Some Linux distributions provide `python-louis` packages for some python versions but not others so it might have to be manually built.

- Install `liblouis20` at OS level since tables are needed  
e.g.:
`zypper install liblouis liblouis-data`

`sudo apt install liblouis liblouis-data`

### Building python package
Download [liblouis20 sources](https://liblouis.io/downloads/)

Build commands. It worked out of the box in my case

```bash
cd /path/to/<liblouis source>
automake
./configure
make

# build python package for your preferred python version
cd python
python3.11 -m build .

# install package
pip3.11 install dist/louis-3.32.0-py3-none-any.whl
```

