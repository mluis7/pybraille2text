# pybraille2text
Translate braille images to text uning pyhon-opencv to detect blobs.

## Image considerations
Blob detection is very sensitive to image quality so currently its tuned for the following specs:

- Width: around 1100 px
- Must be a grayscale image
- If small dots are detected then the contrast should be increased.

![X difference between points](/src/resources/kp-differences.png)
