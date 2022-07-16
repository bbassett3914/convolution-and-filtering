# Image Processing

Three demos of image processing concepts.

## 1. Image Enhancement

### Goal:

Given an image of a climbing wall with several color-coded paths to the top, identify three individual paths. Then enhance 
the image so that those paths stand out from the background and are easier to identify visually.

### Results:

Input Image:

<img alt="Input Image" width="500" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Image-Enhancement/test.jpg"/>

Output Image:

<img alt="Final Result" width="500" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Image-Enhancement/output/test_final_result.jpg"/>

### Report:

I chose to convert to HSV because it puts chromaticity on a one-dimensional scale. At first I tried to work in 
CIE LUV because it is more accurate to human perception. However the two-dimensional chromaticity representation proved too 
complex for identifying specific colors. Yet when I started working in HSV I had all sorts of issues getting consistent 
results. After several hours of struggle it became clear that in OpenCV, HSV works on a different scale from the HTML color 
pickers I was using as a reference. Creating conversion functions solved this issue and meant I didn't have to calculate all 
of the values myself. That way I could refer to a color slider to get a general idea of what parameters to use, then make 
minor adjustments from there.

The second challenge was trying to get masking to work. For some reason my masks kept producing unpredictable results, even 
when I tried using very simple parameters. In RGB I had no issue, but in HSV I could not achieve any useful results. This 
was very frustrating as it forced me to depend on for loops too much. They are computationally complex and take a very long 
time to complete in comparison to masks, but in this assignment I felt I had no choice. Ultimately I was able to write the 
program so it would not rely on too many for loops, and it took only a few minutes to execute.

For the final result, I started by desaturating the original image. This created a grayscale image to use as a background so 
the bright colors of the routes would really stand out. After identifying the desired parameters for each color pathway, I 
used morphological opening and closing to remove small artifacts and make the holds into easily identifiable shapes. Then I 
changed each of the identified routes into solid colors to make them even more recognizable. After that I used a convex hull 
algorithm to outline the boundary of each of the holds. Finally I layered the final result from each identified pathway onto 
the grayscale background.
