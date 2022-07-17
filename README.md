# Image Processing

Three demos of image processing concepts.

1. Image Enhancement

2. Pupil Detection

3. Staff Line Detection


## 1. Image Enhancement

### Goal:

Given an image of a climbing wall with several color-coded paths to the top, identify three individual paths. Then enhance 
the image so that those paths stand out from the background and are easier to identify visually.

### Results:

Input:

<img alt="Input Image" width="500" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Image-Enhancement/test.jpg"/>

Final Result:

<img alt="Final Result" width="500" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Image-Enhancement/output/test_final_result.jpg"/>

### Conclusions:

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

## 2. Pupil Detection

### Goal:

Identify the location of the center of a pupil given an image of an eye.

### Results:

Input:

<img alt="Input Image" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Pupil-Detection/iris.bmp"/>

Sobel Filter:

<img alt="Sobel Filter" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Pupil-Detection/output/iris_sobel_image_gradient.jpg"/>

Ring Convolution:

<img alt="Ring Convolution" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Pupil-Detection/output/iris_ring_convolution.jpg"/>

Final Result:

<img alt="Final Result" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Pupil-Detection/output/iris_final_result.jpg"/>


### Conclusions:

The iris image required a few steps to identify the pupil location, though it did not need averaging to produce good 
results. The details around the eye itself was well enough defined and there were practically no curved objects to interfere 
with the ring detection. Instead I started with the Sobel Gradient filter to find the edges of the image. Then I applied 
a 128x128 ring mask that contained 1s within a 35-45 pixel ring around the center, and 0s elsewhere. Convolving this with 
the Sobel-filtered image produced an image showing where ring-shaped elements might be within the image. Then I used 
thresholding to remove all but the very most important ring location information. That left me with a single dot depicting 
the exact center of the pupil within the original image.

## 3. Staff Line Detection

### Goal:

Convert a poor-quality image of sheet music into full binary black and white. Then isolate and identity the staff lines.

The goal is to create a foundation for converting an image of sheet music into MusicXML, “the standard open format for 
exchanging digital sheet music. Optical music recognition is fascinating to me, because of the incredible range of source 
materials that a OMR program must be able to decipher. Sheet music has been produced in various forms for at least 500 years. 
It is used for nearly every physical instrument in Western music. Many musicians to this day read from copies of handwritten 
manuscripts hundreds of years old. It is even common for musicians to play from copies of copies of copies, each adding 
unique noise. 

My inspiration came from a desire to make existing music more readable, by enhancing and converting it to a fully digital 
format. Most current sheet music software are essentially task-specific image editing programs, where jpeg images are written 
on with primitive digital drawing tools. How much easier would it be to read music if it could be consistently converted to 
digital format where notes and other musical “characters” are rendered in the way that formatted text files can be.

The full process from color image to MusicXML requires 4 major phases, each with its own unique problems. There is binary 
black and white conversion, where text and the music itself is made to stand out from the white background with the highest 
contrast possible. Then the staffs have to be identified and removed. This allows for individual notes and other musical 
symbols to be identified and categorized. Then finally the is reconstruction, where every relevant, recognized element is 
encoded into a digital markup language which the sheet music can later be reproduced from.

This project concerns itself with the binarization of the image as well as staff identification - the first two phases. Staff 
removal can be quite complex and falls outside the scope of this class. Symbol recognition is most successful when machine 
learning is applied, so I am saving that part for me to tackle next semester in that class. The reconstruction and 
reproduction of the sheet music is only possible after character recognition, so I will save that for some time in the 
future.


### Results:

Input:

<img alt="Input Image" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Staff-Line-Detection/test.jpg"/>

Thresholding by Steps:

<img alt="Thresholding by Steps" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Staff-Line-Detection/output/test_thresholding_by_steps.jpg"/>

Adaptive THresholding:

<img alt="Adaptive Thresholding" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Staff-Line-Detection/output/test_adaptive_thresholding_by_mean.jpg"/>

Staff Line Detection:

<img alt="Staff Line Detection" width="300" height="auto" 
src="https://github.com/brendan-bassett/Image-Processing/blob/main/Staff-Line-Detection/output/test_final_result.jpg"/>
