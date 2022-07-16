# Image Processing

Four demos of image processing concepts.

1. Image Enhancement

2. Pupil Detection

3. Staff Line Detection

4. Filter Demonstrations


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

## 3. Filter Demonstrations

### Goal:


### Results:


### Conclusions:

## 4. Filter Demonstrations

### Goal:

Demonstrate the filters most commonly used in image processing. Use masking for speed whenever possible.

### Results:


### Conclusions:

I chose this image beacuse it has many different textures and shapes, with lines, dots, curves and gradients of various 
sizes. It ended up being an excellent choice, really showing what each filter affects most.

For example, the average filter created a very consistent blurriness across the image, but it removed important details 
indiscriminately. The Gaussian filter was more effective at adding blur without dramatically impacting important smaller 
features. In the larger sizes this was ever more noticeable. It’s hard to describe but the larger Gaussian kernels seemed 
more “natural” than the larger average filters. The median filter did not really add blur so much as it removed intense 
spots like in the starry texture on my hoodie. It did add some blur here and there as a side-effect (like in my hair) so I 
would want to use it carefully. Confining the filter to a certain region like the hoodie would do the trick.

As for edge detection there were minor but interesting differences between each. The Sobel filter gradient stood out as quite 
accurate. It captured many fine lines but still responded well to the larger ones. However, the vertical filter allowed 
horizontal information in and the horizontal filter let in vertical information. That I attribute to the 1 values in the 
corners of the 3x3 kernel. That diagonal consideration may be good to produce an accurate gradient, but it would be less 
useful for specifically picking out only horizontal or vertical change. For that the central difference kernel was 
surprisingly useful, though it was not very responsive. The Prewitt kernel was also useful for showing purely horizontal or 
vertical information, though more “bleed” from one orientation to another was still clearly present.

Out of the two Laplacian filters, I would choose the one with all 1s in the outer cells of the kernel. It was much more 
responsive and showed diagonal edges much better. The one filter that truly baffled me was the Laplacian of Gaussian. It did 
not produce very results. The smaller kernels left quite a lot of noise, so it would be hard to use it for edge detection in 
this case. The larger kernels seemed to “sort out” the less intense edges better, but the edges interfered with each other. 
Maybe it would be better used on an image with larger, more defined features.
