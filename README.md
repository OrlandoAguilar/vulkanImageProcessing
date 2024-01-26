This test is an example of using vulkan for image processing. It doesn't render anything to the screen, instead it uses several compute shader to perform different effects and saves the result to a png file.

The effects are executed by using shaders on the gpu. 

The effects included are:
* Laplacian
* Gray Scale
* Binary transformation
* Gaussian (Using two shaders by taking advantage of the separable nature of the effect)

I am leaving also an example of how to capture render doc captures of compute only work by using code.
