To run the code do the following:

cd build
cmake ..
make
./search


Note: This code is tested with with opencv 3.2 library
Note: This code requires ROS and is tested with ROS kinetic
Note: This code depends on saliency package. 

To use the saliency ROS package:

1- copy the files in saliency folder into your catkin folder
2- compile the code using catkin_make
3- run: rosrun saliency saliency 

Alternatively, to use the saliency module you can use the local Attention functions and remove ROS dependencies from the code.
