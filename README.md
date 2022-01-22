# Fundamental_Matrix_Epipolar_Geometry

We work on the estimate of fundamental matrix for encoding epipolar geometry between two images taken by a moving camera.
We use OpenCV under Python.

The estimation method used for fundamental matrix is the following. The algorithm gives an estimate using KAZE points and the proximity between those points using a nearest neighbor criteria between pairs of descriptors. We also introduce a ratio test which is the maximum ratio between the distance to the nearest neighbor and the second nearest neighbor. Then, the fundamental matrix is computed using RANSAC method.

The best result is obtained when the epipolar lines are concurrent in a point, as it means the constraint over the degrees of freedom is respected and is of rank 2. In this case, the rectification is correctly made. If the algorithm succeeds to find a lot of pairings, the result is better.

When the camera has a radial movement, the epipolar point is inside the image. This kind of movement is not successfully rectified as geometric constraints are too high. The best result is obtained when the movement is a translation of the camera in the focal plane.

The important parameters here are the distance between the epipolar plans for the two cameras, and the coordinates of the optical centers.

We compute a stereo rectification, which consists in the transfer of the epipolar point to infinity in the horizontal direction. This method is more adapted for a translation so in the case of a radial movement of the camera, one should compute a polar rectification instead.