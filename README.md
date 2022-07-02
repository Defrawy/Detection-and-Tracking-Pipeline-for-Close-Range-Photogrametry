# Detection-and-Tracking-Pipeline-for-Close-Range-Photogrametry


Detection and tracking tools are important for various tasks. This research develops a pipeline to utilize detection and tracking for close-range photogrammetry. The pipeline loads a short video to detect and track various objects in the scene and separates them into different sub-scenes. The resulting substances can be fed into a structure from motion software and generate a 3D point cloud of the object. Installation instructions can be found [here](https://github.com/Defrawy/deep_sort). 

A data sample of a miniature yellow truck is available on [google drive](https://drive.google.com/file/d/1lWrEl7t5qLk-Yp6Zn7_6cdn4bf1HlDxR/view?usp=sharing). The data is the imagery of a controlled (turntable setup with black background) close-range photogrammetry scene and imagery of an uncontrolled (turntable with rich background) scene. The uncontrolled imagery includes both raw frames without any processing and a masked version of the yellow truck with a black background. The masked version is processed by the detection and tracking pipeline to isolate the object from the scene. The imagery data can be fed to the structure from motion software, and it would result in a point cloud of the input scene. In general, the masked imagery is almost as accurate as the controlled scene imagery, and the unprocessed imagery is usually a deformed and inaccurate representation of the object. 

The data is recorded using iPad Pro 4th generation. However, the camera information was unnecessary for obtaining a high-quality point cloud. So, the pipeline can be used to reconstruct uncontrolled arbitrary videos.


For additional information please contact: meldefrawy@islander.tamucc.edu

