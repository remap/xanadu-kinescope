# xanadu-kinescope

First create a conda env conda `create -n myenv python=3.9`.
In your ZED SDK folder on your laptop run the following

`pip install requests google.auth`

`python get_python_api.py  (From zed SDK folder, might require permisions)`

`pip install numpy==1.26.4`

`pip install matplotlib==3.8.4`

`pip install scipy (installed scipy==1.13.1 on my env)`

`pip install dtaidistance`

DO NOT PIP INSTALL FIREBASE -- IT WILL INTERFERE WITH THE SUBMODULE AND CAUSE IMPORT ISSUES.

`pip install firebase-admin`

`pip install pycocotools`

Then cd into `xanadu-kinescope` then `fb_example` then run your desired file `real_time_oks_sender.py`

OLD Deprecated below.
cocoTest.py runs the evaluation algorithm.
body_tracking.py is the built in Zed python file to track data from the camera.

Ground truth files will store the list of images and their anntoations.
Detection files will store what was actually detected.

One person will be in the GT file, the rest in the D files.
