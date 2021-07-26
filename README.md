# volume-detect-v1.2

needed: <br />
numpy 1.19.5<br />
tensorflow<br />
pillow<br />
detecto<br />

i.e:<br />
pip install detecto<br />
pip install numpy==1.19.5<br />
pip install pillow<br />
pip install tensorflow<br />

V2 code removes some unnecessary code and makes it compatible with "new_keras.h5" <br />

"model.pth" is used for finding the bounding box of the crucible. <br />
"keras_model.h5" is used for the volume detection of 5 categories (not recommended).  <br />
"new_keras.h5" is used for volume detection of 10 categories (better model). 

