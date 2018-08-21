Detect & Track

Track persons and vehicles

## Compile SSD Caffe

    $ git clone https://github.com/weiliu89/caffe.git
    $ cd caffe
    $ git checkout ssd
    $ mkdir build && cd build
    $ cmake ..
    $ make -j8
    $ make pycaffe   # make sure making pycaffe is successfull


## Run Tracking

    # [note] use python2.7
    $ export PYTHONPATH=<PATH_TO_CAFFE>/python
    $ python track.py <input_video>  [1|0]  # 1 if you have GPU and Cuda, 0 otherwise.
