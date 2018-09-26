# keras_rmac_plus
Keras implementation of R-MAC+ descriptors

[[paper](https://arxiv.org/pdf/1806.08565.pdf] [[project](http://implab.ce.unipr.it/?page_id=858)]


![query phase](https://drive.google.com/open?id=1mQ5lddOQgDvkY20pOL490qdf_mcu6qfI)

## Prerequisites for Python3
* Keras (> 2.0.0)
* Tensorflow (> 1.5)
* Scipy
* Sklearn
* OpenCV 3

## Datasets
* Oxford5k
* Paris6k

Download the datasets and put it into the data folder. Then compile the script for the evaluation of the retrieval system.

## Test
` python3 Keras_test_MAC.py  `



## Reference

<pre>@article{magliani2018accurate,
  title={An accurate retrieval through R-MAC+ descriptors for landmark recognition},
  author={Magliani, Federico and Prati, Andrea},
  journal={arXiv preprint arXiv:1806.08565},
  year={2018}
}</pre>
