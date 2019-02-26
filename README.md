# keras_rmac_plus
Keras implementation of R-MAC+ descriptors.

[[paper](https://arxiv.org/pdf/1806.08565.pdf)] [[project](http://implab.ce.unipr.it/?page_id=858)]

The image below represents the query phase exeucted for the R-MAC+ descriptors.

![query phase](http://implab.ce.unipr.it/wp-content/uploads/2018/09/queryImage.png)

## Prerequisites for Python3
* Keras (> 2.0.0)
* Tensorflow (> 1.5)
* Scipy
* Sklearn
* OpenCV 3

## Networks
The pipeline was tested with VGG16 and ResNet50. For the VGG16 the best performance are reached when the features are extracted from the block5_pool, instead for ResNet from the activation_43.
It is possible to try with other networks. Please before to try it, check if there are available the Keras weight for the selected network.

## Datasets
* Holidays
* Oxford5k
* Paris6k

Download the datasets and put it into the data folder. Then compile the script for the evaluation of the retrieval system.

## Test
` python3 Keras_test_MAC.py  `

## Results


| Method        | Network           | Oxford5k  | Paris6k | Holidays |
| :------------- |:-------------:| :-----:| :---:| :---------:|
| R-MAC | VGG16   | 65.56% | 82.80% | 87.65% |
| R-MAC | ResNet50   | 71.77% | 83.31% | 92.55% |
| M-R RMAC+ | ResNet50   | 78.88% | 88.63% | 94.63% / 95.58% |
| M-R RMAC+ with retrieval based on 'db regions' | ResNet50   | 85.39 %   | 91.90%  | 94.37% / 95.87% |

The R-MAC is an our re-implementation of the Tolias et al. 2016 paper, instead M-R RMAC comes from the Gordo et al. 2016 paper.
The last two experiments are also executed on the rotated version of Holidays.

## References

<pre>@article{magliani2018accurate,
  title={An accurate retrieval through R-MAC+ descriptors for landmark recognition},
  author={Magliani, Federico and Prati, Andrea},
  journal={arXiv preprint arXiv:1806.08565},
  year={2018}
}

@article{tolias2015particular,
  title={Particular object retrieval with integral max-pooling of CNN activations},
  author={Tolias, Giorgos and Sicre, Ronan and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:1511.05879},
  year={2015}
}

@inproceedings{gordo2016deep,
  title={Deep image retrieval: Learning global representations for image search},
  author={Gordo, Albert and Almaz{\'a}n, Jon and Revaud, Jerome and Larlus, Diane},
  booktitle={European Conference on Computer Vision},
  pages={241--257},
  year={2016},
  organization={Springer}
}

</pre>
