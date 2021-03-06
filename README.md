## Effect Prediction

We propose an effect prediction model which predicts the next state after taking an action. Furthermore, We assume an incomplete training dataset lacking some effect labels, which puts our work under the paradigm of zero shot learning (ZSL). Our model is a cascaded neural network leveraging one Inception ConvNets and two recurrent neural networks (RNNs). The output layer is essentially an energy-based regression layer, minimizing a hinge-loss in which distance is modeled by T-SNE pairwise similarity computation, and hence hopefully outputs the embedding of effect labels, no matter whether the label exists in training dataset. 

## Dependencies

* Python 2.7
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [skimage](http://scikit-image.org/docs/dev/api/skimage.html)
* [Theano](http://www.deeplearning.net/software/theano/)
* [h5py](http://docs.h5py.org/en/latest/)

## Reference

If you use this code as part of any published research, please acknowledge the
following papers:

**"Action Recognition using Visual Attention."**  
Shikhar Sharma, Ryan Kiros, Ruslan Salakhutdinov. *[arXiv](http://arxiv.org/abs/1511.04119)*

    @article{sharma2015attention,
        title={Action Recognition using Visual Attention},
        author={Sharma, Shikhar and Kiros, Ryan and Salakhutdinov, Ruslan},
        journal={arXiv preprint arXiv:1511.04119},
        year={2015}
    } 

**"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."**  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhutdinov, Richard Zemel, Yoshua Bengio. *To appear ICML (2015)*

    @article{Xu2015show,
        title={Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
        author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1502.03044},
        year={2015}
    }


## Acknowledgements
*Zongsi Zhang, Jianmei Ye, Mengyuan Zhang for creating effect-prediction dataset [EP3A](https://www.dropbox.com/s/es6ipjv5u6ikysr/EP3A.zip?dl=0).

*Yu Zhang, Subbarao Kambhampati, Hannah Kerner, Xiaoyu Zhang for offering design suggestions.

## License
This repsoitory is released under a [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html). The repository uses some code from the project [Action Recognition using Visual Attention](https://github.com/kracwarlock/action-recognition-visual-attention), which is licensed under a revised (3-clause) BSD License; And [Inception](https://github.com/google/inception) which is licensed under Apache License 2.0. 
