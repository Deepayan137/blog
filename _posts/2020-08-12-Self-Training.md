---
toc: true
layout: post
description: An introductory blog describing Semi-supervised Learning Algorithm
categories: [markdown]
title: A primer on semi-supervised Learning
---

Deep Learning (DL) algorithms typically rely on a huge amount of labelled data pairs. However, it is often expensive to collect such annotated datasets in terms of both cost and time. ImageNet, the largest image database in the present day consists of around 14 million images. Each image in it was hand-annotated by several annotators using a crowdsourcing platform known as Amazon Mechanical Turk. There are several other image datasets like PASCAL VOC and MS COCO which consist far fewer images compared to ImageNet (10k and 100k respectively) and it is possible to train a DL network on such dataset satisfactorily to learn a new task. However, the general trend in DL literature suggests that the performance of the models can further be improved if more and more data is added. 

## Transfer Learning

As mentioned above Deep Learning algorithms are essentially data-hungry methods which require tons of data to estimate its millions of parameters. This property effectively renders it useless to problems that have limited training data. However, DL models have this remarkable ability where the representations learned over large datasets can be effectively transferred to tasks which have a limited amount of training data. [2] explored applied this idea and applied it to train deep CNN models on limited labelled data. The observed that

> CNN can act as a generic extractor of mid-level image representation, which can be pre-trained on one dataset (the source task) and then re-used on other target tasks.

The authors pre-trained a network on 1000 classes of ImageNet data and used to perform object detection on Pascal VOC data. Applying this method they were able to achieve a significant gain in performance compared to the existing baseline results. Besides, transfer learning has shown great promise in other computer vision tasks such as segmentation [3] and recognition [4].

Although, transfer learning is a very useful trick for improving gains on tasks with limited training data, yet it **does not take advantage of the massive amounts of unlabeled data** available over the internet. Also, it suffers from the **in consequence of having to annotate a portion of target data** which can prove to be quite a lot if the task involves speech or text recognition. In such a case each word and utterance has to be correctly transcribed manually which can prove to be very tedious and time-consuming.

## Semi-supervised Learning

These above drawbacks can be easily overcome by semi-supervised learning algorithms which are designed is such a way that they can work with both labelled and unlabeled data. Consider Google Images or Instagram. A simple query can fetch you thousands of results. However, the retrieved images are unstructured and unannotated and cannot be put to use if we are using supervised learning algorithms. Semi-supervised learning algorithms, on the other hand, make use of not only the labelled target data but also use the myriads of unlabeled data to learn better representations. This property gives an edge to the SSL algorithms over traditional finetuning approaches. There exist mainly two approaches towards implementing SSL. The first approach involves passing of the unlabeled images through different augmentations and perturbations. Since the images are constant, the model predictions should not be swayed by the perturbations and predict the same label. Forcing the model to come up with the same prediction under a different set of noise/perturbation can act as a source of regularization and together with the supervised loss term, it helps the DL model towards more stable generalization during testing and also helps the model learn more robust invariant features [5, 6]. The second approach involves inferring labels for the unlabeled data and which is then ( *pseudo-labelled data*) used to train the model with a supervised loss term. The second approach which is also known as pseudo-labelling falls under the category of transductive learning where both labelled and unlabeled data is used to improve the performance of a model. Labels for the unlabeled data can be inferred in two ways. 1) By constructing a graph and propagating labels from known to unknown data points and 2) By using an existing pre-trained classifier to invoke the labels unlabeled data points. 

### Label Propagation

Label propagation [7, 8] deals with the construction of a graph between the labelled and unlabeled data points which is later used to propagate the labels from labelled to unlabeled data using cluster assumptions. Label Propagation comprises of two steps: construction of a graph and inference. In the graph construction stage, data points of both labelled and unlabeled data form the nodes while the edges represent the similarity between the data points. Larger edge weights indicate higher similarity between the data points and vice-versa. The most common technique to create a graph is using a clustering algorithm such as kNN where the edge weights are obtained using as a Euclidean based distance function. In the inference stage, the labels from the labelled data are propagated to their nearest unlabeled data points along with a certainty score which is simply the Euclidean distance from the nearest labelled data point.

### Self-training

Self-training [9, 10] was one of the earliest attempts to use unlabeled data to boost model performance. Self-training comprises of two stages: Initially, the model is trained on a limited amount of labelled data using a supervised method. In the next stage, the learned model is used to predict the labels of unlabeled data points. Finally, the model is trained on both the labelled and unlabeled data where the predictions of unlabeled data are treated as target labels. One of the major concerns of self-training is that initial trained model might predict a significant amount of unlabeled data erroneously. This might bring down the performance of a model while training rather than improving. Care should be taken to minimize the number of noisy predictions in the training set. One way to do so would be to screen the predictions effectively and include only those predictions on which the model is highly confident. Despite the screening measure, some noisy predictions still manage to creep into the training data, which hinders the learning of the ML algorithm. Most works in this domain discuss providing perturbations to the input data or model as a way to overcome confirmation bias. We will see in the next few sections of how linear perturbation is known as *mixup* [11] and as well as some other regularization method are used to improve the performance of a text recognition system on an unlabeled target dataset.

In the next post, we will get our hands dirty trying to implement self-training algorithm on a classic computer vision problem of text-recognition or more commonly known as OCR.

[1] [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575)

[2] [Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)

[3] [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

[4] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[5] [Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning](https://arxiv.org/pdf/1606.04586.pdf)]

[6] [Temporal Ensembling for semi-supervised learning](https://arxiv.org/pdf/1610.02242.pdf)

[7] [Label Propagation for Deep Semi-supervised Learning](https://arxiv.org/pdf/1904.04717.pdf)

[8] [Learning by association â€“ a versatile semi-supervised training method for neural networks](https://arxiv.org/abs/1706.00909)

[9] [Probability of error of some adaptive pattern-recognition machines](https://ieeexplore.ieee.org/abstract/document/1053799)

[10] [Learning to recognize patterns without a teacher](Learning to recognize patterns without a teacher)

[11] [mixup beyond empirical risk minimization](https://arxiv.org/pdf/1710.09412.pdf)