## Face Recognition

Principle Components Analysis, AdaBoosting, and the Viola-Jones algorithm are implemented from scratch for face recognition.    

#### Principle Components Analysis

* Training: obtaining the eigenvalues and eigenvectors from training dataset. Each image face in the training dataset is then projected to an eigenface (projected face) using these vectors.

* Testing:  Each image face in the test dataset is also projected to an eigenface by using vectors obtained from training. Then, find the eigenface in the training dataset that is closest to each projected face in the test dataset, then we will use the label that belongs to the training eigenface as our predicted class.

#### AdaBoosting

* Decision Stump is used as a Weak Classifier h(x), which is implemented to predict based on threshold values. Boosting creates a classifier H(x) which is the combination of simple weak classifiers.

#### Viola-Jones
* Haar-like Features: five types of features, namely, two-horizontal, two-vertical, three-horizontal, three-vertical, and four-rectangle are used. Details are shown in [Viola-Jones paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-IJCV-01.pdf).

* Haar-like features can be used to train classifiers but use a single feature a time. The results from this process can be then applied in the boosting algorithm. We will use a dataset of images that contain faces as positive examples. Additionally, use the dataset of images with other objects as negative examples.

### Running the Tests
All results for each project are shown in **Report.pdf**.

### Authors
* **Manqing Mao,** maomanqing@gmail.com

<!-- See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->
