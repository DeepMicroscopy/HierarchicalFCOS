# HierarchicalFCOS
Deep-learning bysed Subtyping for Atypical and Typical Mitosis using a Hierarchical Anchor-Free Object Detector

This repository shows the training scripts we used for our BVM workshop submission "Deep learning-based Subtyping for Atypical and Typical Mitosis using a Hierarchical Anchor-Free Object Detector".

What's not contained in this repository is the databases and images of the datasets we have trained with. Please note that we trained
and evaluated on the respective full (train+test) sets of the respective datasets, as these were available to us and it increases dataset variability and/or evaluation robustness.

In this work, we extended the anchor-free FCOS object detection approach with two hierarchical labels. The architecture is depicted in this graphic:
![image](https://user-images.githubusercontent.com/10051592/207099826-919ee776-b021-499d-a824-19f403484841.png)

The subcategorization of mitotic figures was performed into atypical and normal mitotic figures and
according to the phases of mitosis:

![image](https://user-images.githubusercontent.com/10051592/207100132-137661d0-d3aa-40a0-9ba2-84fbf2d4e6f3.png)


