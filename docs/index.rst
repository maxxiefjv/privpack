====================================
Privpack Project
====================================

Project information:

This project is created as part of a master thesis on privacy-preserving release mechanisms inspired by Generative Adversarial Networks. The goal
of this library is to provide the groundwork for quickly building and testing privpacy-preserving release mechanisms. To read more on the need for such 
systems in the corresponding thesis (in-progress).

This small-library is built on top of pytorch, and is created to provide the basics for a release mechanisms by using a GAN architecture. However,
as a researcher every detail may matter to your study. Therefore, this library is built in such a way that it will not restrict the freedom of your work.

In this document you can find the available classes and helper methods to create your release mechanisms. The classes needed for your quick setup are found 
in the package: privpack/core. This package contains GAN architectures and Basic Criterion classes. Each release mechanisms is built using these building blocks.
The GAN architecture should define the type of data you want to handle, and the criterions define how the privatizer and adversary should behave. The provided 
criterions are in the area of information theory and therefore contain the privacy criterion functions to optimize: Mutual information, Maximal Leakage, Alpha-Leakage and Maximal 
Alpha-Leakage.


Privpack API Documentation
====================================

.. toctree::
   :maxdepth: 2

   api/privpack.core
   api/privpack.model_selection
   api/privpack.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
