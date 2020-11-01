privpack.core.criterion module
================================

Criterions are what define the learning behavior of the privatizer and adversary network. This page shows how 
to setup a GANCriterion in a few steps, and how to extend these with your own criterion functions.

We have defined two types of criterion functions: either of concern for utility or for privacy. To create
a privacy-preserving gan criterion you insantiate the PGANCriterion class, and add the utilities and privacies 
needed by the privatizer and adversary using the therefore predefined functions (). An example is shown as:


.. code-block:: python

   binary_gan_criterion = PGANCriterion()
   binary_gan_criterion.add_privacy_criterion(BinaryMutualInformation())
   binary_gan_criterion.add_privacy_criterion(BinaryHammingDistance(lambd, delta_constraint))


API Documentation
------------------

.. automodule:: privpack.core.criterion
   :members:
   :undoc-members:
   :show-inheritance: