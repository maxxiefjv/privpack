privpack.core.architectures module
========================================

The classes defined in this package contain the groundwork for some basic data pre and post processing. For Example: to generate
privacy-preserving binary data we have pre-defined a privatizer network that maps two-dimensional binary input to a one-hot encoded 
input (4-dimensional). And no training related post-processing.  However, direct predictions are post-processed to their most-likely state.

For the gaussian case we only handle post-processing. As the adversary is defined to learn the parameters of a multivariate gaussian, we implemented
function to convert the adversary network\'s otuput to the parameters mu and sigma. Furthermore, functions are implemented to compute the likelihood 
that the adversary guesses the private variable given the released output.

An example for instanting one of the predefined classes:

.. code-block:: python

   # Instantiating binary release mechanism
   from privpack import BinaryPrivacyPreservingAdversarialNetwork as BinaryGAN
   binary_gan = BinaryGAN(torch.device('cpu'), PGANCriterion())

   #Instantiating gaussian 
   from privpack import GaussianPrivacyPreservingAdversarialNetwork as GaussGAN
   gauss_gan = GaussGAN(torch.device('cpu'), privacy_size, public_size, release_size, 
                        PGANCriterion(), no_hidden_units_per_layer=5, noise_size=1)



API Documentation
------------------

.. automodule:: privpack.core.architectures
   :members:
   :undoc-members:
   :show-inheritance: