# deSpeckNet: Generalizing Deep Learning Based SAR Image Despeckling.

Deep learning (DL) has proven to be a suitable approach for despeckling synthetic aperture radar (SAR) images. So far, most DL models are trained to reduce speckle that follows a particular distribution, either using simulated noise or a specific set of real SAR images, limiting the applicability of these methods
for real SAR images with unknown noise statistics. In this article,we present a DL method, deSpeckNet, 1 that estimates the speckle
noise distribution and the despeckled image simultaneously.Since it does not depend on a specific noise model, deSpeckNet generalizes well across SAR acquisitions in a variety of landcover conditions. We evaluated the performance of deSpeckNet onsingle polarized Sentinel-1 images acquired in Indonesia, The Democratic Republic of Congo, and The Netherlands, a single polarized ALOS-2/PALSAR-2 image acquired in Japan and an Iceye X2 image acquired in Germany. In all cases, deSpeckNet was able to effectively reduce speckle and restore the images in high quality with respect to the state of the art.

# Architecture


![drawing1](https://user-images.githubusercontent.com/48068921/102690422-96f76f00-4205-11eb-9ef0-5d98daecdee6.png)
![drawing_finetune](https://user-images.githubusercontent.com/48068921/102690424-99f25f80-4205-11eb-825b-dd9887935e67.png)


The articles pre-print is available on https://arxiv.org/pdf/2012.03066.pdf
