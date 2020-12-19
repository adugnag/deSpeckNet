# deSpeckNet: Generalizing Deep Learning Based SAR Image Despeckling.

Deep learning (DL) has proven to be a suitable approach for despeckling synthetic aperture radar (SAR) images. So far, most DL models are trained to reduce speckle that follows a particular distribution, either using simulated noise or a specific set of real SAR images, limiting the applicability of these methods
for real SAR images with unknown noise statistics. In this article,we present a DL method, deSpeckNet, 1 that estimates the speckle
noise distribution and the despeckled image simultaneously.Since it does not depend on a specific noise model, deSpeckNet generalizes well across SAR acquisitions in a variety of landcover conditions. We evaluated the performance of deSpeckNet onsingle polarized Sentinel-1 images acquired in Indonesia, The Democratic Republic of Congo, and The Netherlands, a single polarized ALOS-2/PALSAR-2 image acquired in Japan and an Iceye X2 image acquired in Germany. In all cases, deSpeckNet was able to effectively reduce speckle and restore the images in high quality with respect to the state of the art.

# Architecture


![drawing1](https://user-images.githubusercontent.com/48068921/102690422-96f76f00-4205-11eb-9ef0-5d98daecdee6.png)
![drawing_finetune](https://user-images.githubusercontent.com/48068921/102690424-99f25f80-4205-11eb-825b-dd9887935e67.png)



# Results
![Indonesia_images](https://user-images.githubusercontent.com/48068921/102690425-9c54b980-4205-11eb-9609-de105413c27d.png)
![DRC_images](https://user-images.githubusercontent.com/48068921/102690426-9e1e7d00-4205-11eb-8622-c478163ece66.png)
![Results](https://user-images.githubusercontent.com/48068921/102690427-9fe84080-4205-11eb-88ed-ec36582d4448.png)



# Citation
@article{mullissa2020despeck,
  title={deSpeckNet: Generalizing Deep Learning Based SAR Image Despeckling},
  author={Mullissa, A.G., Marcos, Diego., Tuia, Devis., Herold, Martin., Reiche, Johannes.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2020}
}
