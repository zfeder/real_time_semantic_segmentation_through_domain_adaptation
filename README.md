# Authors  
**Federico Filì** (Politecnico di Torino) - [s332158@studenti.polito.it](mailto:s332158@studenti.polito.it)  
**Giorgio Licitra** (Politecnico di Torino) - [s332162@studenti.polito.it](mailto:s332162@studenti.polito.it)  

## Abstract  
This project investigates the use of domain adaptation techniques to enhance the performance of semantic segmentation models in transitioning from synthetic to real-world data. Focusing on the comparison between a classical segmentation model (DeepLabV2) and a real-time segmentation network (BiSeNet), this work demonstrates how data augmentation and adversarial training can mitigate the domain shift challenge. The proposed methods aim to balance accuracy and efficiency for real-time applications such as autonomous driving and urban scene understanding.  

## Introduction  
Semantic segmentation plays a crucial role in computer vision, requiring pixel-wise classification for applications such as autonomous driving, surveillance, and robotics. Real-time segmentation networks like BiSeNet offer efficient solutions but often suffer from reduced accuracy when trained on synthetic datasets like GTA5 and tested on real-world datasets like Cityscapes. This project explores domain adaptation techniques, including data augmentation and adversarial learning, to improve model robustness and reduce the performance gap caused by domain shift.  

## Methods  
- **DeepLabV2**: A classical, high-accuracy semantic segmentation network.  
- **BiSeNet**: A real-time segmentation network optimized for low-latency applications.  
- **Domain Adaptation Techniques**:  
  - Data augmentation: Horizontal flipping, Gaussian blur, color jitter, and random cropping.  
  - Adversarial learning: A discriminator-generator framework to align synthetic and real-world feature distributions.  

## Results  
- **Baseline Performance**: Without domain adaptation, BiSeNet experiences significant mIoU drops when transitioning from synthetic to real-world datasets.  
- **Data Augmentation**: Improved mIoU from 21.11% to 23.93%.  
- **Adversarial Training**: Further enhanced mIoU to 27.01%, highlighting the potential of domain adaptation techniques.  
- **Comparison with DeepLabV2**: While DeepLabV2 achieves higher accuracy, BiSeNet offers a practical trade-off for real-time use cases.  

## Conclusion  
This project highlights the effectiveness of domain adaptation techniques in improving the performance of real-time semantic segmentation models for real-world applications. By combining data augmentation and adversarial learning, BiSeNet achieves substantial improvements, narrowing the performance gap with classical models. Future work could explore alternative adaptation strategies and test these methods across different datasets and domains to further refine and validate their applicability.

## 📜 License
© 2025 Federico Filì and Giorgio Licitra. All rights reserved.

This project is licensed for personal and educational purposes only. Unauthorized distribution, reproduction, or commercial use is strictly prohibited without prior permission.
