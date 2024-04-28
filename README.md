<h1 align="center"> Multi-scale dehazing network via<br>high-frequency feature fusion </h1>


<div align="center">
    <a target='_blank'><strong>徐毓杰</strong>(2020级,2023届)</a>, 
    <a target='_blank'>张永军</a>*, 
    <a target='_blank'>李智</a>, 
    <a target='_blank'>崔忠伟</a>,  
    <a target='_blank'>杨亦童(2020级,2023届)</a>
</div>

<div align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S0097849322001182" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
</div>

## Abstract

Numerous learning-based methods have achieved significant improvements in haze removal. However, the dehazed results of these methods still suffer from the loss of edge details. To solve this problem, we propose a novel multi-scale dehazing network via high-frequency feature fusion (HFMDN). HFMDN is an end-to-end trainable network, which is mainly composed of four components: a base network (Backbone), a frequency branch network (FBN), a frequency attention module (FAM), and a refine block (RB). The Backbone is a multi-scale feature fusion architecture that can share useful information across different scales. For the training phase, we employ the Laplace Operator to obtain the image’s high-frequency (HF) information, which can specifically represent the details of the image (e.g., edges, textures). The FBN takes the HF derived from the original image as an additional prior and utilizes L1 norm loss to constrain the output of FBN to predict the HF of the haze-free image. We further design a frequency attention module (FAM), which automatically learns the weights map of the frequency features to enhance image recovery ability. Furthermore, a refine block (RB) is proposed to extract the features map by fusing the outputs of FBN and Backbone to produce the final haze-free image. The quantitative comparison of the ablation study shows that high-frequency information significantly improves dehazing performance. Extensive experiments also demonstrate that our proposed methods can generate more natural and realistic haze-free images, especially in the contours and details of hazy images. HFMDN performs favorably against the CNN-based state-of-the-art dehazing methods in terms of PSNR, SSIM, and visual effect.

