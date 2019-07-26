# CACFNet
## Introduction
This project is a pytorch implementation of CACF, aimed to make the CACF filter as a differentiable layer to learn the convolutional features and perform the correlation tracking process simultaneously.

I referred the implmentations, [DCFNet](https://arxiv.org/pdf/1704.04057.pdf) by Qiang Wang and the sorce code from the [CACF](https://ivul.kaust.edu.sa/Pages/pub-ca-cf-tracking.aspx).

## What we are doing and going to do
- [X] track with [got10k](https://github.com/got-10k/toolkit) toolkit.
- [X] train from scratch.
- [ ] the result of the CACFNet on the benchmark.

## Paper
1 DCFNet: Qiang Wang, Jin Gao, Junliang Xing, Mengdan Zhang, Weiming Hu. "DCFNet: Discriminant Correlation Filters Network for Visual Tracking." arXiv (2017). [[paper](https://arxiv.org/pdf/1704.04057.pdf)] [[code](https://github.com/foolwood/DCFNet#dcfnet-discriminant-correlation-filters-network-for-visual-tracking)].

2 CACF: Matthias Mueller, Neil Smith, Bernard Ghanem. "Context-Aware Correlation Filter Tracking." CVPR (2017 oral).[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mueller_Context-Aware_Correlation_Filter_CVPR_2017_paper.pdf)] [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Mueller_Context-Aware_Correlation_Filter_2017_CVPR_supplemental.zip)]      [[project](https://ivul.kaust.edu.sa/Pages/pub-ca-cf-tracking.aspx)[code](https://github.com/thias15/Context-Aware-CF-Tracking)].
