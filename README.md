**DE⫶TR**: End-to-End Object Detection with Transformers
========


**Inference Command**
```
python main.py --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path path_to_coco
```

**Note**
* batch_szie must be 1, 不然mask部分會有問題

**Experiments**
1. image size
   * datasets/coco.py line 141: ```T.RandomResize([2000], max_size=3000)```
   * size越大token越多，時間差會比較明顯 (但要跑比較久)
   * 原始的size太小，看不太出差異
2. e_r: r in encoder
3. d_r: r in object queries
  * detr object queries = 100
  * return_intermediate=False (不然output stack有問題)
  * 初步測試效果好像不太好，但也可以實驗顯示不能merge object queries
4. m_r: r of encoder outputs in decoder
