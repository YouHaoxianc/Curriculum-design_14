[王琛玙的数据集](https://pan.baidu.com/s/1O2W36ViBzHlsMIIVD-OJig?pwd=xx4w)

将收集到的jpg图片（文件名不含中文）放入~\images文件夹中，运行~\resize_images.py将图片的高变为384。用[CVAT](https://app.cvat.ai/)对每张图片中三个计数对象标注类型为rectangle的框，对所有计数对象标注类型为points的点，导出为datumaro 1.0格式得到~\default.json，运行~\adapt_annotations&generate_density_maps.py得到~\annotations.json和~\density_maps。
