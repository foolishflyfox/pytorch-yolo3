import sys
import time
import os
import json
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

def detect_images(cfgfile, weightfile, images_dir, target_dir=None):
    m = Darknet(cfgfile)

    # m.print_network()
    m.load_weights(weightfile)
    # print('Loading weights from %s... Done!' % (weightfile))

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    assert os.path.isdir(images_dir)
    if target_dir is None:
        target_dir = os.path.join(images_dir, 'yolo3_prediction')
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
    assert os.path.isdir(target_dir)

    images = [img for img in os.listdir(images_dir) 
                if img[-4:] in ['.png', '.jpg', 'jpeg']]
    for filename in images:
        imgfile = os.path.join(images_dir, filename)
        tp_fname = os.path.split(filename)[1]
        tp_fname = os.path.splitext(tp_fname)[0]
        target_file = os.path.join(target_dir, tp_fname+'.json')
        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((m.width, m.height))
    
        for i in range(2):
            # start = time.time()
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            # finish = time.time()
            # if i == 1:
                # print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(namesfile)
        # plot_boxes(img, boxes, target_file, class_names)
        save_bbox(filename, img, boxes, target_file, class_names)


# 将标签以 json 的形式输出
def save_bbox(filename, img, boxes, target_file, class_names):
    save_obj = {
        'path':filename,
        'labeled': False,
        'size': {'depth':3, 'height': img.height, 'width': img.width},
        'time_labeled': int(time.time()*1000),
        'outputs':{'object':[]}
    }
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * img.width
        y1 = (box[1] - box[3]/2.0) * img.height
        x2 = (box[0] + box[2]/2.0) * img.width
        y2 = (box[1] + box[3]/2.0) * img.height
        if len(box)>=7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            if class_names[cls_id]!='person':
                continue
            save_obj['outputs']['object'].append({
                'bndbox':{
                    'xmin': int(x1.item()),
                    'ymin': int(y1.item()),
                    'xmax': int(x2.item()),
                    'ymax': int(y2.item())
                },
                'name': 'person'
            })

    save_obj['labeled'] = len(save_obj['outputs']['object'])>0
    # print(save_obj)
    with open(target_file, 'w') as f:
        json.dump(save_obj, f)
    print(f"{target_file} finished!" )



def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)




if __name__ == '__main__':
    if len(sys.argv) >= 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        # imgfile = sys.argv[3]
        images_dir = sys.argv[3]
        if len(sys.argv)>4:
            target_dir = sys.argv[4]
        else:
            target_dir = None
        detect_images(cfgfile, weightfile, images_dir, target_dir)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('python detect_images.py cfgfile weightfile images_dir target_dir')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
