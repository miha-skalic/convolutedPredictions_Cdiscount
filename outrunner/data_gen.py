import cv2
import numpy as np
import threading

class Generator(object):
    def __init__(self, batch_size, keys,
                 up_width = 180,
                 saturation_var=0.,
                 brightness_var=0.,
                 contrast_var=0.,
                 color_var=0.,
                 hflip_prob=0.,
                 vflip_prob=0.,
                 rotat_prob=0.,
                 train = True,
                 crop = False,
                 width = 139,
                 vgg = False):
        self.batch_size = batch_size
        self.keys = keys
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotat_prob = rotat_prob
        self.color_jitter = []
        self.up_width = up_width
        self.train = train
        self.gen = self.generate_key()
        self.lock = threading.Lock()
        self.crop = crop
        self.width = width   
        self.vgg = vgg
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        if color_var:
            self.color_var = color_var
            self.color_jitter.append(self.color)
            
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def color(self, rgb):
        r = 1 - self.color_var + 2 * self.color_var * np.random.random() 
        g = 1 - self.color_var + 2 * self.color_var * np.random.random() 
        b = 1 - self.color_var + 2 * self.color_var * np.random.random() 
        rgb[:,:,0] = 255*(rgb[:,:,0]/255)**b
        rgb[:,:,1] = 255*(rgb[:,:,1]/255)**g
        rgb[:,:,2] = 255*(rgb[:,:,2]/255)**r
        return np.clip(rgb, 0, 255)   
    
    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.brightness_var
        rgb = rgb * alpha
        #if alpha < 1 or np.sum(np.sum(rgb1, axis=2)>720)/rgb.shape[0]/rgb.shape[1] < 0.02:
        #    rgb = rgb1
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        #if np.sum(np.sum(rgb1, axis=2)>720)/rgb.shape[0]/rgb.shape[1] < 0.02:
        #    rgb = rgb1
        return np.clip(rgb, 0, 255)
       
    def horizontal_flip(self, img):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
        return img
    
    def vertical_flip(self, img):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
        return img
    
    def rotate90(self, img):
        if np.random.random() < self.rotat_prob:
            img = np.rot90(img)
            img = img.copy()
        return img
    
    def r_crop(self, img, w=139):            
        x = np.random.randint(w,self.up_width+1)
        img = cv2.resize(img, (x, x))
        ws = np.random.randint(x-w+1)
        hs = np.random.randint(x-w+1)
        return img[ws:ws+w, hs:hs+w,:] 
    
    def generate_key(self):
        while True:
            if self.train:
                np.random.shuffle(self.keys)

            targets = []
            for key in self.keys:            
                targets.append(key)
                if len(targets) == self.batch_size:
                    tmp_targets = np.array(targets)
                    targets = []
                    yield tmp_targets

    def generate(self, keys):
        inputs = []
        targets = []
        for key in keys:            
            img = cv2.imread('train/'+key[0])
            
            if self.crop:
                img = self.r_crop(img, self.width)                    
            elif self.width != img.shape[0]:
                img = cv2.resize(img, (self.width, self.width))

            y = np.zeros(5270)
            y[key[1]] = 1
                
            if self.train:
                np.random.shuffle(self.color_jitter)
            for jitter in self.color_jitter:
                img = jitter(img)
            if self.hflip_prob > 0:
                img = self.horizontal_flip(img)
            if self.vflip_prob > 0:
                img = self.vertical_flip(img)
            if self.rotat_prob > 0:
                img = self.rotate90(img)

            if self.vgg == True:
                inputs.append(img.astype('float32')-[103.939, 116.779, 123.68])
            else:
                inputs.append(img.astype('float32')/127.5 - 1)
            targets.append(y)
              
        return np.array(inputs), np.array(targets)
                    
    def __next__(self):
        with self.lock:
            keys = next(self.gen)

        return self.generate(keys)
    
    
    
    