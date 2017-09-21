import numpy as np
from PIL import Image,ImageEnhance
import multiprocessing as mp
from functools import partial
from collections import Counter
import tensorflow as tf

class DataGenerator(object):

    def __init__(self,batch_size,n_labels,train_size,n_slices,
                 image_size, n_channels, resize_to, data_type, session):
        self.batch_size = batch_size
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_channels = n_channels
        self.resize_to = resize_to
        self.train_size = train_size

        self.slice_size = train_size//n_slices
        self.steps_per_slice = self.slice_size//batch_size
        self.n_slices = n_slices

        self.slice_index = 0
        self.step_index_in_slice = 0

        self.slice_index_changed = True

        self.current_image_slice, self.current_label_slice = None, None
        self.labels_to_index_map = {}
        print('Slice size: ',self.slice_size)
        print('Steps per slice: ',self.steps_per_slice)
        print('Number of slices: ',self.n_slices)

        self.tf_image_ph = tf.placeholder(shape=(self.batch_size,image_size,image_size,n_channels),dtype=tf.float32)
        self.tf_label_ph = tf.placeholder(shape=(self.batch_size,self.n_labels),dtype=tf.float32)

        self.session = session
        self.tf_augment_data_func = self.tf_augment_data_with(data_type)

    def tf_augment_data_with(self, dataset_type):
        tf_image_batch = self.tf_image_ph

        if dataset_type == 'imagenet-250':
            tf_image_batch = tf.image.resize_images(tf_image_batch,[self.resize_to+32,self.resize_to+32])
            tf_image_batch = tf.random_crop(tf_image_batch,[self.batch_size,self.resize_to,self.resize_to,self.n_channels],seed=13423905832)

        if dataset_type != 'svhn-10':
            tf_image_batch = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), tf_image_batch)

        tf_image_batch = tf.image.random_brightness(tf_image_batch,0.5)
        tf_image_batch = tf.image.random_contrast(tf_image_batch,0.5,1.5)

        # Not necessary they are already normalized
        #tf_image_batch = tf.map_fn(lambda img: tf.image.per_image_standardization(img), tf_image_batch)

        return tf_image_batch

    def generate_data_with_label_sequence(self, dataset_images, dataset_labels, label_sequence, dataset_info):
        '''
        Generating data with a specific label sequence
        :param label_sequence: A list of list where the size is [n_steps, n_labels]
        :return: A generator
        '''
        return self.get_data_batch_for_labels(dataset_images, dataset_labels, label_sequence, dataset_info)

    def get_data_batch_for_labels(self, dataset_images, dataset_labels, label_sequence, dataset_info):
        global step_in_slice, steps_per_slice, slice_index

        dataset_name, resize_to, n_labels = dataset_info['dataset_name'], dataset_info['resize_to'], dataset_info['n_labels']

        if self.slice_index_changed:
            print('Load data slice')
            self.current_image_slice = dataset_images[self.slice_index*self.slice_size:(self.slice_index+1)*self.slice_size,:,:,:]
            self.current_label_slice = dataset_labels[self.slice_index*self.slice_size:(self.slice_index+1)*self.slice_size,0]
            for label in range(self.n_labels):
                self.labels_to_index_map[label] = list(np.where(self.current_label_slice == label)[0].flatten())
            print('Slice index changed')

        # USE for Testing
        #print('Summary of the label slice')
        #print('Slice index: ',self.slice_index)
        #print('Class distribution: ',Counter(label_slice.tolist()))

        lbl_cnt = Counter(label_sequence)
        img_indices = []
        for label in range(n_labels):
            img_indices_for_label = np.random.choice(self.labels_to_index_map[label],size=lbl_cnt[label])
            img_indices.extend(img_indices_for_label.tolist())

        assert len(img_indices)==self.batch_size,'Selected random indices count is not same as batch size'
        # has one additional axis for the split axis
        image_list= np.split(self.current_image_slice[img_indices,:,:,:],self.batch_size,axis=0)
        sorted_label_list = sorted(label_sequence)

        rng_state = np.random.get_state()
        np.random.shuffle(image_list)
        np.random.set_state(rng_state)
        np.random.shuffle(sorted_label_list)

        train_images = self.session.run(self.tf_augment_data_func,feed_dict={self.tf_image_ph:np.squeeze(np.stack(image_list))})
        train_labels = np.asarray(sorted_label_list).reshape(-1,1)

        train_ohe_labels = np.zeros((self.batch_size,self.n_labels),dtype=np.float32)
        train_ohe_labels[np.arange(self.batch_size),train_labels[:,0]] = 1.0

        # Check if one hot encoding is done right
        assert np.all(np.argmax(train_ohe_labels,axis=1)==train_labels.flatten())

        self.step_index_in_slice = (self.step_index_in_slice+1)%self.steps_per_slice

        # whenever steps_per_slice number of steps completed,
        # increment the slice index by 1
        if self.step_index_in_slice == 0:
            self.slice_index = (self.slice_index+1)%self.n_slices
            self.slice_index_changed = True
        else:
            self.slice_index_changed = False

        return train_images,train_ohe_labels

    def get_augmented_sample_for(self, image, label, resize_to, dataset_type):
        global image_abuse_threshold
        global save_gen_data,gen_save_count,gen_perist_dir

        # Convert image to 255 color and uint8 data type
        image = image[0,:,:,:]

        uint8_image = image - np.min(image)
        uint8_image = (uint8_image*255.0/np.max(uint8_image)).astype(np.uint8)
        image_size = uint8_image.shape[0]

        assert image_size == uint8_image.shape[1], "Image has different width and height"

        im = Image.fromarray(uint8_image)

        if dataset_type=='imagenet-250':
            im.thumbnail((resize_to+32,resize_to+32), Image.ANTIALIAS)

        if dataset_type=='imagenet-250' and np.random.random()<0.5:
            center = image_size//2
            x,y = np.random.randint(center-(resize_to//2)-16,center-(resize_to//2)+16),np.random.randint(center-(resize_to//2)-16,center-(resize_to//2)+16)
            im = im.crop((x,y,x+resize_to,y+resize_to))
        elif dataset_type=='imagenet-250':
            im.thumbnail((resize_to, resize_to), Image.ANTIALIAS)

        if np.random.random()<0.01:
            angle = np.random.choice([15, 30, 345])
            im = im.rotate(angle)

        if np.random.random()<0.5:
            bright_amount = np.random.random()*1.4+0.25
            bri_enhancer = ImageEnhance.Brightness(im)
            im = bri_enhancer.enhance(bright_amount)

        if np.random.random()<0.5:
            cont_amount = np.random.random()*1.4 + 0.25
            cont_enhancer = ImageEnhance.Contrast(im)
            im = cont_enhancer.enhance(cont_amount)

        if dataset_type!= 'svhn-10' and np.random.random()<0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        # normalization
        sample_img = self.normalize_img(np.asarray(im,dtype=np.uint8))

        if resize_to > 0 :
            assert sample_img.shape[0]==resize_to, "Image size (%d) doesnt match resize_to (%d)"%(sample_img.shape[0],resize_to)
        else:
            assert sample_img.shape[0] == image_size, "Image size (%d) doesnt match resize_to (%d)" % (
            sample_img.shape[0], resize_to)

        return sample_img,label


    def normalize_img(self, img_uint8):
        img = img_uint8.astype('float32')
        img -= np.mean(img)
        img /= max(np.std(img),1.0)
        return img