import numpy as np
from PIL import Image,ImageEnhance
import multiprocessing as mp
from functools import partial


batch_size = 128
n_labels = 250
slice_size = batch_size * np.min(50,n_labels)
steps_per_slice = slice_size//batch_size//2
n_slices = train_size / slice_size

slice_index = 0
step_in_slice = 0


def generate_data_with_label_sequence(dataset_images,dataset_labels, label_sequence):
    '''
    Generating data with a specific label sequence
    :param label_sequence: A list of list where the size is [n_steps, n_labels]
    :return: A generator
    '''
    yield get_data_batch_for_labels(label_sequence)


def get_data_batch_for_labels(dataset_images, dataset_labels, label_sequence, dataset_info):
    global step_in_slice, steps_per_slice, slice_index

    dataset_name, resize_to, n_labels = dataset_info['dataset_name'],dataset_info['resize_to'],dataset_info['n_labels']

    part_augment_func = partial(get_augmented_sample_for, resize_to, dataset_name)
    image_slice = dataset_images[slice_index*slice_size:(slice_index+1)*slice_size,:,:,:]
    label_slice = dataset_labels[slice_index*slice_size:(slice_index+1)*slice_size,0]

    image_list = []
    for label in label_sequence:
        img_idx = np.random.choice(list(np.where(dataset_labels == label)[0].flatten()))
        image_list.append(image_slice[img_idx,:,:,:])

    # do not use all the CPUs if there are a lot only use half of them
    # if using all, leave one free
    cpu_count = mp.cpu_count() - 1 if mp.cpu_count() < 32 else mp.cpu_count() // 2
    pool = mp.Pool(cpu_count)
    print('Using %d CPU cores' % cpu_count)

    # Prepare data in parallel
    preprocessed_images = pool.map(part_augment_func,image_list,chunksize=len(label_sequence)//cpu_count)
    train_images = np.stack(preprocessed_images,axis=0)
    pool.close()
    pool.join()

    train_ohe_labels = (np.arange(n_labels) == label_slice[:]).astype(np.float32)
    step_in_slice = (step_in_slice+1)%steps_per_slice

    # whenever steps_per_slice number of steps completed,
    # increment the slice index by 1
    if step_in_slice == 0:
        slice_index = (slice_index+1)%n_slices

    return train_images,train_ohe_labels


def get_augmented_sample_for(image, resize_to, dataset_type):
    global image_abuse_threshold
    global save_gen_data,gen_save_count,gen_perist_dir

    # Convert image to 255 color and uint8 data type
    uint8_image = image - np.min(image)
    uint8_image = (uint8_image*255.0/np.max(uint8_image)).astype(np.uint8)

    image_size = uint8_image.shape[1]
    assert image_size == uint8_image.shape[2], "Image has different width and height"
    im = Image.fromarray(uint8_image)

    if dataset_type=='cifar-10' or dataset_type=='cifar-100':
        x, y = np.random.randint(0,6), np.random.randint(0,6)
        im = im.crop((x, y, x + resize_to, y + resize_to))

    if dataset_type=='imagenet-250':
        im.thumbnail((resize_to+32,resize_to+32), Image.ANTIALIAS)

    if dataset_type=='imagenet-250' and np.random.random()<0.25:
        center = image_size//2
        x,y = np.random.randint(center-(resize_to//2)-16,center-(resize_to//2)+16),np.random.randint(center-(resize_to//2)-16,center-(resize_to//2)+16)
        im = im.crop((x,y,x+resize_to,y+resize_to))
    elif dataset_type=='imagenet-250':
        im.thumbnail((resize_to, resize_to), Image.ANTIALIAS)

    if np.random.random()<0.01:
        angle = np.random.choice([15, 30, 345])
        im = im.rotate(angle)

    if np.random.random()<0.4:
        bright_amount = np.random.random()*1.4+0.25
        bri_enhancer = ImageEnhance.Brightness(im)
        im = bri_enhancer.enhance(bright_amount)

    if np.random.random()<0.4:
        cont_amount = np.random.random()*1.4 + 0.25
        cont_enhancer = ImageEnhance.Contrast(im)
        im = cont_enhancer.enhance(cont_amount)

    if dataset_type!= 'svhn-10' and np.random.random()<0.4:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    # normalization
    sample_img = normalize_img(np.asarray(im,dtype=np.uint8))

    assert sample_img.shape[0]==resize_to, "Image size (%d) doesnt match resize_to (%d)"%(sample_img.shape[0],resize_to)

    return sample_img


def normalize_img(img_uint8):
    img = img_uint8.astype('float32')
    return (img-np.mean(img))/max(np.std(img),1.0/img.size)