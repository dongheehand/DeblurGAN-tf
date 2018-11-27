from PIL import Image
import numpy as np
import random
import math
import os

def image_loader(image_path, load_x, load_y, is_train = True):
    
    imgs = sorted(os.listdir(image_path))
    img_list = []
    for ele in imgs:
        img = Image.open(os.path.join(image_path, ele))
        if is_train:
            img = img.resize((load_x, load_y), Image.BICUBIC)
        img_list.append(np.array(img))
    
    return img_list

def data_augument(lr_img, hr_img, aug):
    
    if aug < 4:
        lr_img = np.rot90(lr_img, aug)
        hr_img = np.rot90(hr_img, aug)
    
    elif aug == 4:
        lr_img = np.fliplr(lr_img)
        hr_img = np.fliplr(hr_img)
        
    elif aug == 5:
        lr_img = np.flipud(lr_img)
        hr_img = np.flipud(hr_img)
        
    elif aug == 6:
        lr_img = np.rot90(np.fliplr(lr_img))
        hr_img = np.rot90(np.fliplr(hr_img))
        
    elif aug == 7:
        lr_img = np.rot90(np.flipud(lr_img))
        hr_img = np.rot90(np.flipud(hr_img))
        
    return lr_img, hr_img

def batch_gen(blur_imgs, sharp_imgs, patch_size, batch_size, random_index, step, augment):
    
    img_index = random_index[step * batch_size : (step + 1) * batch_size]
    
    all_img_blur = []
    all_img_sharp = []
    
    for _index in img_index:
        all_img_blur.append(blur_imgs[_index])
        all_img_sharp.append(sharp_imgs[_index])
    
    blur_batch = []
    sharp_batch = []
    
    for i in range(len(all_img_blur)):
        
        ih, iw, _ = all_img_blur[i].shape
        ix = random.randrange(0, iw - patch_size +1)
        iy = random.randrange(0, ih - patch_size +1)
        
        img_blur_in = all_img_blur[i][iy:iy + patch_size, ix:ix + patch_size]
        img_sharp_in = all_img_sharp[i][iy:iy + patch_size, ix:ix + patch_size]        
        
        if augment:
            
            aug = random.randrange(0,8)
            img_blur_in, img_sharp_in = data_augument(img_blur_in, img_sharp_in, aug)

        blur_batch.append(img_blur_in)
        sharp_batch.append(img_sharp_in)
        
    blur_batch = np.array(blur_batch)
    sharp_batch = np.array(sharp_batch)
    
    return blur_batch, sharp_batch


# In[ ]:


def recursive_forwarding(blur, chopSize, session, net_model, chopShave = 20):
    b, h, w, c = blur.shape
    wHalf = math.floor(w / 2)
    hHalf = math.floor(h / 2)
    
    wc = wHalf + chopShave
    hc = hHalf + chopShave
    
    inputPatch = np.array((blur[:, :hc, :wc, :], blur[:, :hc, (w-wc):, :], blur[:,(h-hc):,:wc,:], blur[:,(h-hc):,(w-wc):,:]))
    outputPatch = []
    if wc * hc < chopSize:
        for ele in inputPatch:
            output = session.run(net_model.output, feed_dict = {net_model.blur : ele})
            outputPatch.append(output)

    else:
        for ele in inputPatch:
            output = recursive_forwarding(ele, chopSize, session, net_model, chopShave)
            outputPatch.append(output)
    
    upper = np.concatenate((outputPatch[0][:,:hHalf,:wHalf,:], outputPatch[1][:,:hHalf,wc-w+wHalf:,:]), axis = 2)
    rower = np.concatenate((outputPatch[2][:,hc-h+hHalf:,:wHalf,:], outputPatch[3][:,hc-h+hHalf:,wc-w+wHalf:,:]), axis = 2)
    output = np.concatenate((upper,rower),axis = 1)
    
    return output

