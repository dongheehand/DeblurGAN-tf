from PIL import Image
import numpy as np
import random
import math
import os

_rgb_to_YCbCr_kernel = [[65.738 / 256 , -37.945 / 256, 112.439 / 256],
                       [129.057 / 256, -74.494 / 256, -94.154 / 256],
                       [25.064 / 256, 112.439 / 256, -18.214 / 256]]


_YCbCr_to_rgb_kernel = [[298.082 / 256, 298.082 / 256, 298.082 / 256],
                       [0, -100.291 / 256, 516.412 / 256],
                       [408.583 / 256, -208.120 / 256, 0]]

def recursive_forwarding(LR, scale, chopSize, session, net_model, chopShave = 10):
    b, h, w, c = LR.shape
    wHalf = math.floor(w / 2)
    hHalf = math.floor(h / 2)
    
    wc = wHalf + chopShave
    hc = hHalf + chopShave
    
    inputPatch = np.array((LR[:, :hc, :wc, :], LR[:, :hc, (w-wc):, :], LR[:,(h-hc):,:wc,:], LR[:,(h-hc):,(w-wc):,:]))
    outputPatch = []

    if wc * hc < chopSize:
        for ele in inputPatch:
            output = session.run(net_model.output, feed_dict = {net_model.LR : ele})
            outputPatch.append(output)

    else:
        for ele in inputPatch:
            output = recursive_forwarding(ele, scale, chopSize, session, net_model, chopShave)
            outputPatch.append(output)

    w = scale * w
    wHalf = scale * wHalf
    wc = scale * wc
    
    h = scale * h
    hHalf = scale * hHalf
    hc = scale * hc
    
    chopShave = scale * chopSize
    
    upper = np.concatenate((outputPatch[0][:,:hHalf,:wHalf,:], outputPatch[1][:,:hHalf,wc-w+wHalf:,:]), axis = 2)
    rower = np.concatenate((outputPatch[2][:,hc-h+hHalf:,:wHalf,:], outputPatch[3][:,hc-h+hHalf:,wc-w+wHalf:,:]), axis = 2)
    output = np.concatenate((upper,rower),axis = 1)
    
    return output

def cpu_rgb_to_ycbcr(rgb_img):
    
    ycbcr = rgb_img.dot(_rgb_to_YCbCr_kernel)
    
    ycbcr[:,:,0] += 16
    ycbcr[:,:,1] += 128
    ycbcr[:,:,2] += 128
    

    ycbcr = np.round(ycbcr)
    ycbcr = np.clip(ycbcr, 0.0, 255.0)
    ycbcr = ycbcr.astype(np.uint8)
    
    return ycbcr

def cpu_psnr(GT, HR, edge = 0):
    
    GT = GT.astype(np.float32)
    HR = HR.astype(np.float32)
    
    if edge != 0 :
        GT = GT[edge:-edge, edge:-edge, :]
        HR = HR[edge:-edge, edge:-edge, :]
        
    MSE = np.mean((GT - HR) ** 2)
    
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(MSE)
    
    
def self_ensemble(args, model, sess, LR_img, is_recursive = False):
    
    input_img_list = []
    output_img_list = []
    for i in range(4):
        input_img_list.append(np.rot90(LR_img, i))
    input_img_list.append(np.fliplr(LR_img))
    input_img_list.append(np.flipud(LR_img))
    input_img_list.append(np.rot90(np.fliplr(LR_img)))
    input_img_list.append(np.rot90(np.flipud(LR_img)))
    
    
    for ele in input_img_list:
        
        input_img = np.expand_dims(ele, axis = 0)
        
        if is_recursive :
            output_img = recursive_forwarding(input_img, args.scale, args.chop_size, sess, model, args.chop_shave)
            output_img_list.append(output_img[0])
            
        else:
            output_img = sess.run(model.output, feed_dict = {model.LR : input_img})
            output_img_list.append(output_img[0])
            
    reshaped_output_img_list = []
    for i in range(4):
        output_img = output_img_list[i]
        output_img = np.rot90(output_img, 4-i)
        output_img = output_img.astype(np.float32)
        reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[4]
    output_img = np.fliplr(output_img)
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[5]
    output_img = np.flipud(output_img)
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[6]
    output_img = np.fliplr(np.rot90(output_img,3))
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[7]
    output_img = np.flipud(np.rot90(output_img,3))
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    overall_img = sum(reshaped_output_img_list) / 8
    overall_img = np.clip(np.round(overall_img), 0.0, 255.0)
    overall_img = overall_img.astype(np.uint8)
        
    return overall_img

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

