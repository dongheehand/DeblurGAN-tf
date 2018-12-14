import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util
from skimage.measure import compare_ssim as ssim



def train(args, model, sess, saver):
    
    if args.fine_tuning :
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for fine-tuning!")
        print("model path is %s"%(args.pre_trained_model))
        
    num_imgs = len(os.listdir(args.train_Sharp_path))
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs',sess.graph)
    if args.test_with_train:
        f = open("valid_logs.txt", 'w')
    
    epoch = 0
    step = num_imgs // args.batch_size
    
    if args.in_memory:
        
        blur_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y)
        sharp_imgs = util.image_loader(args.train_Sharp_path, args.load_X, args.load_Y)
        
        while epoch < args.max_epoch:
            random_index = np.random.permutation(len(blur_imgs))
            for k in range(step):
                s_time = time.time()
                blur_batch, sharp_batch = util.batch_gen(blur_imgs, sharp_imgs, args.patch_size, args.batch_size, random_index, k, args.augmentation)
                
                for t in range(args.critic_updates):
                    _, D_loss = sess.run([model.D_train, model.D_loss], feed_dict = {model.blur : blur_batch, model.sharp : sharp_batch, model.epoch : epoch})
                    
                _, G_loss = sess.run([model.G_train, model.G_loss], feed_dict = {model.blur : blur_batch, model.sharp : sharp_batch, model.epoch : epoch})
                             
                e_time = time.time()
            
            if epoch % args.log_freq == 0:
                summary = sess.run(merged, feed_dict = {model.blur : blur_batch, model.sharp: sharp_batch})
                train_writer.add_summary(summary, epoch)
                if args.test_with_train:
                    test(args, model, sess, saver, f, epoch, loading = False)
                print("%d training epoch completed" % epoch)
                print("D_loss : %0.4f, \t G_loss : %0.4f"%(D_loss, G_loss))
                print("Elpased time : %0.4f"%(e_time - s_time))
            if ((epoch) % args.model_save_freq ==0):
                saver.save(sess, './model/DeblurrGAN', global_step = epoch, write_meta_graph = False)
            
            epoch += 1

        saver.save(sess, './model/DeblurrGAN_last', write_meta_graph = False)
    
    else:
        while epoch < args.max_epoch:
            
            sess.run(model.data_loader.init_op['tr_init'])
            
            for k in range(step):
                s_time = time.time()
                
                for t in range(args.critic_updates):
                    _, D_loss = sess.run([model.D_train, model.D_loss], feed_dict = {model.epoch : epoch})
                    
                _, G_loss = sess.run([model.G_train, model.G_loss], feed_dict = {model.epoch : epoch})
                             
                e_time = time.time()
            
            if epoch % args.log_freq == 0:
                summary = sess.run(merged)
                train_writer.add_summary(summary, epoch)
                if args.test_with_train:
                    test(args, model, sess, saver, f, epoch, loading = False)
                print("%d training epoch completed" % epoch)
                print("D_loss : %0.4f, \t G_loss : %0.4f"%(D_loss, G_loss))
                print("Elpased time : %0.4f"%(e_time - s_time))
            if ((epoch) % args.model_save_freq ==0):
                saver.save(sess, './model/DeblurrGAN', global_step = epoch, write_meta_graph = False)
            
            epoch += 1

        saver.save(sess, './model/DeblurrGAN_last', global_step = epoch, write_meta_graph = False)
        
    if args.test_with_train:
        f.close()
        
        
def test(args, model, sess, saver, file, step = -1, loading = False):
        
    if loading:
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for test!")
        print("model path is %s"%args.pre_trained_model)
        
    blur_img_name = sorted(os.listdir(args.test_Blur_path))
    sharp_img_name = sorted(os.listdir(args.test_Sharp_path))
    
    PSNR_list = []
    ssim_list = []
    
    if args.in_memory :
        
        blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train = False)
        sharp_imgs = util.image_loader(args.test_Sharp_path, args.load_X, args.load_Y, is_train = False)
        
        for i, ele in enumerate(blur_imgs):
            blur = np.expand_dims(ele, axis = 0)
            sharp = np.expand_dims(sharp_imgs[i], axis = 0)
            output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim], feed_dict = {model.blur : blur, model.sharp : sharp})
            if args.save_test_result:
                output = Image.fromarray(output[0])
                split_name = blur_img_name[i].split('.')
                output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))

            PSNR_list.append(psnr)
            ssim_list.append(ssim)

    else:
        
        sess.run(model.data_loader.init_op['val_init'])

        for i in range(len(blur_img_name)):
            
            output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim])
            
            if args.save_test_result:
                output = Image.fromarray(output[0])
                split_name = blur_img_name[i].split('.')
                output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))
                
            PSNR_list.append(psnr)
            ssim_list.append(ssim)
            
    length = len(PSNR_list)
    
    mean_PSNR = sum(PSNR_list) / length
    mean_ssim = sum(ssim_list) / length
    
    if step == -1:
        file.write('PSNR : 0.4f SSIM : %0.4f'%(mean_PSNR, mean_ssim))
        file.close()
        
    else :
        file.write("%d-epoch step PSNR : %0.4f SSIM : %0.4f \n"%(step, mean_PSNR, mean_ssim))


            
def test_only(args, model, sess, saver):
    
    saver.restore(sess,args.pre_trained_model)
    print("saved model is loaded for test only!")
    print("model path is %s"%args.pre_trained_model)
    
    blur_img_name = sorted(os.listdir(args.test_Blur_path))

    if args.in_memory :
        
        blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train = False)
        
        for i, ele in enumerate(blur_imgs):
            blur = np.expand_dims(ele, axis = 0)
            
            if args.chop_forward:
                output = util.recursive_forwarding(blur, args.chop_size, sess, model, args.chop_shave)
                output = Image.fromarray(output[0])
            
            else:
                output = sess.run(model.output, feed_dict = {model.blur : blur})
                output = Image.fromarray(output[0])
            
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))

    else:
        
        sess.run(model.data_loader.init_op['te_init'])

        for i in range(len(blur_img_name)):
            output = sess.run(model.output)
            output = Image.fromarray(output[0])
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))    

