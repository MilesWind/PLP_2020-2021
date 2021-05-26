print('loading imports');

import numpy as np;
import random
import time;
import copy;
import os;

from pathlib import Path;
import PIL;

import tensorflow as tf;

from tensorflow.keras.models import Sequential, load_model;

from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape;
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D;
from tensorflow.keras.layers import LeakyReLU, Dropout;
from tensorflow.keras.layers import BatchNormalization;

from tensorflow.keras.initializers import RandomUniform;

from tensorflow.keras.preprocessing.image import load_img;
from tensorflow.keras.preprocessing.image import save_img;
from tensorflow.keras.preprocessing.image import img_to_array;
from tensorflow.keras.preprocessing.image import array_to_img;

from tensorflow.keras.optimizers import Adam, RMSprop;

from tensorflow.keras.initializers import RandomNormal;

def load_data():

    data = [];
    directory = os.fsencode('Training');

    for file in os.listdir(directory):
        filename = os.fsdecode(file);
        if filename == '.DS_Store':
            continue;

        try:
            image = load_img("Training/"+filename);
            image = img_to_array(image);
            data.append(image);
        except:
            pass;

    return data;
    

class GAN(object):
    def __init__(self, img_rows, img_cols, channel, depth):

        self.img_rows = img_rows;
        self.img_cols = img_cols;
        self.channel = channel;
        self.depth = depth;

        self.k_init = RandomNormal(mean=0., stddev=1.);
        self.b_init = RandomNormal(mean=0., stddev=1.);
        
        self.D = None;
        self.G = None;
        self.AM = None;
        self.DM = None;

    
    def discriminator(self):
        if self.D:
            return self.D;
        self.D = Sequential();
        depth = self.depth;
        dropout = 0.4;

        input_shape = (self.img_rows, self.img_cols, self.channel);
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'));
        self.D.add(LeakyReLU(alpha=0.2));
        self.D.add(Dropout(dropout));

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'));
        self.D.add(LeakyReLU(alpha=0.2));
        self.D.add(Dropout(dropout));

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'));
        self.D.add(LeakyReLU(alpha=0.2));
        self.D.add(Dropout(dropout));

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'));
        self.D.add(LeakyReLU(alpha=0.2));
        self.D.add(Dropout(dropout));

        self.D.add(Flatten());
        self.D.add(Dense(1));
        self.D.add(Activation('sigmoid'));
        
        return self.D;

    def generator(self):
        if self.G:
            return self.G;
        self.G = Sequential();
        dropout = 0.4;


        self.G.add(Dense(int(self.img_rows/32)*int(self.img_cols/32)*self.depth, input_dim=100));
        self.G.add(BatchNormalization(momentum=0.9));
        self.G.add(Activation('relu'));
        self.G.add(Reshape((int(self.img_rows/32), int(self.img_cols/32), self.depth)));
        self.G.add(Dropout(dropout));

        self.G.add(UpSampling2D());
        self.G.add(Conv2DTranspose(int(self.depth/2), 5, padding='same'));
        self.G.add(BatchNormalization(momentum=0.9));
        self.G.add(Activation('relu'));


        self.G.add(UpSampling2D());
        self.G.add(Conv2DTranspose(int(self.depth/4), 5, padding='same'));
        self.G.add(BatchNormalization(momentum=0.9));
        self.G.add(Activation('relu'));

        self.G.add(UpSampling2D());
        self.G.add(Conv2DTranspose(int(self.depth/8), 5, padding='same'));
        self.G.add(BatchNormalization(momentum=0.9));
        self.G.add(Activation('relu'));

        self.G.add(UpSampling2D());
        self.G.add(Conv2DTranspose(int(self.depth/16), 5, padding='same'));
        self.G.add(BatchNormalization(momentum=0.9));
        self.G.add(Activation('relu'));

        self.G.add(UpSampling2D());
        self.G.add(Conv2DTranspose(int(self.depth/32), 5, padding='same'));
        self.G.add(BatchNormalization(momentum=0.9));
        self.G.add(Activation('relu'));

        self.G.add(Conv2DTranspose(3, 5, padding='same'));
        self.G.add(Activation('sigmoid'));
        
        return self.G;

    def discriminator_model(self):
        if self.DM:
            return self.DM;
        optimizer = RMSprop(learning_rate=0.01, clipvalue=1.0, decay=0);
        self.DM = Sequential();
        self.DM.add(self.discriminator());
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']);
        return self.DM;

    def adversarial_model(self):
        if self.AM:
            return self.AM;
        optimizer = RMSprop(learning_rate=0.005, clipvalue=1.0, decay=0);
        self.AM = Sequential();
        self.AM.add(self.generator());
        self.AM.add(self.discriminator());
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']);
        return self.AM;

class Face_GAN:
    def __init__(self, load=False):
        self.img_rows = 256;
        self.img_cols = 256;
        self.channel = 3;
        self.depth = 32;
                
        self.GAN = GAN(self.img_rows, self.img_cols, self.channel, self.depth);
        print('\nGAN setup succeded\n');
        if load:
            self.Load();
            print('\nmodel load succeded\n');
        self.discriminator = self.GAN.discriminator_model();
        print('\ndiscriminator setup succeded\n');
        self.adversarial = self.GAN.adversarial_model();
        print('\nadversarial setup succeded\n');
        self.generator = self.GAN.generator();
        print('\ngenerator setup succeded\n');

    def Save(self):
        e = self.GAN.discriminator();
        e.save(Path('model_storage/discriminator'));
        e = self.GAN.generator();
        e.save(Path('model_storage/generator'));
        e = self.GAN.discriminator_model();
        e.save(Path('model_storage/discriminator_model'));
        e = self.GAN.adversarial_model();
        e.save(Path('model_storage/adversarial_model'));

    def Load(self):
        self.GAN.D = load_model(Path('model_storage/discriminator'));
        self.GAN.G = load_model(Path('model_storage/generator'));
        self.GAN.AM = load_model(Path('model_storage/adversarial_model'));
        self.GAN.DM = load_model(Path('model_storage/discriminator_model'));

    def Generate(self, str_path, noise=None):
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[1, 100]);
            save_img(str_path, array_to_img(np.squeeze(self.generator.predict(noise))));
        else:
            save_img(str_path, array_to_img(np.squeeze(self.generator.predict(noise))));
        
    def Train(self, train_steps=999999999, data_width=100, save_interval=10):

        print('\nloading data...');
        data = load_data();
        
        for i in range(train_steps):
            e = ElapsedTimer();

            #images for discriminator:
            print('\ngenerating images...');
            
            noise = np.random.uniform(-1.0, 1.0, size=[data_width, 100]);
            images_fake = self.generator.predict(noise);
            
            # training discriminator:
            print('\ntraining discriminator...');
            
            x = np.concatenate((random.sample(data, data_width), images_fake));
            y = np.ones([2*data_width, 1]);
            y[data_width:, :] = 0;
            
            d_loss = self.discriminator.train_on_batch(x, y);


            # train data reformatting for adversarial:
            print('\nreformating data...');
            
            y = np.ones([data_width, 1]);
            noise = np.random.uniform(-1.0, 1.0, size=[data_width, 100]);
            
            # training adversarial:
            print('\ntraining adversarial...');
            
            a_loss = self.adversarial.train_on_batch(noise, y);
            

            log_msg = "%d: [D loss: %f, acc: %f]" % (i+1, d_loss[0], d_loss[1]);
            log_msg = "%s: [A loss: %f, acc: %f]" % (log_msg, a_loss[0], a_loss[1]);
            print(log_msg);

            e.elapsed_time();
            f.Generate('output/output-%s.jpg' % str(i+1), np.ones((1, 100)));
    
            if save_interval > 0:
                if (i+1) % save_interval == 0:
                    print('\nsaving model...');
                    self.Save();



class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time();
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec";
        elif sec < (60 * 60):
            return str(sec / 60) + " min";
        else:
            return str(sec / (60 * 60)) + " hr";
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time));




































































