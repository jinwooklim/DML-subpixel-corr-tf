#!/usr/bin/env python
"""
Tests for the correlation Tensorflow operation.

.. moduleauthor:: Justin
"""

import unittest
import numpy as np
from scipy.misc import imresize
import tensorflow as tf
from correlation import correlation

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end=" ")
        print("")

class CorrelationOpTest(unittest.TestCase):

    # def test_raisesExceptionWithIncompatibleDimensions(self):
    #     ''' correlation only accepts 4-dimension tensors, with dimensions (batch_size,height,width,num_channels) '''
    #     with tf.device('/gpu:0'):
    #         with tf.Session(''):
    #             with self.assertRaises(ValueError):
    #                 cl.corr([1, 2], [[1, 2], [3, 4]], sigma=sigma, kernel_size=kernel_size).eval()
    #             with self.assertRaises(ValueError):
    #                 self.assertRaises(cl.corr([1, 2], [1, 2, 3, 4], sigma=sigma, kernel_size=kernel_size).eval(), ValueError)
    #             with self.assertRaises(ValueError):
    #                 self.assertRaises(cl.corr([1, 2, 3], [[1, 2], [3, 4]], sigma=sigma, kernel_size=kernel_size).eval(), ValueError)
    #
    #
    # def test_raisesExceptionWithTooManyOffsets(self):
    #     ''' correlation only accepts up to 2601==51x51  offsets '''
    #     with tf.device('/gpu:0'):
    #         with tf.Session(''):
    #             with self.assertRaises(ValueError):
    #                 # Current max_displacement/stride is 25
    #                 my_shape = (1,21,21,1)
    #                 a = np.ones((my_shape))
    #                 b = np.ones((my_shape))
    #                 self.assertRaises(cl.corr(a,b,stride=1,max_displacement=50, sigma=sigma, kernel_size=kernel_size),ValueError);


    def test_correlationHardCoded(self):
        with tf.device('/gpu:0'):
            with tf.Session('') as sess:
                batch_size = 1
                width = 64 # 3
                height = 48 # 3
                depth = 256
                stride_1 = 2
                stride_2 = 1
                kernel_size = 1
                max_displacement = 40 # 3
                padding = 40 # 3
    
                my_shape = [batch_size, width, height, depth]
                a = np.float32(np.ones(my_shape))
                b = np.float32(np.ones(my_shape))

                a = tf.convert_to_tensor(a, tf.float32)
                b = tf.convert_to_tensor(b, tf.float32)

                print("a : ", np.shape(a))
                print("b : ", np.shape(b))
    
                # ResizeMethod.NEAREST_NEIGHBOR
                a_resize = tf.image.resize_images(a, [(width * 2) - 1, (height * 2) - 1], method=1)
                # ResizeMethod.BILINEAR
                b_resize = tf.image.resize_images(b, [(width * 2) - 1, (height * 2) - 1], method=0)
    
                print("a_resize : ", np.shape(a_resize))
                print("b_resize : ", np.shape(b_resize))
    
                result = correlation(a_resize,
                                     b_resize,
                                     kernel_size,
                                     max_displacement,
                                     stride_1,
                                     stride_2,
                                     padding)#.eval() # Sub-pixel
    
                print("result : ", np.shape(result))

                grad_a = tf.gradients(result[:, :, :, 0], a)
                print(grad_a)
                # print(np.shape(sess.run(grad_a)))

                grad_b = tf.gradients(result[:, :, :, 0], b)
                print(grad_b)
                # print(np.shape(sess.run(grad_b)))
    
                # matprint(result[0, :, :, 0])
                # print("\n")
                # matprint(result[0, :, :, 1])
                # print("\n")


    # def test_correlationGradientAHardCoded(self):
    #     with tf.device('/gpu:0'):
    #         with tf.Session('') as sess:
    #             batch_size = 1
    #             width = 2 # 64
    #             height = 2 # 48
    #             depth = 1 # 256
    #             stride_1 = 2
    #             stride_2 = 1
    #             kernel_size = 1
    #             max_displacement = 1 # 20
    #             padding = 1 # 20

    #             expected_depth = (2 * int(max_displacement / stride_2) + 1)**2
    #             offsets = []
    #             for row_offset in np.arange(-max_displacement, max_displacement+1, stride_2):
    #                 for col_offset in np.arange(-max_displacement, max_displacement+1, stride_2):
    #                      offsets.append((row_offset, col_offset))

    #             my_shape = (batch_size, height, width, depth)

    #             # ResizeMethod.NEAREST_NEIGHBOR
    #             a = np.ones(my_shape, dtype=np.float32)
    #             a_resize = tf.image.resize_images(a, [(height * 2) - 1, (width * 2) - 1], method=1)

    #             # ResizeMethod.BILINEAR
    #             b = np.ones(my_shape, dtype=np.float32)
    #             b_resize = tf.image.resize_images(b, [(height * 2) - 1, (width * 2) - 1], method=0)

    #             result = correlation(a_resize,
    #                                  b_resize,
    #                                  kernel_size,
    #                                  max_displacement,
    #                                  stride_1,
    #                                  stride_2,
    #                                  padding)

    #             self.assertEqual(int(result.shape[3]), expected_depth)

    #             # Check if it's aligned at all offsets
    #             for offset_index, offset in enumerate(offsets):
    #                 result_slices  = result[:, :, :, offset_index]
    #                 grad_a = tf.gradients(result_slices, a_resize)
    #                 gradient_a = sess.run(grad_a)
    #                 row_offset = offset[0]
    #                 col_offset = offset[1]
    #                 a_row_begin = 0
    #                 a_row_end = height - row_offset
    #                 b_row_begin = row_offset
    #                 b_row_end = height

    #                 a_col_begin = 0
    #                 a_col_end   = width-col_offset
    #                 b_col_begin = col_offset
    #                 b_col_end   = width

    #                 if(row_offset < 0):
    #                    a_row_begin = -row_offset
    #                    a_row_end  = height
    #                    b_row_begin = 0
    #                    b_row_end = height + row_offset
    #                 if(col_offset < 0):
    #                    a_col_begin = -col_offset
    #                    a_col_end  = width
    #                    b_col_begin = 0
    #                    b_col_end = width + col_offset
    #                 final_height = a_row_end - a_row_begin
    #                 final_width = a_col_end - a_col_begin

    #                 print(np.shape(gradient_a)) # (1, 1, 3, 3, 1) # (1, 1, 95, 127, 256)
    #                 matprint(gradient_a[0][0, a_row_begin:a_row_end, a_col_begin:a_col_end, 0])
    #                 # np.testing.assert_array_equal(gradient_a[0][0, a_row_begin:a_row_end, a_col_begin:a_col_end, 0],
    #                 #                               2*np.ones((final_height, final_width)))


    # def test_correlationGradientAHardCodedLarge(self):
    #     with tf.device('/gpu:0'):
    #         with tf.Session('') as sess:
    #             batch_size = 1;
    #             height = 41;
    #             width = 41;
    #             depth = 1;
    #             stride = 4;
    #             max_displacement = 24;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             offsets = []
    #             for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                 for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                      offsets.append((row_offset,col_offset));
    #
    #             my_shape = (batch_size,height,width,depth)
    #             a = tf.placeholder(tf.float32, shape = my_shape)
    #             feed_a = np.ones(my_shape,dtype=np.float32)
    #             b = 2 * np.ones(my_shape,dtype=np.float32)
    #             # result = cl.corr(a, b,stride=stride,max_displacement=max_displacement)
    #             # print(result)
    #             #
    #             result = cl.corr(a, b, stride=stride, max_displacement=max_displacement, sigma=sigma, kernel_size=kernel_size)
    #             print(result)
    #             #
    #             self.assertEqual(int(result.shape[3]), expected_depth)
    #
    #             # Check if it's aligned at all offsets
    #             for offset_index,offset in enumerate(offsets):
    #                 result_slices  = result[:,:,:,offset_index]
    #                 grad_a = tf.gradients(result_slices,a);
    #                 gradient_a = sess.run(grad_a,feed_dict={a : feed_a});
    #                 row_offset = offset[0]
    #                 col_offset = offset[1]
    #                 a_row_begin = 0
    #                 a_row_end = height-row_offset
    #                 b_row_begin = row_offset
    #                 b_row_end = height
    #
    #                 a_col_begin = 0
    #                 a_col_end   = width-col_offset
    #                 b_col_begin = col_offset
    #                 b_col_end   = width
    #
    #                 if(row_offset < 0):
    #                    a_row_begin = -row_offset
    #                    a_row_end  = height
    #                    b_row_begin = 0
    #                    b_row_end = height+row_offset
    #                 if(col_offset < 0):
    #                    a_col_begin = -col_offset
    #                    a_col_end  = width
    #                    b_col_begin = 0
    #                    b_col_end = width+col_offset
    #                 final_height = a_row_end-a_row_begin
    #                 final_width = a_col_end-a_col_begin
    #                 np.testing.assert_array_equal(gradient_a[0][0,a_row_begin:a_row_end,a_col_begin:a_col_end,0], 2*np.ones((final_height,final_width)))


    # def test_correlationGradientBHardCoded(self):
    #     with tf.device('/gpu:0'):
    #         with tf.Session('') as sess:
    #             batch_size = 1;
    #             height = 21;
    #             width = 21;
    #             depth = 1;
    #             stride = 2;
    #             max_displacement = 20;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             stride = 2;
    #             max_displacement = 20;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             offsets = []
    #             for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                 for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                      offsets.append((row_offset,col_offset));
    #
    #             my_shape = (batch_size,height,width,depth)
    #             a = np.ones(my_shape,dtype=np.float32)
    #             feed_b = 2*np.ones(my_shape,dtype=np.float32)
    #             b = tf.placeholder(tf.float32, shape = my_shape)
    #             result = cl.corr(a, b, sigma=sigma, kernel_size=kernel_size)
    #
    #             # Check if it's aligned at all offsets
    #             for offset_index,offset in enumerate(offsets):
    #                 result_slices  = result[:,:,:,offset_index]
    #                 grad_b = tf.gradients(result_slices,b);
    #                 gradient_b = sess.run(grad_b,feed_dict={b : feed_b});
    #                 row_offset = offset[0]
    #                 col_offset = offset[1]
    #                 a_row_begin = 0
    #                 a_row_end = height-row_offset
    #                 b_row_begin = row_offset
    #                 b_row_end = height
    #
    #                 a_col_begin = 0
    #                 a_col_end   = width-col_offset
    #                 b_col_begin = col_offset
    #                 b_col_end   = width
    #
    #                 if(row_offset < 0):
    #                    a_row_begin = -row_offset
    #                    a_row_end  = height
    #                    b_row_begin = 0
    #                    b_row_end = height+row_offset
    #                 if(col_offset < 0):
    #                    a_col_begin = -col_offset
    #                    a_col_end  = width
    #                    b_col_begin = 0
    #                    b_col_end = width+col_offset
    #                 final_height = b_row_end-b_row_begin
    #                 final_width = b_col_end-b_col_begin
    #                 np.testing.assert_array_equal(gradient_b[0][0,b_row_begin:b_row_end,b_col_begin:b_col_end,0], np.ones((final_height,final_width)))


    # def test_correlationRandom(self):
    #     with tf.device('/gpu:0'):
    #         with tf.Session(''):
    #             batch_size = 1;
    #             height = 21;
    #             width = 21;
    #             depth = 1;
    #             my_shape = (batch_size,height,width,depth)
    #             stride = 2;
    #             max_displacement = 20;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             offsets = []
    #             for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                 for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                      offsets.append((row_offset,col_offset));
    #
    #             for i in range(10):
    #                 a_rand = np.random.randint(10, size = my_shape)
    #                 b_rand = np.random.randint(10, size = my_shape)
    #
    #                 result = cl.corr(a_rand, b_rand,stride=stride,max_displacement=max_displacement, sigma=sigma, kernel_size=kernel_size).eval()
    #                 self.assertEqual(result.shape[3], expected_depth)
    #                 for offset_index, offset in enumerate(offsets):
    #                     row_offset = offset[0]
    #                     col_offset = offset[1]
    #                     a_row_begin = 0
    #                     a_row_end = height-row_offset
    #                     b_row_begin = row_offset
    #                     b_row_end = height
    #
    #                     a_col_begin = 0
    #                     a_col_end   = width-col_offset
    #                     b_col_begin = col_offset
    #                     b_col_end   = width
    #
    #                     if(row_offset < 0):
    #                        a_row_begin = -row_offset
    #                        a_row_end  = height
    #                        b_row_begin = 0
    #                        b_row_end = height+row_offset
    #                     if(col_offset < 0):
    #                        a_col_begin = -col_offset
    #                        a_col_end  = width
    #                        b_col_begin = 0
    #                        b_col_end = width+col_offset
    #                     a_slice = a_rand[:,a_row_begin:a_row_end,a_col_begin:a_col_end,:]
    #                     b_slice = b_rand[:,b_row_begin:b_row_end,b_col_begin:b_col_end,:];
    #                     result_rand_full = a_slice*b_slice
    #                     result_rand = np.sum(result_rand_full,axis=-1)/depth
    #                     np.testing.assert_array_equal(result[0,a_row_begin:a_row_end,a_col_begin:a_col_end,offset_index], result_rand[0,:,:])


    # def test_correlationGradientARandom(self):
    #     with tf.device('/gpu:0'):
    #         with tf.Session('') as sess:
    #             batch_size = 1;
    #             height = 21;
    #             width = 21;
    #             depth = 1;
    #             stride = 2;
    #             max_displacement = 20;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             stride = 2;
    #             max_displacement = 20;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             offsets = []
    #             for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                 for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                      offsets.append((row_offset,col_offset));
    #
    #             my_shape = (batch_size,height,width,depth)
    #             a = tf.placeholder(tf.float32, shape = my_shape)
    #             feed_a = np.random.randint(10,size=my_shape).astype(np.float32)
    #             b = 2 * np.random.randint(10,size=my_shape).astype(np.float32)
    #             result = cl.corr(a, b, kernel_size=kernel_size, sigma=sigma)
    #
    #             # Check if it's aligned at all offsets
    #             for offset_index,offset in enumerate(offsets):
    #                 row_offset = offset[0]
    #                 col_offset = offset[1]
    #                 a_row_begin = 0
    #                 a_row_end = height-row_offset
    #                 b_row_begin = row_offset
    #                 b_row_end = height
    #
    #                 a_col_begin = 0
    #                 a_col_end   = width-col_offset
    #                 b_col_begin = col_offset
    #                 b_col_end   = width
    #
    #                 if(row_offset < 0):
    #                    a_row_begin = -row_offset
    #                    a_row_end  = height
    #                    b_row_begin = 0
    #                    b_row_end = height+row_offset
    #                 if(col_offset < 0):
    #                    a_col_begin = -col_offset
    #                    a_col_end  = width
    #                    b_col_begin = 0
    #                    b_col_end = width+col_offset
    #                 result_slice = result[:,:,:,offset_index]
    #                 grad_a = tf.gradients(result_slice,a);
    #                 gradient_a = sess.run(grad_a,feed_dict={a : feed_a});
    #                 np.testing.assert_array_equal(gradient_a[0][0,a_row_begin:a_row_end,a_col_begin:a_col_end,0], b[0,b_row_begin:b_row_end,b_col_begin:b_col_end,0])


    # def test_correlationGradientBRandom(self):
    #     with tf.device('/gpu:0'):
    #         with tf.Session('') as sess:
    #             batch_size = 1;
    #             height = 21;
    #             width = 21;
    #             depth = 1;
    #             stride = 2;
    #             max_displacement = 20;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             stride = 2;
    #             max_displacement = 20;
    #             expected_depth = (2*int(max_displacement/stride)+1)**2;
    #             offsets = []
    #             for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                 for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
    #                      offsets.append((row_offset,col_offset));
    #
    #             my_shape = (batch_size,height,width,depth)
    #             a = np.random.randint(10,size=my_shape).astype(np.float32)
    #             feed_b = np.random.randint(10,size=my_shape).astype(np.float32)
    #             b = tf.placeholder(tf.float32, shape = my_shape)
    #             result = cl.corr(a, b, sigma=sigma, kernel_size=kernel_size)
    #
    #             # Check if it's aligned at all offsets
    #             for offset_index,offset in enumerate(offsets):
    #                 row_offset = offset[0]
    #                 col_offset = offset[1]
    #                 a_row_begin = 0
    #                 a_row_end = height-row_offset
    #                 b_row_begin = row_offset
    #                 b_row_end = height
    #
    #                 a_col_begin = 0
    #                 a_col_end   = width-col_offset
    #                 b_col_begin = col_offset
    #                 b_col_end   = width
    #
    #                 if(row_offset < 0):
    #                    a_row_begin = -row_offset
    #                    a_row_end  = height
    #                    b_row_begin = 0
    #                    b_row_end = height+row_offset
    #                 if(col_offset < 0):
    #                    a_col_begin = -col_offset
    #                    a_col_end  = width
    #                    b_col_begin = 0
    #                    b_col_end = width+col_offset
    #                 result_slice = result[:,:,:,offset_index]
    #                 grad_b = tf.gradients(result_slice,b);
    #                 gradient_b = sess.run(grad_b,feed_dict={b : feed_b});
    #                 np.testing.assert_array_equal(a[0,a_row_begin:a_row_end,a_col_begin:a_col_end,0], gradient_b[0][0,b_row_begin:b_row_end,b_col_begin:b_col_end,0])

                
if __name__ == '__main__':
    suite=unittest.TestLoader().loadTestsFromTestCase(CorrelationOpTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
