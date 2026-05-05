'''
 Pre processing of libraries Stage
    import tensorrt
    import pycuda
    import cv2
    import numpy as np and etc
   
'''
import tensorrt as trt
import pycuda.autoinit

import pycuda.driver as cuda
import cv2

import numpy as np

import os
from PIL import Image


import matplotlib.pyplot as plt


'''
    class name :TRTInference
    INIT: self.logger
    params [1]: engine_path, context
    params [2]: input shape and output shape, class_labels
    
'''
class TRTInference:

    def __init__(self, engine_file_path, input_shape, output_shape, class_labels_file):

        self.logger = trt.Logger(trt.Logger.WARNING)

        self.engine_file_path = engine_file_path

        # load engine 

        self.engine = self.load_engine(self.engine_file_path)

        # init context 
        self.context = self.engine.create_execution_context()


        self.input_shape = input_shape

        self.output_shape = output_shape

        self.class_labels_file = class_labels_file

        # open class file

        with open(class_labels_file, 'r') as class_read:
            self.class_labels = [line.strip() for line in class_read.readlines()]
        
    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_deserialized = runtime.deserialize_cuda_engine(f.read())
        return engine_deserialized
    

    '''
    param [1] : image path
    results(return) : img_list , img_path
    img resolition : 1, 3 224, 224,
    '''
    def preprocess_img(self, image_path):
        img_list = []

        img_path = []

        for img_original in os.listdir(image_path):
            if img_original.endswith('.jpg') or img_original.endswith('.png') or img_original.endswith('.jpeg'):
                img_full_path = os.path.join(image_path, img_original)
                # open image
                image = Image.open(img_full_path).convert('RGB')
                # resize img
                img = image.resize((self.input_shape[2], self.input_shape[3]), Image.NEAREST)
                img_np = np.array(img).astype(np.float32) / 255.0
                img_np = img_np.transpose((2,0,1))
                img_np = np.expand_dims(img_np, axis=0)
                # Visualize and save preprocessed image (as uint8 for display)
                img_disp = (img_np[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                plt.imshow(img_disp)
                plt.title(f'Preprocessed: {img_original}')
                plt.axis('off')
                plt.savefig(f'preprocessed_{os.path.splitext(img_original)[0]}.png')
                plt.close()
                img_list.append(img_np)
                img_path.append(img_full_path)
        return img_list, img_path
    
    # processing labels with corresponding images
    def postprocess_img(self, outputs, confidence_threshold=0.2):
        # outputs shape: (batch, 1000)
        classes_indices = []
        for output in outputs:
            # Apply softmax to get probabilities
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / np.sum(exp_output)
            # Top-5 predictions
            top5_idx = np.argsort(probs)[::-1][:5]
            print("Top-5 predictions:")
            for idx in top5_idx:
                print(f"  {self.class_labels[idx]}: {probs[idx]*100:.2f}%")
            # Plot bar chart for top-5
            top5_labels = [self.class_labels[idx] for idx in top5_idx]
            top5_scores = [probs[idx] for idx in top5_idx]
            plt.figure(figsize=(8, 4))
            plt.barh(top5_labels[::-1], [s*100 for s in top5_scores[::-1]])
            plt.xlabel('Probability (%)')
            plt.title('Top-5 Output Probabilities')
            plt.tight_layout()
            plt.savefig('output_top5_bar.png')
            plt.close()
            class_idx = np.argmax(probs)
            confidence = probs[class_idx]
            if confidence >= confidence_threshold:
                print(f"Class Detected: {self.class_labels[class_idx]} (confidence: {confidence*100:.2f}%)")
                classes_indices.append(self.class_labels[class_idx])
            else:
                print(f"No confident prediction (max confidence: {confidence*100:.2f}%)")
                classes_indices.append('Uncertain')
        return classes_indices
    
    # inference detection 
    ''' param [0] = self
        param [1] = image_path
        target: Inference Detection on GPU Local
    ''' 
    def inference_detection(self, image_path):
        # list 

        input_list, full_img_paths = self.preprocess_img(image_path)

        results = []

        confidence_threshold = 0.2  # You can adjust this value as needed
        import time
        import subprocess
        print("[INFO] Checking GPU status before inference...")
        try:
            smi_output = subprocess.check_output(['nvidia-smi']).decode()
            print("[nvidia-smi output before inference]:\n", smi_output)
        except Exception as e:
            print("[WARNING] Could not run nvidia-smi:", e)
        for inputs, full_img_path in zip(input_list, full_img_paths):
            inputs = np.ascontiguousarray(inputs)
            outputs = np.empty(self.output_shape, dtype=np.float32)
            d_inputs = cuda.mem_alloc(inputs.nbytes)
            d_outputs = cuda.mem_alloc(outputs.nbytes)
            bindings = [int(d_inputs), int(d_outputs)]
            start = time.time()
            try:
                cuda.memcpy_htod(d_inputs, inputs)
                self.context.execute_v2(bindings)
                cuda.memcpy_dtoh(outputs, d_outputs)
                end = time.time()
                print(f"TensorRT inference time for {os.path.basename(full_img_path)}: {end - start:.4f} seconds")
                print("[INFO] Checking GPU status after inference...")
                try:
                    smi_output = subprocess.check_output(['nvidia-smi']).decode()
                    print("[nvidia-smi output after inference]:\n", smi_output)
                except Exception as e:
                    print("[WARNING] Could not run nvidia-smi:", e)
                result = self.postprocess_img(outputs, confidence_threshold=confidence_threshold)
                results.append(result)
                self.display_recognized_images(full_img_path, result)
            finally:
                d_inputs.free()
                d_outputs.free()
        return results
    
    '''
        param[0] : image_path
        param[1] : class_label
        target: Displaying and Saving detected images
    '''
    def display_recognized_images(self, image_path, class_label):
        
        image = Image.open(image_path) 

        for class_name in class_label:
            
            # create one directory for detected images
            path_to_detected_imgs = "images_detected"

            # check path existence

            if not os.path.exists(path_to_detected_imgs):
                os.makedirs(path_to_detected_imgs)

            plt.imshow(image)

            plt.title(f'Recognized Image : {class_name}')

            plt.axis('off')

            save_img = os.path.join(path_to_detected_imgs, f'{class_name}.jpg')

            plt.savefig(save_img)

            plt.close()

            return image
        
engine_file_path = 'resnet.engine'

input_shape = (1, 3, 224, 224)

output_shape = (1, 1000)

class_labels = 'imagenet_classes.txt'

path_to_org_images = 'images'



inference = TRTInference(engine_file_path, input_shape, output_shape, class_labels)

inference.inference_detection(path_to_org_images)