import sys
import os
import time
import pydicom
import numpy
import cv2
import scipy.ndimage

def analyze_series_flow_original(dicom_path):
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pixelArray = dcm.pixel_array
        if len(pixelArray.shape) == 4:
            pixelArray = pixelArray[:, :, :, 0]
        if len(pixelArray.shape) == 2:
            pixelArray = numpy.expand_dims(pixelArray, axis=0)
        n_slices = pixelArray.shape[0]
        
        if n_slices == 1:
            return 0, 0, 0, 0, 3, "Single image, TIMI 3 assumed."
            
        pa_f = pixelArray.astype(numpy.float32)
        pmin, pmax = pa_f.min(), pa_f.max()
        if pmax > pmin:
            pa_f = (pa_f - pmin) / (pmax - pmin) * 255.0

        scores = []
        for i in range(n_slices):
            frame = pa_f[i]
            small = cv2.resize(frame, (256, 256))
            blurred = scipy.ndimage.gaussian_filter(small.astype(float), 2)
            grad = numpy.abs(numpy.gradient(blurred))
            scores.append(numpy.sum(grad))
            
        scores = numpy.array(scores)
        best_ix = int(numpy.argmax(scores))
        
        if best_ix < 3 and n_slices > 10:
            alt_best = int(numpy.argmax(scores[3:])) + 3
            if scores[alt_best] > scores[best_ix] * 0.8:
                best_ix = alt_best
                
        end_ix = n_slices - 1
        
        baseline = numpy.mean(scores[:3]) if len(scores) > 3 else scores[0]
        threshold = baseline + numpy.std(scores) * 1.5
        
        start_ix = 0
        for i in range(n_slices):
            if scores[i] > threshold:
                start_ix = i
                break
                
        if start_ix >= best_ix:
            start_ix = max(0, best_ix - 15)
            
        tfc = best_ix - start_ix
        
        return best_ix, start_ix, end_ix, tfc
    except Exception as e:
        return None

def analyze_series_flow_optimized(dicom_path):
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pixelArray = dcm.pixel_array
        
        # 1. Handle color/screenshots correctly (same as load_dicom_data)
        if len(pixelArray.shape) == 4:
            pixelArray = pixelArray[:, :, :, 0]
        elif len(pixelArray.shape) == 3 and pixelArray.shape[2] == 3:
            pixelArray = pixelArray[numpy.newaxis, :, :, 0]
        elif len(pixelArray.shape) == 2:
            pixelArray = pixelArray[numpy.newaxis, ...]
            
        n_slices = pixelArray.shape[0]
        
        if n_slices == 1:
            return 0, 0, 0, 0
            
        pa_f = pixelArray.astype(numpy.float32)
        pmin, pmax = pa_f.min(), pa_f.max()
        if pmax > pmin:
            pa_f = (pa_f - pmin) / (pmax - pmin) * 255.0

        scores = []
        for i in range(n_slices):
            frame = pa_f[i]
            # Use 128x128 for speed (contrast features are large scale)
            small = cv2.resize(frame, (128, 128))
            # Use OpenCV GaussianBlur (much faster than scipy)
            blurred = cv2.GaussianBlur(small, (0, 0), 2)
            # Use OpenCV Sobel for gradient (much faster than numpy.gradient)
            grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            scores.append(numpy.sum(numpy.abs(grad_x) + numpy.abs(grad_y)))
            
        scores = numpy.array(scores)
        best_ix = int(numpy.argmax(scores))
        
        if best_ix < 3 and n_slices > 10:
            alt_best = int(numpy.argmax(scores[3:])) + 3
            if scores[alt_best] > scores[best_ix] * 0.8:
                best_ix = alt_best
                
        end_ix = n_slices - 1
        
        baseline = numpy.mean(scores[:3]) if len(scores) > 3 else scores[0]
        threshold = baseline + numpy.std(scores) * 1.5
        
        start_ix = 0
        for i in range(n_slices):
            if scores[i] > threshold:
                start_ix = i
                break
                
        if start_ix >= best_ix:
            start_ix = max(0, best_ix - 15)
            
        tfc = best_ix - start_ix
        
        return best_ix, start_ix, end_ix, tfc
    except Exception as e:
        print("Error:", e)
        return None

path = "/var/www/analiza-dicom/tailscale_cache/3002/3002-0002/3002-0002 angio1-dicomcleaner/DICOM/DICOM/I1"
print("Original:")
t0 = time.time()
res1 = analyze_series_flow_original(path)
print("Res:", res1, "in", time.time() - t0)

print("Optimized:")
t0 = time.time()
res2 = analyze_series_flow_optimized(path)
print("Res:", res2, "in", time.time() - t0)

path_color = "/var/www/analiza-dicom/tailscale_cache/3002/3002-0002/3002-0002 angio1-dicomcleaner/DICOM/DICOM/I21"
print("Color Original:")
t0 = time.time()
res_col1 = analyze_series_flow_original(path_color)
print("Res:", res_col1, "in", time.time() - t0)

print("Color Optimized:")
t0 = time.time()
res_col2 = analyze_series_flow_optimized(path_color)
print("Res:", res_col2, "in", time.time() - t0)
