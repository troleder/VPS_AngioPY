import sys
import os
import time
import pydicom
import numpy
import cv2
import scipy.ndimage

def analyze_series_flow(dicom_path, file_size=None):
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
        
        return best_ix, start_ix, end_ix, tfc, 3, "Assumed TIMI 3"
    except Exception as e:
        return 0, 0, 0, 0, 0, f"Error: {e}"

folder_path = "/var/www/analiza-dicom/tailscale_cache/3002/3002-0002"
dicom_files = []
for root, dirs, files in os.walk(folder_path):
    for f in files:
        if f.startswith('.'):
            continue
        if f.lower().endswith(('.zip', '.rar', '.bmp', '.dat', '.log', '.angioframes', '.coreg', '.oct', '.params', '.dbf', '.png', '.jpg', '.txt', '.pdf', '.xlsx', '.docx', '.exe', '.dll', '.ini', '.sys')):
            continue
        dicom_files.append(os.path.join(root, f))

print(f"Found {len(dicom_files)} DICOM files.")
total_time = 0
for idx, path in enumerate(dicom_files):
    t0 = time.time()
    res = analyze_series_flow(path)
    dt = time.time() - t0
    total_time += dt
    print(f"File {idx+1}/{len(dicom_files)}: {os.path.basename(path)} ({os.path.getsize(path)/(1024*1024):.2f} MB) -> Res: {res} in {dt:.3f}s")

print(f"Total time for {len(dicom_files)} files: {total_time:.2f}s")
