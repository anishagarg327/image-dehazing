import cv2;
import math;
import numpy as np;

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def Dehaze(fn):
    src = cv2.imread(fn);
    I = src.astype('float64')/255;
    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    J = Recover(I,t,A,0.1);
    return J

def SaveImage(J):
    cv2.imwrite("./image/image_dehazed.jpg",J*255);

def DehazeVideo(frame):
    I = frame.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(frame, te)
    J = Recover(I, t, A, 0.1)
    return (J * 255).astype('uint8')

def get_ffmpeg_exe():
    import shutil
    bin_path = shutil.which('ffmpeg')
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'

def ProcessVideo(input_path, output_path):
    ffmpeg_bin = get_ffmpeg_exe()
    import subprocess
    import tempfile
    import os

    # Create a temporary directory for frame processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames using ffmpeg
        extract_cmd = [
            ffmpeg_bin, '-i', input_path,
            '-vf', 'scale=min(640\\,iw):-2,fps=15',
            os.path.join(temp_dir, 'frame_%04d.png')
        ]
        subprocess.run(extract_cmd, check=True)

        # Get the list of frames
        frames = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
        
        # Process each frame
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, frame)
            # Read the frame
            img = cv2.imread(frame_path)
            # Process the frame
            processed = DehazeVideo(img)
            # Save the processed frame
            cv2.imwrite(frame_path, processed)
            print(f"Processing frame {i+1}/{len(frames)}")

        # Combine frames back into video using ffmpeg
        if len(frames) > 90:
            for extra in frames[90:]:
                try:
                    os.remove(os.path.join(temp_dir, extra))
                except Exception:
                    pass
            frames = frames[:90]

        combine_cmd = [
            ffmpeg_bin, '-y',
            '-framerate', '15',
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'veryfast',
            output_path
        ]
        subprocess.run(combine_cmd, check=True)

if __name__ == '__main__':
    import sys

    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        input_path = "./video/input.mp4"
        output_path = "./video/output.mp4"

    ProcessVideo(input_path, output_path)
    
