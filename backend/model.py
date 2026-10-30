import torch
import torch.nn as nn
import torchvision.transforms as tfs
import cv2
import numpy as np
from PIL import Image
import io

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FFA(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1,res2,res3],dim=1))
        w = w.view(-1,self.gps, self.dim)[:,:,:,None,None]
        out = w[:,0,::] * res1 + w[:,1,::] * res2 + w[:,2,::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1

class VideoProcessor:
    def __init__(self, model_path):
        self.net = FFA(gps=3, blocks=12)
        self.net = nn.DataParallel(self.net)
        self.net = self.net.to(device)
        ckp = torch.load(model_path, map_location=device, weights_only=False)
        self.net.load_state_dict(ckp['model'])
        self.net.eval()

    def process_frame(self, frame, is_video=False):
        orig_h, orig_w = frame.shape[:2]
        # Video frames on CPU need smaller dimension for fast processing
        max_dim = (320 if is_video else 480) if device == 'cpu' else 1280
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            w, h = (int(orig_w * scale) // 4) * 4, (int(orig_h * scale) // 4) * 4
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        else:
            w, h = orig_w, orig_h
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        transform = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
        ])
        frame = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            dehazed = self.net(frame)

        dehazed = dehazed.squeeze(0).cpu()
        dehazed = dehazed.permute(1, 2, 0).numpy() * 255
        dehazed = dehazed.astype(np.uint8)
        dehazed = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
        if (w, h) != (orig_w, orig_h):
            dehazed = cv2.resize(dehazed, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return dehazed

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Unable to read image")

        dehazed = self.process_frame(frame)
        return dehazed

    def process_video(self, video_bytes):
        import subprocess
        import tempfile
        import os

        # Create temporary files
        temp_input = 'temp_input.mp4'
        temp_output = 'temp_output.mp4'

        try:
            # Save input video bytes to temporary file
            with open(temp_input, 'wb') as f:
                f.write(video_bytes)

            # Create a temporary directory for frame processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract frames using ffmpeg
                extract_cmd = [
                    get_ffmpeg_exe(), '-i', temp_input,
                    os.path.join(temp_dir, 'frame_%d.png')
                ]
                subprocess.run(extract_cmd, check=True)

                # Get the list of frames
                frames = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
                total_frames = len(frames)

                # Process each frame
                for i, frame in enumerate(frames, 1):
                    frame_path = os.path.join(temp_dir, frame)
                    # Read the frame
                    img = cv2.imread(frame_path)
                    if img is None:
                        raise ValueError(f"Unable to read frame: {frame_path}")

                    # Process the frame
                    processed = self.process_frame(img, is_video=True)

                    # Save the processed frame
                    cv2.imwrite(frame_path, processed)
                    print(f"Processing frame {i}/{total_frames}")

                # Combine frames back into video using ffmpeg
                combine_cmd = [
                    get_ffmpeg_exe(), '-y',  # -y to overwrite output file if it exists
                    '-framerate', '30',  # or use original video's framerate
                    '-i', os.path.join(temp_dir, 'frame_%d.png'),
                    '-c:v', 'libx264',  # use H.264 codec
                    '-pix_fmt', 'yuv420p',  # pixel format for better compatibility
                    '-crf', '23',  # quality setting (lower = better quality, 23 is default)
                    '-preset', 'medium',  # encoding speed preset
                    temp_output
                ]
                subprocess.run(combine_cmd, check=True)

                # Read the processed video
                with open(temp_output, 'rb') as f:
                    processed_video = f.read()

                return processed_video

        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
