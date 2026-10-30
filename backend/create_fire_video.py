import cv2
import numpy as np
import os
import math
import imageio_ffmpeg
import subprocess

def create_fire_rescue_video(output_path, num_frames=80, width=640, height=360, fps=20):
    print(f"Generating fire rescue test video: {num_frames} frames at {width}x{height}...")

    # We will write frames to a temporary directory and encode with ffmpeg (H.264)
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_gen_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Precompute corridor geometry
    vanish_x, vanish_y = width // 2, height // 2 - 20
    
    # Random seed for repeatable ember particles
    np.random.seed(42)
    num_embers = 45
    embers_x = np.random.uniform(width * 0.2, width * 0.8, num_embers)
    embers_y = np.random.uniform(height * 0.4, height * 0.95, num_embers)
    embers_speed_y = np.random.uniform(2.0, 5.0, num_embers)
    embers_speed_x = np.random.uniform(-1.0, 1.0, num_embers)
    embers_size = np.random.randint(2, 5, num_embers)

    for f in range(num_frames):
        t = f / fps
        cam_z = f * 3.5  # moving forward into the hallway
        
        # 1. Base clean scene (Hallway with doors, exit sign, person silhouette)
        scene = np.zeros((height, width, 3), dtype=np.uint8)

        # Ceiling (dark industrial)
        cv2.rectangle(scene, (0, 0), (width, height), (30, 28, 32), -1)

        # Floor (tiles with perspective lines)
        floor_pts = np.array([
            [0, height],
            [width, height],
            [vanish_x + 60, vanish_y + 40],
            [vanish_x - 60, vanish_y + 40]
        ], np.int32)
        cv2.fillPoly(scene, [floor_pts], (45, 42, 40))

        # Floor grid perspective lines
        for offset in np.linspace(-width, width * 2, 18):
            pt_start = (int(vanish_x + (offset - vanish_x) * 0.15), int(vanish_y + 40))
            pt_end = (int(offset), height)
            cv2.line(scene, pt_start, pt_end, (65, 58, 55), 1)

        # Left Wall
        left_wall = np.array([
            [0, 0],
            [vanish_x - 60, vanish_y - 50],
            [vanish_x - 60, vanish_y + 40],
            [0, height]
        ], np.int32)
        cv2.fillPoly(scene, [left_wall], (60, 50, 48))

        # Right Wall
        right_wall = np.array([
            [width, 0],
            [vanish_x + 60, vanish_y - 50],
            [vanish_x + 60, vanish_y + 40],
            [width, height]
        ], np.int32)
        cv2.fillPoly(scene, [right_wall], (55, 48, 45))

        # End Wall (Far end of corridor)
        end_wall = np.array([
            [vanish_x - 60, vanish_y - 50],
            [vanish_x + 60, vanish_y - 50],
            [vanish_x + 60, vanish_y + 40],
            [vanish_x - 60, vanish_y + 40]
        ], np.int32)
        cv2.fillPoly(scene, [end_wall], (40, 35, 38))

        # Doors on left and right walls (moving with cam_z)
        door_positions = [60, 180, 320]
        for dp in door_positions:
            rel_z = dp - cam_z
            if rel_z > 15:
                # scale factor by distance
                scale = 120.0 / rel_z
                dw = int(35 * scale)
                dh = int(90 * scale)
                
                # Left door
                dl_x = int(vanish_x - 60 * scale - dw)
                dl_y = int(vanish_y + 40 * scale - dh)
                if 0 <= dl_x < width and dl_y < height:
                    cv2.rectangle(scene, (dl_x, dl_y), (dl_x + dw, dl_y + dh), (30, 25, 25), -1)
                    cv2.rectangle(scene, (dl_x, dl_y), (dl_x + dw, dl_y + dh), (15, 10, 10), 2)
                    # Door sign
                    cv2.putText(scene, "ROOM", (dl_x + 2, dl_y + int(dh*0.3)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.25, (200, 200, 200), 1)

                # Right door
                dr_x = int(vanish_x + 60 * scale)
                dr_y = int(vanish_y + 40 * scale - dh)
                if 0 <= dr_x < width and dr_y < height:
                    cv2.rectangle(scene, (dr_x, dr_y), (dr_x + dw, dr_y + dh), (30, 25, 25), -1)
                    cv2.rectangle(scene, (dr_x, dr_y), (dr_x + dw, dr_y + dh), (15, 10, 10), 2)

        # EMERGENCY EXIT Sign at the end of hallway
        exit_w = 46
        exit_h = 18
        exit_x = vanish_x - exit_w // 2
        exit_y = vanish_y - 40
        cv2.rectangle(scene, (exit_x, exit_y), (exit_x + exit_w, exit_y + exit_h), (0, 180, 0), -1)
        cv2.rectangle(scene, (exit_x, exit_y), (exit_x + exit_w, exit_y + exit_h), (255, 255, 255), 1)
        cv2.putText(scene, "EXIT", (exit_x + 6, exit_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Target in hallway: Person needing rescue at end of corridor!
        person_h = 60
        person_w = 22
        person_x = vanish_x - 10
        person_y = vanish_y + 40 - person_h
        # Silhouette of trapped person (head, body)
        cv2.circle(scene, (person_x + person_w // 2, person_y + 10), 8, (20, 20, 25), -1) # Head
        cv2.rectangle(scene, (person_x, person_y + 18), (person_x + person_w, person_y + person_h), (35, 45, 65), -1) # Jacket
        # Warning/Rescue Marker
        cv2.putText(scene, "[RESCUE TARGET]", (person_x - 30, person_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1)

        # 2. Fire flames & glowing reflections (left door area)
        fire_flicker = math.sin(f * 0.9) * 0.25 + math.cos(f * 1.7) * 0.2 + 0.9
        fire_base_x = int(width * 0.12)
        fire_base_y = int(height * 0.78)
        
        # Glow overlay
        glow = np.zeros_like(scene, dtype=np.float32)
        cv2.circle(glow, (fire_base_x + 30, fire_base_y - 40), int(120 * fire_flicker), (0, 90, 255), -1)
        cv2.circle(glow, (fire_base_x + 20, fire_base_y - 20), int(80 * fire_flicker), (0, 180, 255), -1)
        glow = cv2.GaussianBlur(glow, (75, 75), 0)
        scene = cv2.addWeighted(scene.astype(np.float32), 1.0, glow, 0.7, 0)
        scene = np.clip(scene, 0, 255).astype(np.uint8)

        # Animated Flame tongues
        for flame_i in range(5):
            flame_x = fire_base_x + flame_i * 14 + int(math.sin(f * 0.8 + flame_i) * 6)
            flame_h = int((65 + math.sin(f * 1.2 + flame_i * 2) * 20) * fire_flicker)
            flame_pts = np.array([
                [flame_x - 12, fire_base_y],
                [flame_x + 12, fire_base_y],
                [flame_x + int(math.sin(f + flame_i) * 5), fire_base_y - flame_h]
            ], np.int32)
            # Outer flame (Orange)
            cv2.fillPoly(scene, [flame_pts], (0, 120, 255))
            # Inner core (Yellow)
            inner_pts = np.array([
                [flame_x - 6, fire_base_y],
                [flame_x + 6, fire_base_y],
                [flame_x, fire_base_y - int(flame_h * 0.75)]
            ], np.int32)
            cv2.fillPoly(scene, [inner_pts], (0, 230, 255))

        # 3. Realistic Atmospheric Smoke / Haze Overlay (Optical Scattering Model)
        # Depth map: pixels further away (near vanish point) have greater depth
        Y, X = np.ogrid[:height, :width]
        dist_from_bottom = (height - Y) / float(height)
        dist_from_vanish = np.sqrt(((X - vanish_x) / width) ** 2 + ((Y - vanish_y) / height) ** 2)
        depth = (1.0 - dist_from_vanish * 0.8) * 0.8 + dist_from_bottom * 0.2
        depth = np.clip(depth, 0.1, 1.0)

        # Dynamic smoke density noise (swirling turbulent clouds)
        smoke_phase1 = f * 0.05
        smoke_phase2 = f * 0.08
        smoke_noise = np.sin(X * 0.02 + smoke_phase1) * np.cos(Y * 0.025 + smoke_phase2)
        smoke_noise += np.sin(X * 0.04 - smoke_phase2 * 0.5) * np.cos(Y * 0.04 + smoke_phase1 * 0.7) * 0.5
        smoke_noise = (smoke_noise + 1.5) / 3.0  # Normalize to [0, 1]
        
        # Heavy smoke concentration towards ceiling and far hallway
        smoke_density = depth * 0.75 + (1.0 - (Y / height)) * 0.25
        smoke_density = np.clip(smoke_density * (0.8 + 0.3 * smoke_noise), 0.15, 0.92)

        # Transmission map t(x) = exp(-beta * depth)
        beta = 1.95
        transmission = np.exp(-beta * smoke_density)
        transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)

        # Atmospheric Light A (thick grayish-ash smoke with warm orange fire scatter)
        A = np.zeros((height, width, 3), dtype=np.float32)
        A[:, :, 0] = 195  # Blueish-gray smoke
        A[:, :, 1] = 205  # Greenish-gray
        A[:, :, 2] = 225  # Warm tinted smoke near fire
        
        # Apply Haze Equation: I = J * t + A * (1 - t)
        hazy_frame = scene.astype(np.float32) * transmission + A * (1.0 - transmission)
        hazy_frame = np.clip(hazy_frame, 0, 255).astype(np.uint8)

        # 4. Animated Ember particles floating upwards
        for ei in range(num_embers):
            embers_y[ei] -= embers_speed_y[ei]
            embers_x[ei] += embers_speed_x[ei] + math.sin(f * 0.3 + ei) * 0.8
            # Reset if goes off top or sides
            if embers_y[ei] < 20 or embers_x[ei] < 20 or embers_x[ei] > width - 20:
                embers_y[ei] = np.random.uniform(height * 0.7, height * 0.95)
                embers_x[ei] = np.random.uniform(width * 0.2, width * 0.8)
            
            ex = int(embers_x[ei])
            ey = int(embers_y[ei])
            es = embers_size[ei]
            cv2.circle(hazy_frame, (ex, ey), es, (0, 160, 255), -1) # Orange ember
            cv2.circle(hazy_frame, (ex, ey), max(1, es - 1), (0, 240, 255), -1) # Bright yellow core

        # Save frame image
        frame_file = os.path.join(temp_dir, f"frame_{f:04d}.png")
        cv2.imwrite(frame_file, hazy_frame)

    print("Frames generated. Encoding into MP4 using ffmpeg...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    encode_cmd = [
        ffmpeg_exe, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "22",
        "-preset", "fast",
        output_path
    ]
    subprocess.run(encode_cmd, check=True)

    # Clean up temporary frames
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Video created successfully at: {output_path}")

if __name__ == "__main__":
    out = "D:/mini-project/image-dehazing/backend/video/fire_smoke_rescue.mp4"
    create_fire_rescue_video(out, num_frames=75, width=640, height=360, fps=20)
