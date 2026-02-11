import os
import shutil
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
src_dir = "/work/Sultan/runs/inference_on_vid2_with_S_model_with_crops_11_2_26/crops/vehicle/"
dst_dir = "/work/Sultan/data/for_resnet/1/"
# Create destination if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Get list of images
images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
images.sort() # Keep them in order

class CropSelector:
    def __init__(self, image_list):
        self.image_list = image_list
        self.batch_size = 50
        self.current_idx = 0
        
    def on_click(self, event):
        if event.inaxes is None:
            return
        
        # Get the filename from the title of the clicked subplot
        img_name = event.inaxes.get_title()
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"COPIED: {img_name} -> {dst_dir}")
            
            # Visual feedback: Dim the image once clicked
            event.inaxes.set_alpha(0.2)
            event.canvas.draw()

    def show_next_batch(self):
        if self.current_idx >= len(self.image_list):
            print("No more images left!")
            return

        plt.clf()
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        batch = self.image_list[self.current_idx : self.current_idx + self.batch_size]
        
        for i, img_name in enumerate(batch):
            ax = fig.add_subplot(10, 5, i + 1)
            img = cv2.imread(os.path.join(src_dir, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img)
            ax.set_title(img_name, fontsize=7)
            ax.axis('off')
        
        #plt.suptitle(f"Click images to copy to ResNet folder. Close window for next 25.\nBatch: {self.current_idx//50 + 1}", fontsize=11)
        plt.tight_layout()
        plt.show()
        
        self.current_idx += self.batch_size
        self.show_next_batch()

# Run the selector
selector = CropSelector(images)
plt.figure(figsize=(16, 12))
selector.show_next_batch()