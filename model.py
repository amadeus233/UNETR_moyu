from logging import config
import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore

class OrangeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(OrangeBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreenBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,padding=0,stride=2)

    def forward(self, x):
        return self.deconv(x)

class BlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlueBlock, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,padding=0,stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.deconv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class GreyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreyBlock, self).__init__()    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        return x
    
class UNETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch Embedding Using Convolution - Cut Image Into Small Squares And Transform To Vectors
        patch_size = (config["patch_height"], config["patch_width"])
        n_patches = (config["image_height"] // patch_size[0]) * (config["image_width"] // patch_size[1])
        
        # Store patch grid dimensions for reshaping in forward
        self.patch_h = config["image_height"] // patch_size[0]
        self.patch_w = config["image_width"] // patch_size[1]
        
        # Linear Projection Layer - Change Picture To Numbers
        # Use Conv2d to project patches to hidden dimension directly
        self.patch_embedding = nn.Conv2d(
            in_channels=config["num_channels"],
            out_channels=config["hidden_dim"],
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional Embedding - Give Position Information To Network
        self.register_buffer('positions', torch.arange(start=0, end=n_patches, step=1).long())
        self.positions_embeddings = nn.Embedding(n_patches, config["hidden_dim"])
        
        # Transformer Encoder - The Brain Of Model Learn Features
        # CRITICAL: Use ModuleList so layers are registered and move to GPU correctly
        self.encoder_layers = nn.ModuleList()
        for i in range(config["num_layers"]):
            layer = nn.TransformerEncoderLayer(d_model=config["hidden_dim"],nhead=config["num_heads"], dim_feedforward = config["mlp_dim"], dropout = config["dropout_rate"], activation = nn.GELU(), batch_first = True)
            self.encoder_layers.append(layer)
        
        # CNN Decoder Layers - Pre-define all convolutional blocks in __init__
        # This ensures they are registered as modules and move to GPU correctly
        
        # Z9-Z12 decoder pathway
        self.z9_blue1 = BlueBlock(config["hidden_dim"], 512)
        self.z12_green1 = GreenBlock(config["hidden_dim"], 512)
        self.z9z12_orange1 = OrangeBlock(1024, 512)  # 512+512 concatenated
        self.z9z12_orange2 = OrangeBlock(512, 512)
        
        # Z6-Z9-Z12 decoder pathway
        self.z9z12_green2 = GreenBlock(512, 256)
        self.z6_blue1 = BlueBlock(config["hidden_dim"], 256)  # First upsampling
        self.z6_blue2 = BlueBlock(256, 256)  # Second upsampling to match
        self.z6z9z12_orange1 = OrangeBlock(512, 256)  # 256+256 concatenated
        self.z6z9z12_orange2 = OrangeBlock(256, 256)
        
        # Z3-Z6-Z9-Z12 decoder pathway
        self.z6z9z12_green3 = GreenBlock(256, 128)
        self.z3_blue1 = BlueBlock(config["hidden_dim"], 128)  # First upsampling
        self.z3_blue2 = BlueBlock(128, 128)  # Second upsampling to match
        self.z3_green4 = GreenBlock(128, 128)  # Third upsampling to match z6z9z12_d5
        self.z3z6z9z12_orange1 = OrangeBlock(256, 128)  # 128+128 concatenated
        self.z3z6z9z12_orange2 = OrangeBlock(128, 128)
        
        # Z0-Z3-Z6-Z9-Z12 decoder pathway (final stages)
        self.z3z6z9z12_green4 = GreenBlock(128, 64)
        self.z0_orange1 = OrangeBlock(config["num_channels"], 64)  # num_channels from input
        self.z0_orange2 = OrangeBlock(64, 64)
        self.z0z3z6z9z12_orange1 = OrangeBlock(128, 64)  # 64+64 concatenated
        self.z0z3z6z9z12_orange2 = OrangeBlock(64, 64)
        
        # Final classification head - output single channel for binary segmentation
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
            
    
    
    def forward(self, inputs):
        
        batch_size = inputs.shape[0]
        
        # Move positional embeddings to same device as inputs - CRITICAL FIX
        positions = self.positions.to(inputs.device)
        
        # Linear Projection And Positional Embedding - First Step Processing Input
        patch_embedding = self.patch_embedding(inputs)  # (batch, hidden_dim, h/patch_h, w/patch_w)
        # print("inputs shape : ", inputs.shape)
        # print("Patch Embedding Shape After Conv : ", patch_embedding.shape)
        
        # Flatten spatial dimensions to get (batch, num_patches, hidden_dim)
        patch_embedding = patch_embedding.flatten(2).transpose(-1, -2)  # Rearrange to (batch, num_patches, hidden_dim)
        # print("Patch Embedding Shape After Flattening : ", patch_embedding.shape)
        
        positions_embeddings = self.positions_embeddings(positions)
        # print("Positions Embeddings Shape : ", positions_embeddings.shape)
        x = patch_embedding + positions_embeddings
        # print("Concatenated Embeddings Shape : ", x.shape)
        
        # Transformer Encoder - Deep Learning Feature Extraction Happens Here
        connection_map = [3, 6, 9, 12]
        feature_map = []
        j=1
        for layer in self.encoder_layers:
            x = layer(x)
            if j in connection_map:
                feature_map.append(x)
            j=j+1
        
            
        # Convolutional Decoder - Rebuild Segmentation Mask From Features
        z3, z6, z9, z12 = feature_map
        # Use calculated patch dimensions for reshaping
        hidden_shape = (batch_size, self.config["hidden_dim"], self.patch_h, self.patch_w)
        print("DEBUG - Patch dimensions: patch_h=%d, patch_w=%d" % (self.patch_h, self.patch_w))
        print("DEBUG - Feature shapes before reshape: z3=%s, z6=%s, z9=%s, z12=%s" % (str(z3.shape), str(z6.shape), str(z9.shape), str(z12.shape)))
        z3 = z3.view(hidden_shape)
        z6 = z6.view(hidden_shape)
        z9 = z9.view(hidden_shape)
        z12 = z12.view(hidden_shape)
        print("DEBUG - Feature shapes after reshape: z3=%s, z6=%s, z9=%s, z12=%s" % (str(z3.shape), str(z6.shape), str(z9.shape), str(z12.shape)))
        
        
        # z9 and z12 Decoder Part - Combine Two Deep Level Features Together
        z9_d1 = self.z9_blue1(z9)
        z12_d1 = self.z12_green1(z12)
        print("DEBUG - z9_d1=%s, z12_d1=%s" % (str(z9_d1.shape), str(z12_d1.shape)))
        z9z12_d1 = torch.cat([z9_d1, z12_d1], dim = 1)
        z9z12_d2 = self.z9z12_orange1(z9z12_d1)
        z9z12_d2 = self.z9z12_orange2(z9z12_d2)
        print("DEBUG - Z9-Z12 decoder output: z9z12_d2=%s" % str(z9z12_d2.shape))
        
        # z6 and z9-z12 Decoder Part - Add Middle Level Feature Information
        z9z12_d3 = self.z9z12_green2(z9z12_d2)  # [B, 256, 64, 64]
        z6_d1 = self.z6_blue1(z6)  # [B, 256, 32, 32]
        z6_d1 = self.z6_blue2(z6_d1)  # [B, 256, 64, 64] - match z9z12_d3
        print("DEBUG - z9z12_d3=%s, z6_d1(after 2nd blue)=%s" % (str(z9z12_d3.shape), str(z6_d1.shape)))
        z6z9z12_d3 = torch.cat([z6_d1, z9z12_d3], dim = 1)
        z6z9z12_d4 = self.z6z9z12_orange1(z6z9z12_d3)
        z6z9z12_d4 = self.z6z9z12_orange2(z6z9z12_d4)
        print("DEBUG - Z6-Z9-Z12 decoder output: z6z9z12_d4=%s" % str(z6z9z12_d4.shape))
        
        # z3 and z6-z9-z12 Decoder Part - More Shallow Features Join Network
        z6z9z12_d5 = self.z6z9z12_green3(z6z9z12_d4)  # [B, 128, 128, 128]
        z3_d1 = self.z3_blue1(z3)  # [B, 128, 32, 32]
        z3_d1 = self.z3_blue2(z3_d1)  # [B, 128, 64, 64]
        z3_d1 = self.z3_green4(z3_d1)  # [B, 128, 128, 128] - NOW MATCH z6z9z12_d5!
        print("DEBUG - z6z9z12_d5=%s, z3_d1(after green4)=%s" % (str(z6z9z12_d5.shape), str(z3_d1.shape)))
        z3z6z9z12_d5 = torch.cat([z3_d1, z6z9z12_d5], dim = 1)
        z3z6z9z12_d6 = self.z3z6z9z12_orange1(z3z6z9z12_d5)
        z3z6z9z12_d6 = self.z3z6z9z12_orange2(z3z6z9z12_d6)
        
        # z0 and z3-z6-z9-z12 Decoder Part - Original Image Connect With All Processed Features
        z0 = inputs.view(batch_size, self.config["num_channels"], self.config["image_width"], self.config["image_height"])
        # print("Z0 after Reshaping : ", z0.shape)
        z3z6z9z12_d7 = self.z3z6z9z12_green4(z3z6z9z12_d6)
        z0_d1 = self.z0_orange1(z0)
        z0_d1 = self.z0_orange2(z0_d1)
        z0z3z6z9z12_d7 = torch.cat([z0_d1, z3z6z9z12_d7], dim = 1)
        z0z3z6z9z12_d8 = self.z0z3z6z9z12_orange1(z0z3z6z9z12_d7)
        z0z3z6z9z12_d8 = self.z0z3z6z9z12_orange2(z0z3z6z9z12_d8)
        # print("Z0-Z3-Z6-Z9-Z12 DECODER OUTPUT SHAPE : ", z0z3z6z9z12_d8.shape)
        
        
        # Output (the mask) - Final Prediction Result Come Out
        output = self.final_conv(z0z3z6z9z12_d8)
        return output
        
    
if __name__ == "__main__":
    
    configuration = {}
    # For Example Purpose - If Your Image Size Is 256x256 With 3 Color Channels
    image_shape = np.array([256, 256, 3])
    configuration["image_height"] = image_shape[0]
    configuration["image_width"] = image_shape[1]
    configuration["num_channels"] = image_shape[2]
    configuration["patch_height"] = 16
    configuration["patch_width"] = 16
    configuration["num_patches"] = (configuration["image_height"] * configuration["image_width"]) // (configuration["patch_height"] * configuration["patch_width"]) 
    configuration["num_layers"] = 12
    configuration["hidden_dim"] = 768
    configuration["mlp_dim"] = 3072
    configuration["dropout_rate"] = 0.3
    

    
# Load And Preprocess An Input Image - Prepare Data For Model
    image_path = "MRI4gt.png"  
    preprocess = transforms.Compose([ transforms.Resize((configuration["image_height"], configuration["image_width"])),transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)  
    # print("Image Tensor Shape:", image_tensor.shape)
    
    # Create Model Instance - Initialize Neural Network Structure
    model = UNETR(configuration)
    output = model(patches)
    # print("Model Output Shape:", output.shape)
        
       
    # Apply Sigmoid Function - Change Values Between 0 And 1 For Probability Meaning
    output_prob = torch.sigmoid(output)
    output_image = output_prob.squeeze(0).squeeze(0).detach().numpy()  
    # print("Output Image Shape:", output_image.shape)
    # print("Output min/max values:", output_image.min(), output_image.max())
    
    # Save The Segmentation Mask Result - Write Predicted Mask To File
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input MRI Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    im2 = plt.imshow(output_image, cmap="hot", vmin=0, vmax=1)
    plt.title(f"Segmentation Mask (Probabilities)\nMin: {output_image.min():.4f}, Max: {output_image.max():.4f}")
    plt.colorbar(im2)
    plt.axis("off")
    
    # Binary threshold at 0.5
    binary_mask = (output_image > 0.5).astype(float)
    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Binary Segmentation (>0.5)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_result.png", dpi=300, bbox_inches='tight')
    print("\nResult saved to: segmentation_result.png")
    plt.show()