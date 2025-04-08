import objc
import numpy as np
from PIL import Image
from io import BytesIO
import sys

# Import Metal components with fallbacks for different Metal versions
try:
    from Quartz import CIImage, CIContext, CGColorSpaceCreateDeviceRGB
    from Metal import (
        MTLCreateSystemDefaultDevice,
        MTLTextureDescriptor,
        MTLPixelFormatRGBA8Unorm,
        MTLPixelFormatRGBA8Unorm_sRGB,  # Alternative format for better sRGB handling
        MTLTextureUsageShaderRead,
        MTLTextureUsageRenderTarget,
        MTLTextureUsagePixelFormatView
    )
    from Metal import (
        MTLRegionMake2D,
        MTLRegionMake3D,
    )
    from MetalPerformanceShaders import MPSImageLanczosScale
except ImportError as e:
    print(f"Error importing Metal components: {e}", file=sys.stderr)
    print("Some Metal components couldn't be imported. CPU fallback will be used.", file=sys.stderr)
    # Re-raise if critical components are missing
    if "MTLCreateSystemDefaultDevice" in str(e):
        print("ERROR: Critical Metal components missing. Make sure PyObjC is installed.", file=sys.stderr)
        raise


# ▮ Step 1: Metal Setup
device = MTLCreateSystemDefaultDevice()
if device is None:
    print("Error: Could not create Metal device. GPU acceleration unavailable.", file=sys.stderr)
    sys.exit(1)
    
print(f"Using Metal device: {device.name()}")
command_queue = device.newCommandQueue()
ci_context = CIContext.contextWithMTLDevice_(device)


# ▮ Step 2: Load Image + Convert to CIImage
def pil_image_to_ciimage(pil_img):
    """Convert a PIL image to a Core Image (CIImage) object"""
    with BytesIO() as buf:
        pil_img.save(buf, format="PNG")
        data = buf.getvalue()
    
    NSData = objc.lookUpClass("NSData")
    if NSData is None:
        raise RuntimeError("Failed to find NSData class")
        
    ns_data = NSData.dataWithBytes_length_(data, len(data))
    if ns_data is None:
        raise RuntimeError("Failed to create NSData from bytes")
        
    ci_image = CIImage.imageWithData_(ns_data)
    if ci_image is None:
        # Try alternate approach with NSImage
        NSImage = objc.lookUpClass("NSImage")
        if NSImage is None:
            raise RuntimeError("Failed to create CIImage and NSImage class not found")
            
        ns_image = NSImage.alloc().initWithData_(ns_data)
        if ns_image is None:
            raise RuntimeError("Failed to create NSImage from data")
            
        ci_image = CIImage.imageWithNSImage_(ns_image)
        if ci_image is None:
            raise RuntimeError("Failed to create CIImage from NSImage")
    
    return ci_image


# ▮ Step 3: Render CIImage to Metal Texture
def ciimage_to_texture(ci_image, width, height):
    # Ensure width and height are powers of 2 for optimal Metal performance
    # or at least properly aligned values
    width = int(width)
    height = int(height)
    
    if width <= 0 or height <= 0:
        print(f"[WARNING] Invalid dimensions: {width}x{height}, adjusting to minimum size", file=sys.stderr)
        width = max(width, 4)
        height = max(height, 4)
    
    # Try to use sRGB format first for better color handling, safer on Apple GPUs
    try:
        tex_desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm_sRGB, width, height, False)
        tex_desc.setUsage_(MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView)
    except (NameError, AttributeError):
        # Fall back to standard RGBA format if sRGB isn't available
        tex_desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm, width, height, False)
        tex_desc.setUsage_(MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget)
    
    # Check device limits - hardcoded safe limit for all Metal devices
    # Apple Metal devices typically support at least 16384x16384
    max_texture_size = 16384  # Conservative limit that works on all Metal GPUs
    
    if width > max_texture_size or height > max_texture_size:
        print(f"[WARNING] Texture size {width}x{height} exceeds safe limit of {max_texture_size}", file=sys.stderr)
        width = min(width, max_texture_size)
        height = min(height, max_texture_size)
        # Create new descriptor with adjusted size
        tex_desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm, width, height, False)
        tex_desc.setUsage_(MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget)
    
    # Allocate texture
    texture = device.newTextureWithDescriptor_(tex_desc)
    if texture is None:
        raise RuntimeError(f"Failed to allocate Metal texture of size {width}x{height}")
    
    # Set up command buffer
    command_buffer = command_queue.commandBuffer()
    if command_buffer is None:
        raise RuntimeError("Failed to create Metal command buffer")
    
    # Create color space
    color_space = CGColorSpaceCreateDeviceRGB()
    if color_space is None:
        raise RuntimeError("Failed to create RGB color space")
    
    # Render CIImage to texture
    ci_context.render_toMTLTexture_commandBuffer_bounds_colorSpace_(
        ci_image,
        texture,
        command_buffer,
        ci_image.extent(),
        color_space
    )
    
    # Execute and wait for completion
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    
    if command_buffer.error() is not None:
        raise RuntimeError(f"Metal command buffer error: {command_buffer.error().localizedDescription()}")
    
    return texture


# ▮ Step 4: Resize Texture using GPU
def resize_texture(input_texture, output_size=(512, 512)):
    out_width, out_height = output_size
    
    # Validate input
    if input_texture is None:
        raise ValueError("Input texture cannot be None")
    
    # Ensure dimensions are integers and positive
    out_width = max(int(out_width), 4)  # Minimum texture size
    out_height = max(int(out_height), 4)
    
    # Check device limits - hardcoded safe limit for all Metal devices
    max_texture_size = 16384  # Conservative limit that works on all Metal GPUs
    if out_width > max_texture_size or out_height > max_texture_size:
        print(f"[WARNING] Output size {out_width}x{out_height} exceeds safe limit of {max_texture_size}", file=sys.stderr)
        out_width = min(out_width, max_texture_size)
        out_height = min(out_height, max_texture_size)
    
    # Try to use sRGB format first for better color handling, safer on Apple GPUs
    try:
        out_desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm_sRGB, out_width, out_height, False)
        out_desc.setUsage_(MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView)
    except (NameError, AttributeError):
        # Fall back to standard RGBA format if sRGB isn't available
        out_desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm, out_width, out_height, False)
        out_desc.setUsage_(MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget)
    
    # Create output texture
    output_texture = device.newTextureWithDescriptor_(out_desc)
    if output_texture is None:
        raise RuntimeError(f"Failed to allocate output texture of size {out_width}x{out_height}")
    
    # Create scaling filter
    try:
        lanczos = MPSImageLanczosScale.alloc().initWithDevice_(device)
        if lanczos is None:
            raise RuntimeError("Failed to create Lanczos scaling filter")
    except Exception as e:
        raise RuntimeError(f"Failed to create MPS scaling filter: {str(e)}")
    
    # Create command buffer
    command_buffer = command_queue.commandBuffer()
    if command_buffer is None:
        raise RuntimeError("Failed to create command buffer for resize operation")
    
    # Encode scaling operation
    try:
        lanczos.encodeToCommandBuffer_sourceTexture_destinationTexture_(
            command_buffer, input_texture, output_texture)
    except Exception as e:
        raise RuntimeError(f"Failed to encode scaling operation: {str(e)}")
    
    # Execute command buffer
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    
    # Check for errors
    if command_buffer.error() is not None:
        raise RuntimeError(f"Resize command buffer error: {command_buffer.error().localizedDescription()}")
    
    return output_texture


# ▮ Step 5: Convert MTLTexture → PIL Image
def texture_to_pil_image(texture):
    """Convert a Metal texture to a PIL image"""
    if texture is None:
        raise ValueError("Input texture cannot be None")
    
    # Try the most reliable method first - direct Core Graphics approach
    try:
        return _direct_texture_to_pil(texture)
    except Exception as e:
        print(f"[WARNING] Direct texture conversion failed: {str(e)}", file=sys.stderr)
        
        # Fall back to traditional method with more error handling
        try:
            width = texture.width()
            height = texture.height()
            
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid texture dimensions: {width}x{height}")
            
            # Calculate buffer size and create buffer
            bytes_per_row = 4 * width  # RGBA = 4 bytes per pixel
            buffer_size = bytes_per_row * height
            buffer = bytearray(buffer_size)
            
            # Try to get a Metal region
            try:
                # Try different region creation methods
                region = None
                try:
                    region = MTLRegionMake2D(0, 0, width, height)
                except Exception:
                    try:
                        region = MTLRegionMake3D(0, 0, 0, width, height, 1)
                    except Exception:
                        raise RuntimeError("Could not create Metal region")
                
                # Get the bytes from the texture
                texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(buffer, bytes_per_row, region, 0)
                
                # Convert to numpy array
                array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
                
                # Convert to PIL
                img = Image.fromarray(array)
                
                # Check if the image appears corrupted (checkered pattern)
                if _is_likely_corrupted(img):
                    print("[WARNING] Image appears corrupted, trying alternative method", file=sys.stderr)
                    return _alternative_texture_to_pil(texture)
                
                return img
                
            except Exception as e:
                print(f"[WARNING] Traditional texture reading failed: {str(e)}", file=sys.stderr)
                return _alternative_texture_to_pil(texture)
                
        except Exception as e:
            print(f"[ERROR] All texture conversion methods failed: {str(e)}", file=sys.stderr)
            # Last resort - use the safest method
            return _alternative_texture_to_pil(texture)

# Check if an image is likely corrupted (has a checkered pattern)
def _is_likely_corrupted(img):
    """
    Detect if the image appears to have a checkered pattern
    indicating corruption in the pixel data
    """
    try:
        # Convert to numpy array
        arr = np.array(img)
        
        # A checkered pattern often has high variance in small regions
        # Sample some random blocks and check their variance
        block_size = 16
        num_blocks = min(10, arr.shape[0] // block_size, arr.shape[1] // block_size)
        
        if num_blocks < 2:
            return False  # Too small to check
        
        # Check variance in random sample blocks
        for _ in range(5):  # Check 5 random blocks
            x = np.random.randint(0, arr.shape[1] - block_size)
            y = np.random.randint(0, arr.shape[0] - block_size)
            block = arr[y:y+block_size, x:x+block_size]
            
            # Check for alternating pattern
            # Checkered patterns often have high variance between adjacent pixels
            h_diff = np.abs(block[:-1, :] - block[1:, :]).mean()
            v_diff = np.abs(block[:, :-1] - block[:, 1:]).mean()
            
            if h_diff > 50 and v_diff > 50:
                return True  # Likely a checkered pattern
        
        return False
        
    except Exception:
        # If we can't analyze, assume it's not corrupted
        return False

# Reliable direct method using Core Graphics
def _direct_texture_to_pil(texture):
    """Direct method to convert MTLTexture to PIL using Core Graphics"""
    from Quartz import (
        CGBitmapContextCreate,
        CGColorSpaceCreateDeviceRGB,
        CGImageCreate,
        CGBitmapContextCreateImage,
        CGDataProviderCreateWithData,
        CGRectMake
    )
    
    width = texture.width()
    height = texture.height()
    
    # Create a buffer for the raw pixel data
    bytes_per_row = 4 * width
    buffer_size = bytes_per_row * height
    buffer = bytearray(buffer_size)
    
    # Create a Core Graphics bitmap context
    try:
        # Get the raw pixel data from the texture
        if hasattr(texture, 'getBytes_bytesPerRow_fromRegion_mipmapLevel_'):
            # Try the traditional Metal method first
            region = MTLRegionMake2D(0, 0, width, height)
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(buffer, bytes_per_row, region, 0)
        else:
            # Fallback to CIImage if direct access fails
            ci_image = CIImage.imageWithMTLTexture_options_(texture, None)
            if ci_image is None:
                raise RuntimeError("Failed to create CIImage from texture")
                
            # Render to a new buffer
            color_space = CGColorSpaceCreateDeviceRGB()
            context = CGBitmapContextCreate(
                buffer,
                width,
                height,
                8,  # 8 bits per component
                bytes_per_row,
                color_space,
                2  # kCGImageAlphaPremultipliedLast
            )
            
            if context is None:
                raise RuntimeError("Failed to create CGBitmapContext")
                
            # Render the CIImage to the bitmap context
            ci_context.render_toBitmapData_rowBytes_bounds_format_colorSpace_(
                buffer,
                bytes_per_row,
                CGRectMake(0, 0, width, height),
                32,  # RGBA
                color_space
            )
            
        # Convert to numpy array and then to PIL
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        return Image.fromarray(array)
        
    except Exception as e:
        print(f"[ERROR] Direct texture conversion failed: {str(e)}", file=sys.stderr)
        return _alternative_texture_to_pil(texture)

# Alternative conversion path using CIImage
def _alternative_texture_to_pil(texture):
    """Fallback method to convert MTLTexture to PIL using CIImage as intermediary"""
    try:
        print("[INFO] Using alternative texture conversion path", file=sys.stderr)
        
        # Create CIImage from texture
        ci_image = CIImage.imageWithMTLTexture_options_(texture, None)
        if ci_image is None:
            raise RuntimeError("Failed to create CIImage from texture")
        
        # Create CGImage from CIImage
        cg_image = ci_context.createCGImage_fromRect_(ci_image, ci_image.extent())
        if cg_image is None:
            raise RuntimeError("Failed to create CGImage from CIImage")
        
        # Create NSBitmapImageRep from CGImage
        NSBitmapImageRep = objc.lookUpClass("NSBitmapImageRep")
        if NSBitmapImageRep is None:
            raise RuntimeError("Failed to find NSBitmapImageRep class")
            
        bitmap_rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
        if bitmap_rep is None:
            raise RuntimeError("Failed to create NSBitmapImageRep from CGImage")
        
        # Convert to PNG data
        NSData = objc.lookUpClass("NSData")
        if NSData is None:
            raise RuntimeError("Failed to find NSData class")
            
        png_data = bitmap_rep.representationUsingType_properties_(4, None)  # 4 = NSPNGFileType
        if png_data is None:
            raise RuntimeError("Failed to convert bitmap to PNG data")
        
        # Convert to Python bytes
        bytes_data = png_data.bytes().tobytes()
        
        # Create PIL image from bytes
        return Image.open(BytesIO(bytes_data))
    except Exception as e:
        print(f"[ERROR] Alternative texture conversion failed: {str(e)}", file=sys.stderr)
        # Create a blank image as last resort
        width = texture.width()
        height = texture.height()
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))


# ▮ Fast CPU-based resize using numpy
def _fast_numpy_resize(pil_image, output_size):
    """
    Fast CPU-based resize using numpy and linear interpolation.
    """
    import numpy as np
    from scipy import ndimage
    
    # Convert PIL to numpy array
    img_array = np.array(pil_image)
    
    # Calculate scaling factors
    scale_x = output_size[0] / img_array.shape[1]
    scale_y = output_size[1] / img_array.shape[0]
    
    # Resize using scipy's ndimage zoom
    resized = ndimage.zoom(img_array, (scale_y, scale_x, 1), order=1)
    
    # Convert back to PIL
    return Image.fromarray(resized.astype(np.uint8))

# ▮ CoreImage-based resize as backup GPU method
def _ci_resize_image(pil_image, output_size):
    """
    Use CoreImage for GPU-accelerated resizing (alternative to Metal).
    """
    # Convert PIL to CIImage
    ci_image = pil_image_to_ciimage(pil_image)
    
    # Create a scale transform
    scale_x = output_size[0] / pil_image.width
    scale_y = output_size[1] / pil_image.height
    scaled_image = ci_image.imageByApplyingTransform_(
        CIImage.affineTransformMakeScale(scale_x, scale_y)
    )
    
    # Create a CG image from the CI image
    cg_image = ci_context.createCGImage_fromRect_(
        scaled_image, scaled_image.extent()
    )
    
    # Get NSBitmapImageRep from CGImage
    NSBitmapImageRep = objc.lookUpClass("NSBitmapImageRep")
    bitmap_rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
    
    # Convert to PNG data
    NSData = objc.lookUpClass("NSData") 
    png_data = bitmap_rep.representationUsingType_properties_(4, None)  # 4 = NSPNGFileType
    
    # Convert to Python bytes
    bytes_data = png_data.bytes().tobytes()
    
    # Create PIL image from bytes
    return Image.open(BytesIO(bytes_data))

# ▮ High-Level Resize API
def resize_image_metal(image_path_or_pil, output_size=(512, 512), allow_cpu_fallback=False):
    """
    Resize image using Metal GPU acceleration.
    
    Args:
        image_path_or_pil: Path to image or PIL Image object
        output_size: Tuple of (width, height) for output image
        allow_cpu_fallback: If True, fall back to CPU when Metal fails
        
    Returns:
        PIL Image object
    """
    if isinstance(image_path_or_pil, str):
        pil_image = Image.open(image_path_or_pil).convert("RGBA")
    else:
        pil_image = image_path_or_pil.convert("RGBA")

    print(f"[MPS GPU] Using Metal Performance Shaders for GPU-accelerated resizing: {pil_image.width}x{pil_image.height} → {output_size[0]}x{output_size[1]}")
    
    # Try primary Metal approach
    try:
        # Convert PIL image to CIImage
        ci_image = pil_image_to_ciimage(pil_image)
        if ci_image is None:
            raise RuntimeError("Failed to convert PIL image to CIImage")
        
        # Convert CIImage to Metal texture
        texture = ciimage_to_texture(ci_image, pil_image.width, pil_image.height)
        if texture is None:
            raise RuntimeError("Failed to convert CIImage to Metal texture")
        
        # Resize using GPU
        resized_texture = resize_texture(texture, output_size)
        if resized_texture is None:
            raise RuntimeError("Failed to resize texture")
        
        # Convert back to PIL
        result = texture_to_pil_image(resized_texture)
        
        # Check if result is corrupted
        if _is_likely_corrupted(result):
            print("[WARNING] Result appears corrupted, trying alternative method", file=sys.stderr)
            raise RuntimeError("Output image appears corrupted")
            
        return result
    
    except Exception as e:
        print(f"[ERROR] Primary Metal GPU processing failed: {str(e)}", file=sys.stderr)
        
        # Try alternative Core Image approach if still want GPU
        if not allow_cpu_fallback:
            try:
                print("[INFO] Trying CoreImage resize as alternative GPU method", file=sys.stderr)
                return _ci_resize_image(pil_image, output_size)
            except Exception as e2:
                print(f"[ERROR] CoreImage fallback failed: {str(e2)}", file=sys.stderr)
                raise RuntimeError(f"All GPU methods failed: {str(e)} / {str(e2)}") from e
        
        # CPU fallback
        if allow_cpu_fallback:
            print("[FALLBACK] Using CPU-based resizing", file=sys.stderr)
            try:
                # Try fast numpy resize first
                return _fast_numpy_resize(pil_image, output_size)
            except Exception:
                # Last resort - PIL resize
                return pil_image.resize(output_size, Image.LANCZOS)
        else:
            raise
    

# ▮ Pure CoreImage resize (completely bypass Metal)
def _pure_ci_resize(pil_image, output_size):
    """
    Use CoreImage directly for resizing without Metal textures.
    This avoids all the potential texture corruption issues.
    """
    from Quartz import (
        CIFilter,
        CGRectMake,
        CGBitmapContextCreate,
        CGColorSpaceCreateDeviceRGB,
        CGBitmapContextCreateImage,
        CGContextDrawImage
    )
    
    try:
        # Get dimensions
        width, height = pil_image.size
        out_width, out_height = output_size
        
        # Convert PIL to CIImage through NSData
        ci_image = pil_image_to_ciimage(pil_image)
        
        # Create a lanczos scale filter
        scale_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
        if scale_filter is None:
            # Fall back to affine transform
            scale_x = output_size[0] / pil_image.width
            scale_y = output_size[1] / pil_image.height
            scaled_ci_image = ci_image.imageByApplyingTransform_(
                CIImage.affineTransformMakeScale(scale_x, scale_y)
            )
        else:
            # Use Lanczos filter
            scale_filter.setValue_forKey_(ci_image, "inputImage")
            
            # Calculate scale (preserve aspect ratio)
            scale = min(out_width / width, out_height / height)
            aspect_ratio = 1.0  # Square pixels
            
            scale_filter.setValue_forKey_(scale, "inputScale")
            scale_filter.setValue_forKey_(aspect_ratio, "inputAspectRatio")
            
            # Get result
            scaled_ci_image = scale_filter.valueForKey_("outputImage")
        
        # Create CGImage from CIImage
        cg_image = ci_context.createCGImage_fromRect_(
            scaled_ci_image, 
            scaled_ci_image.extent()
        )
        
        # Create a bitmap context
        color_space = CGColorSpaceCreateDeviceRGB()
        bytes_per_row = 4 * out_width
        buffer = bytearray(bytes_per_row * out_height)
        
        context = CGBitmapContextCreate(
            buffer,
            out_width,
            out_height,
            8,  # 8 bits per component
            bytes_per_row,
            color_space,
            2  # kCGImageAlphaPremultipliedLast
        )
        
        # Draw the image into the context
        CGContextDrawImage(context, CGRectMake(0, 0, out_width, out_height), cg_image)
        
        # Convert to numpy array and then to PIL
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(out_height, out_width, 4)
        return Image.fromarray(array)
        
    except Exception as e:
        print(f"[ERROR] Pure CoreImage resize failed: {str(e)}", file=sys.stderr)
        # Fall back to PIL
        return pil_image.resize(output_size, Image.LANCZOS)

# ▮ Simplified API - for easier usage
def resize_image(image_path_or_pil, output_size=(512, 512), progressive_resize=True, use_metal=False):
    """
    Public API for image resizing that defaults to CoreImage but allows Metal or CPU fallback.
    
    Args:
        image_path_or_pil: Path to image or PIL Image object
        output_size: Tuple of (width, height) for output image
        progressive_resize: If True, very large images will be resized in stages
        use_metal: If True, try Metal first (may cause checkered artifacts)
        
    Returns:
        PIL Image object
    """
    # Load image if needed
    if isinstance(image_path_or_pil, str):
        pil_image = Image.open(image_path_or_pil).convert("RGBA")
    else:
        pil_image = image_path_or_pil.convert("RGBA")
    
    # For very large images, do progressive resizing to avoid memory issues
    width, height = pil_image.size
    if progressive_resize and (width > 4000 or height > 4000):
        print(f"[INFO] Progressive resize for large image: {width}x{height}", file=sys.stderr)
        
        # Calculate intermediate size (half the original dimensions)
        intermediate_w = max(width // 2, output_size[0])
        intermediate_h = max(height // 2, output_size[1])
        
        # First resize with PIL to intermediate size
        intermediate = pil_image.resize((intermediate_w, intermediate_h), Image.LANCZOS)
        
        # Then use GPU for final resize
        pil_image = intermediate  # Continue with the intermediate image
    
    # Try the most reliable approach first - pure CoreImage
    # This avoids the Metal texture issues entirely
    try:
        print(f"[GPU] Using CoreImage GPU acceleration for resizing: {pil_image.width}x{pil_image.height} → {output_size[0]}x{output_size[1]}")
        return _pure_ci_resize(pil_image, output_size)
    except Exception as e:
        print(f"[ERROR] CoreImage resize failed: {str(e)}", file=sys.stderr)
        
        # Try Metal only if specifically requested
        if use_metal:
            try:
                return resize_image_metal(pil_image, output_size, allow_cpu_fallback=True)
            except Exception as e2:
                print(f"[ERROR] Metal resize failed: {str(e2)}", file=sys.stderr)
        
        # CPU fallback
        print("[FALLBACK] Using CPU-based resizing", file=sys.stderr)
        try:
            # Try fast numpy resize first
            return _fast_numpy_resize(pil_image, output_size)
        except Exception:
            # Last resort - PIL resize
            return pil_image.resize(output_size, Image.LANCZOS)


# Run as script for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process images using GPU acceleration")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--method", choices=["coreimage", "metal", "cpu"], default="coreimage", 
                        help="Processing method (default: coreimage)")
    parser.add_argument("--no-progressive", action="store_true", help="Disable progressive resize for large images")
    args = parser.parse_args()
    
    try:
        print(f"Processing {args.input} to {args.output} at {args.width}x{args.height} using {args.method}")
        
        if args.method == "cpu":
            # Force CPU processing
            pil_image = Image.open(args.input).convert("RGBA")
            result = pil_image.resize((args.width, args.height), Image.LANCZOS)
        elif args.method == "metal":
            # Force Metal processing with fallback
            pil_image = Image.open(args.input).convert("RGBA")
            result = resize_image_metal(
                pil_image,
                output_size=(args.width, args.height),
                allow_cpu_fallback=True
            )
        else:  # coreimage (default)
            # Use the high-level API with CoreImage (most reliable)
            result = resize_image(
                args.input,
                output_size=(args.width, args.height),
                progressive_resize=not args.no_progressive,
                use_metal=False  # Avoid Metal by default due to checkered issues
            )
            
        result.save(args.output)
        print(f"Successfully processed image: {args.output}")
    except Exception as e:
        print(f"Error processing image: {str(e)}", file=sys.stderr)
        sys.exit(1)