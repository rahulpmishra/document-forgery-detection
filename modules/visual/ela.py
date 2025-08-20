from PIL import Image, ImageChops, ImageEnhance

def ela_image(img_path: str, quality: int = 90, enhance: float = 10.0):
    """Perform Error Level Analysis and return a PIL Image with amplified differences."""
    orig = Image.open(img_path).convert('RGB')
    tmp_path = img_path + '.ela.jpg'
    orig.save(tmp_path, 'JPEG', quality=quality)
    comp = Image.open(tmp_path)
    diff = ImageChops.difference(orig, comp)
    return ImageEnhance.Brightness(diff).enhance(enhance) 
