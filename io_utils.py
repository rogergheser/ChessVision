import cairosvg
import io
from PIL import Image

def svg_to_image(svg_data):
    # Create a blank image
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(io.BytesIO(png_data))
    
    return image