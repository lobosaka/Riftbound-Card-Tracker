from PIL import Image, ImageDraw, ImageFont

image = Image.new("RGB", (600, 200), "white")
draw = ImageDraw.Draw(image)

text = "Hello OCR 123"

draw.text((50, 70), text, fill="black")

image.save("pilot/input.png")