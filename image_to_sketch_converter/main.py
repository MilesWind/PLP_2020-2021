from PIL import Image
import random as ran

image = Image.open("sample_inputs/" + input("file:")).convert("RGB")
image.resize((int(round(image.size[0])), int(round(image.size[1]))))
pixelMap_w = image.load()
print("...")
pixelMap_r = []
for x in range(image.size[0]):
	pixelMap_r.append([])
	for y in range(image.size[1]):
		pixelMap_r[x].append(pixelMap_w[x,y])


threshold = 16
r = 1
shading = 1/2

points = []
for X in range(-r, r+1):
	for Y in range(-r, r+1):
		if (X**2 + Y**2) ** 0.5 <= r:
			points.append([X, Y])


def solid(x,y,i):
	pixel = pixelMap_r[x][y][i]
	sum = 0
	for p in points:
		sum += abs(pixel - pixelMap_r[x+X][y+Y][i])
	if sum / len(points) > threshold:
		return 0
	else:
		return 255

print("...")

for x in range(image.size[0]):
	for y in range(image.size[1]):
		if (
			x > r and 
			y > r
			and 
			x < image.size[0]-r and 
			y < image.size[1]-r
		):
			p = int(round(sum((solid(x,y,0),solid(x,y,1),solid(x,y,2)))/3))
			pixelMap_w[x,y] = (p, p, p)

print("...")

for x in range(image.size[0]):
	for y in range(image.size[1]):
		average = [pixelMap_w[x,y][0], pixelMap_w[x,y][1], pixelMap_w[x,y][2]]
		if ran.randint(0, 255)*shading > pixelMap_r[x][y][0]:
			average[0] = 0
		if ran.randint(0, 255)*shading > pixelMap_r[x][y][1]:
			average[1] = 0
		if ran.randint(0, 255)*shading > pixelMap_r[x][y][2]:
			average[2] = 0
		p = int(round(sum(average)/3))
		pixelMap_w[x,y] = (p, p, p)

print("done!")

image.save("output_file.png")
