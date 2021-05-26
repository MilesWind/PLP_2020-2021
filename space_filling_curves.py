import copy;
import time;
import math;
import random;

def Curve(order, seed, end=True):
	if order == 0:
		return [[0.5, 0.5]];

	channel = 2**order / 2
	a = Curve(order-1, seed, False);
	b = copy.deepcopy(a);
	c = copy.deepcopy(b);
	d = copy.deepcopy(c);
	curve = [];


	s = list(bin(seed)[2:]);
	for i in range(8):
		if len(s) > i:
			s[i] = int(s[i]);
		else:
			s.append(0);

	x = [];
	y = [];

	for i in range(len(a)):
		x.append(a[i][1]+s[0]/2);
		y.append(a[i][0]+s[1]/2);

	mid_x = sum(x) / len(x);
	mid_y = sum(y) / len(y);

	for i in range(len(a)):
		curve.append([x[i][0], y[i][1]])
	
	x = [];
	y = [];

	for i in range(len(b)):
		x.append(b[i][0]+s[2]/2);
		y.append(b[i][1]+channel+s[3]/2);

	mid_x = sum(x) / len(x);
	mid_y = sum(y) / len(y);

	for i in range(len(b)):
		curve.append([b[i][0]+s[2]/2, b[i][1]+channel+s[3]/2])

	for i in range(len(b)):
		curve.append([b[i][0]+s[2]/2, b[i][1]+channel+s[3]/2]);
	for i in range(len(c)):
		curve.append([c[i][0]+channel+s[4]/2, c[i][1]+channel+s[5]/2]);
	for i in range(len(d)):
		curve.append([(channel-d[i][1])+channel+s[6]/2, (channel-d[i][0])+s[7]/2]);
	
	if end:
		for i in range(len(curve)):
			curve[i] = [curve[i][0]/channel/2, curve[i][1]/channel/2];
	
	return curve;

def rgb_lerp(start, end, i):
	r = int(round(start[0] + (end[0] - start[0]) * i));
	g = int(round(start[1] + (end[1] - start[1]) * i));
	b = int(round(start[2] + (end[2] - start[2]) * i));
	return '#%02x%02x%02x' % (r, g, b);


import tkinter;
width = 300;
height = 300;
root = tkinter.Tk();
canvas = tkinter.Canvas(root, width=width, height=height, bg='black');
canvas.pack();

seed = input('seed (0 - 255) | 0 for hilbert curve | r for random: ');
if seed == 'r':
	seed = random.randint(0, 256);
	print(seed);
	print(bin(seed));
else:
	seed = int(round(float(seed)));
curve = Curve(int(round(float(input('iteration (1 - 7): ')))), seed);
speed = float(input('animation speed (1 - 10) | 0 for instant: '));

if speed != 0 and speed != 10:
	speed = 1/(speed*4) * 0.2;
if speed == 10:
	speed = 0.000000000001;

start = (255, 128, 64);
end = (0, 128, 191);


for i in range(1, len(curve)):
	canvas.create_line(
		10+curve[i-1][0]*(height-20),
		10+curve[i-1][1]*(height-20),
		10+curve[i][0]*(height-20),
		10+curve[i][1]*(height-20),
		width = 2,
		fill = rgb_lerp(start, end, i/len(curve))
	);
	if speed != 0:
		canvas.update();
		time.sleep(speed);
