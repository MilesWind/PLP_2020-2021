class circle:
	def __init__(self, active, position=(0,0), velocity=(0,0), radius=1,G=(0, 1)):
		self.active = active
		self.position = position
		self.velocity = velocity
		self.radius = radius
		self.G = G

		self.new_pos = None
		self.new_velo = None
	

	def calculate_propreties(self, enviroment):

		self.new_velo = (
			self.velocity[0] + self.G[0],
			self.velocity[1] + self.G[1]
		)

		for circ in enviroment:
			if circ != self:
				if collision(self, circ):
					self.collide(circ)
	
	def apply_properties(self, delta):

		if self.new_pos:
			self.position = self.new_pos
			self.new_pos = None
		if self.new_velo:
			self.velocity = self.new_velo
			self.new_velo = None

		self.position = (
			self.position[0] + self.velocity[0] * delta,
			self.position[1] + self.velocity[1] * delta
		)
	
	def collide(self, circ):
		

		vector = normalized(difference(self.position, circ.position))
		vector = (vector[0] * circ.radius, vector[1] * circ.radius)
		coli_collision_point = (circ.position[0] + vector[0], circ.position[1] + vector[1])

		vector = normalized(difference(circ.position, self.position))
		vector = (vector[0] * self.radius, vector[1] * self.radius)
		self_collision_point = (self.position[0] + vector[0], self.position[1] + vector[1])

		vector = difference(coli_collision_point, self_collision_point)
		self.new_pos = (self.position[0] + vector[0]/1.8, self.position[1] + vector[1]/1.8)

		vector = normalized(vector)

		velo_a = (-self.velocity[0], -self.velocity[1])
		product = dot(velo_a, vector) + dot(circ.velocity, vector)
		proj = (product * vector[0], product * vector[1])

		self.new_velo = difference(proj, velo_a)





def collision(a, b):
	if distance(a.position, b.position) < a.radius + b.radius:
		return True
	return False


def distance(a,b):
	return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

def dot(a,b):
	return a[0] * b[0] + a[1] * b[1]

def difference(target,origin):
	return (target[0] - origin[0], target[1] - origin[1])


def normalized(a):
	mag = magnitude(a)
	return (a[0] / mag, a[1] / mag)

def magnitude(a):
	return (a[0]**2 + a[1]**2) ** 0.5





from tkinter import Tk, Canvas
width = 600
height = 300
root = Tk()
c = Canvas(root, width=width, height=height)
c.pack()

delta = 0.001
enviroment = []

global start, shape, graphics, alive
start = None
shape = None
graphics = []
alive = False

def update_enviroment(e=None):
	if e:
		global alive
		alive = True
		c.delete('all')

	global graphics
	for g in graphics:
		c.delete(g)
	graphics = []
	for circ in enviroment:
		circ.G = normalized(difference((width/2,height/2), circ.position))
		if circ.active:
			circ.calculate_propreties(enviroment)

	for circ in enviroment:
		if circ.active:
			circ.apply_properties(delta)
		graphics.append(c.create_oval(
			circ.position[0] - circ.radius,
			circ.position[1] - circ.radius,
			circ.position[0] + circ.radius,
			circ.position[1] + circ.radius
		))
	c.update()
	root.after(0, update_enviroment)


def start_spawn(e):
	global start
	start = (e.x, e.y)
	update_spawn(e)

def update_spawn(e):
	global shape
	if start:
		if (e.x, e.y) != start:
			c.delete(shape)
		end = (e.x, e.y)
		d = distance(start, end)
		shape = c.create_oval(
			start[0] - d,
			start[1] - d,
			start[0] + d,
			start[1] + d
		)
		c.update()

def end_spawn(e):
	global start
	if alive:
		c.delete(shape)
	end = (e.x, e.y)
	enviroment.append(
		circle(True, start, (0,0), distance(start, end), (0, 0))
	)
	start = None



root.bind("<Button-1>", start_spawn)
root.bind("<Motion>", update_spawn)
root.bind("<ButtonRelease-1>", end_spawn)
root.bind("<Return>", update_enviroment)
root.mainloop()
