from numpy import *

h = 0.01
t = arange(0,100*pi,h)
y = [1,0]
y1 = [1,0]
y2 = [1,0]
y3 = [1,0]
allowable_error = 0.0000001
h_average = 0
steps = 0

def RK_timestep(yLower, yUpper, step):
	global y1, y2, y3
	k = zeros((2,4))
	k[0,0] = step	* yLower[1]
	k[1,0] = -step	* yLower[0]
	k[0,1] = step	*(yLower[1] + k[1,0] /2 )
	k[1,1] = -step	*(yLower[0] + k[0,0] /2 )
	k[0,2] = step	*(yLower[1] + k[1,1] /2 )
	k[1,2] = -step	*(yLower[0] + k[0,1] /2 )
	k[0,3] = step	*(yLower[1] + k[1,2] )
	k[1,3] = -step	*(yLower[0] + k[0,2] )

	yUpper[0] = yLower[0] + k[0,0]/6 + k[0,1]/3 + k[0,2]/3 + k[0,3]/6
	yUpper[1] = yLower[1] + k[1,0]/6 + k[1,1]/3 + k[1,2]/3 + k[1,3]/6
	return

t = 0
while (t<100*pi):
	RK_timestep(y, y1, h/2.0)
	RK_timestep(y1, y2, h/2.0)
	RK_timestep(y, y3, h)
	local_error = fabs(y3[0]-y2[0])
	if(local_error > allowable_error):
		h = h/2
	elif(local_error < 0.2*allowable_error):
		h = 1.1*h
	y[0] = y2[0]+(y2[0]-y3[0])/15
	y[1] = y2[1]+(y2[0]-y3[0])/15
	h_average += h
	t+=h
	steps+=1

print( "Final error = " + str(cos(t) - y[0]) )
print( "Average h = " + str(t/steps) )
