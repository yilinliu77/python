import math

def lonLat2Mercator(v_lonLat):
	x = v_lonLat[0] * 20037508.34 / 180
	y = math.log(math.tan((90 + v_lonLat[1]) * math.pi / 360.)) / (math.pi / 180.)
	y = y * 20037508.34 / 180.
	return (x,y)
