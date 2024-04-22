from vedo import *

obstacles = np.array([(2,6,2,8,0,5),(9,14,8,12,0,4),(0,2,12,18,0,3),(18,20,3,8,0,9)])  # bounding box (xmin,xmax,ymin,ymax,zmin,zmax)


# Define room dimensions
length = 20
width = 20
height = 16

# Create a Plotter instance
plotter = Plotter()

def room(plotter, room_length, room_width, room_height):


    # Create the floor using a Box
    floor = Box(pos=[room_length / 2, room_width / 2,0],
                    length=room_length,
                    width=room_width,
                    height=0.1)  # A thin box to represent the floor
    floor.color("lightgrey")  # Optional: set the color of the floor


    # Create the back wall using a Box
    back_wall = Box(pos=[room_length / 2, 0, room_height / 2],
                        length=room_length,
                        width=0.1,  # Thin width for the wall
                        height=room_height)
    back_wall.color("white")  # Optional: set the color of the wall

    # Create the left wall using a Box
    left_wall = Box(pos=[0, room_width / 2, room_height / 2],
                        length=0.1,  # Thin width for the wall
                        width=room_width,
                        height=room_height)
    left_wall.color("white")  # Optional: set the color of the wall



    # Add the floor and walls to the plotter
    plotter += floor
    plotter += back_wall
    plotter += left_wall


    # Set the camera position for a good view of the room
    plotter.camera.SetPosition([5, 12, 6])
    plotter.camera.SetFocalPoint([0, 0, 0])
    plotter.camera.SetViewUp([0, 0, 1])

    # Create a light source in the plotter
    light_source = Light(pos=(room_length, room_width+5, room_height), focal_point=(room_length / 2, room_width / 2, room_height / 2), angle=360, c=None, intensity=1)
    light_source.SetColor([1, 1, 1])  # Set color (R, G, B) format

    # Add the light source to the plotter
    plotter.add(light_source)

# room(plotter, length, width, height)

# if obstacles is not None:
#     for obstacle in obstacles:
#         bounding_box = obstacle.tolist()
#         box = Box(size=bounding_box)
#         box.color(c=(135,206,250))
#         box.opacity(1)
#         plotter.add(box)
#     plotter.show(axes=1)
#     # show(key_pts[0], key_pts[1],Points(pts[0]),Points(pts[1]),ln[0], ln[1],box,axes=1).close()
# else:
#     plotter.show(axes=1, interactive=True).close()