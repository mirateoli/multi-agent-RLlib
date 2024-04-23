from vedo import *
import design_spaces as DS
import numpy as np

def select_pts(obstacles):
    # open a vedo window and select start/end points for pipes
    # points must be points along obstacle mesh

    def empty_button(obj, ename):
        return


    def format_point_list(pt_list):
        res = "Current Points Selected: \n"
        if len(pt_list) > 0:
            for pt in pt_list:
                res+= str(pt.tolist()) + "\n"
        else:
            res+= "[ ]"
        return res

    def hover_func(event):
        if not event.actor:
            return
        if hasattr(event.actor, 'name'):
            if event.actor.name == "Box":
                p = np.round(event.picked3d)
                if p is None:
                    return
                pt = event.actor.closest_point(p, n=1)
                sph = Sphere(pt, r=sphere_radius).alpha(0.1)
                sph.name = "tmp_sphere"
                plotter.remove("tmp_sphere").add(sph).render()

    def remove_pt_func(event):
        nonlocal point_list
        if event.actor.name[0:8] == "mysphere":
            plotter.remove(event.actor.name)
            p = event.actor.pos()
            point_list = [arr for arr in point_list if not np.array_equal(arr, p)]
            txt.text(format_point_list(point_list))

    def select_pt_func(event):
        nonlocal point_list
        if not event.actor:
            return
        if hasattr(event.actor, 'name'):
            if event.actor.name == "tmp_sphere":
                p = np.rint(event.actor.pos())
                plotter.remove("tmp_sphere")
                sph = Sphere(p, r=sphere_radius)
                sph.name = "mysphere" + str(len(point_list))
                plotter.add(sph).render()
                point_list.append(p)
                txt.text(format_point_list(point_list))

            if event.actor.name == "Button":
                nonlocal num_pipes
                num_pipes += 1
                pipe_txt.text("Pipes to Route:"+str(num_pipes))
    
    plotter = Plotter()
    sphere_radius = 0.3
    point_list = []
    num_pipes = 1
    plotter.reset_camera()

    plotter.add_callback('mouse hover', hover_func)
    plotter.add_callback('mouse left click', select_pt_func)    
    plotter.add_callback('mouse right click', remove_pt_func)    

    txt = Text2D("Click Away!", pos="bottom-left", bg='white', font='Calco')
    plotter.add(txt)

    pipe_txt = Text2D("Pipes to Route:"+str(num_pipes),pos="top-left", bg='white', font='Calco', bold=True)
    plotter.add(pipe_txt)

    DS.room(plotter, DS.length, DS.width, DS.height)
    for obstacle in obstacles:
        bounding_box = obstacle.tolist()
        box = Box(size=bounding_box)
        box.color(c=(135,206,250))
        box.opacity(0.7)
        plotter.add(box)

    button = plotter.add_button(
        empty_button,
        pos=(0.5, 0.90),
        states=["Add Pipe"],
        c=["w"],
        bc=["db"],
        font="courier",
        size=30,
        bold=True,
        italic=False
    )

    button.name = "Button"

    plotter.camera.SetPosition([50, 120, 60])
    plotter.show()

    return num_pipes, point_list


# num_pipes, key_pts = select_pts(DS.obstacles)
# print(num_pipes)
# print(key_pts)


