import numpy as np
from scipy.linalg import solve as slv
import matplotlib.pyplot as plt
import math

def main():

    dimensions = 3  # sets the dimensions of the problem

    nodes = read_nodes_from_text("nodes.txt")
    elements = read_elements_from_text("elements.txt")
    forces = read_forces_from_text("forces.txt")
    disp = read_displacements_from_text("displacements.txt")

    # checking each dataset to see if it is one dimensional since my code specifically wants 2 dimensional arrays.
    # If 1d it makes it 2d of size 1 by N where N is the amount ov values that need to be stored.
    if elements.ndim == 1:
        elements = elements.reshape((1, -1))
    if disp.ndim == 1:
        disp = disp.reshape((1, -1))
    if forces.ndim == 1:
        forces = forces.reshape((1, -1))
    else:
        exit(1)

    u, external, n, v, m1, m2 = solver(nodes, elements, forces, disp, dimensions)


    u = u.round(5)
    external = external.round(5)
    n = n.round(5)
    v = v.round(5)
    m1 = m1.round(5)
    m2 = m2.round(5)

    print("\nu\n", u.reshape((-1,1)))
    print("\nFe\n", external.reshape((-1,1)))
    print("\nn\n", n.reshape((-1, 1)))
    print("\nv\n", v.reshape((-1, 1)))
    print("\nm1\n", m1.reshape((-1, 1)))
    print("\nm2\n", m2.reshape((-1, 1)))

    # plt.figure(figsize=(10, 5))
    #
    # node_tracker = []
    # node_counter = 1
    # el_counter = 1
    # x_1 = []
    # x_2 = []
    # y_1 = []
    # y_2 = []
    #
    # # I = b * pow(h, 3) / 12
    #
    # for i, row in enumerate(elements):
    #     # using your in class pseudo code, you used indexing starting at 1 so i subtract each row value by 1 to get 0 indexing
    #     x_1.append(nodes[int(row[0]) - 1, 0])
    #     y_1.append(nodes[int(row[0]) - 1, 1])
    #     x_2.append(nodes[int(row[1]) - 1, 0])
    #     y_2.append(nodes[int(row[1]) - 1, 1])
    #
    #     plt.plot([x_1[i], x_2[i]], [y_1[i], y_2[i]])
    #     # plt.text((x_1[i] + (x_2[i] - x_1[i]) / 2) - 0.5, (y_1[i] + (y_2[i] - y_1[i]) / 2) + 0.01, "E{}".format(i), fontsize=10)
    #
    #     if (x_1[i], y_1[i]) not in node_tracker:
    #         plt.text(x_1[i], y_1[i], "N1", fontsize=10, fontweight = 'bold')
    #         node_tracker.append(tuple([x_1[i], y_1[i]]))
    #         node_counter += 1
    #     if (x_2[i], y_2[i]) not in node_tracker:
    #         print(x_2[i], y_2[i])
    #         plt.text(x_2[i], y_2[i], "N{}".format(node_counter), fontsize=10, fontweight = 'bold')
    #         node_tracker.append(tuple([x_2[i],y_2[i]]))
    #         node_counter += 1
    # print(node_tracker)
    # plt.show()
    # # make u a 2d array
    # u = u.reshape((-1, 2))

    # for row_num, value in enumerate(u):
    #     print("\nNode",row_num + 1,":\nx displacement =", value[0],"\ny displacement =", value[1])
    #
    # # Plotting the truss structure to depict which nodes the data represents
    # figure, fig1 = plt.subplots(1, 1)
    #
    # for row in elements:
    #     # using your in class pseudo code, you used indexing starting at 1 so i subtract each row value by 1 to get 0 indexing
    #     x_1 = nodes[int(row[0]) - 1, 0]
    #     y_1 = nodes[int(row[0]) - 1, 1]
    #     x_2 = nodes[int(row[1]) - 1, 0]
    #     y_2 = nodes[int(row[1]) - 1, 1]
    #     x_3 = nodes[int(row[2]) - 1, 0]
    #     y_3 = nodes[int(row[2]) - 1, 1]
    #     x_4 = nodes[int(row[3]), 0]
    #     y_4 = nodes[int(row[3]), 1]
    #     fig1.plot([x_1, x_2], [y_1, y_2])
    # fig1.text(x_1-0.1, y_1+0.01, "node: 3", fontsize=10)
    # fig1.text(x_2-0.1, y_2-0.04, "node: 4", fontsize=10)
    # fig1.text(x_3, y_3-0.04, "node: 1", fontsize=10)
    # fig1.text(x_4, y_4+0.01, "node: 2", fontsize=10)
    # plt.title("Truss Structure With Global Node Convention Labeled")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    #
    # plt.show()

# solves for the u
def solver(nodes, elements, forces, disp, dimensions):
    number_of_elements = len(elements)  # sets the number of elements to the amount of rows in the elements file (this is why I skip the first column)

    number_of_nodes = len(nodes)  # sets the number of nodes to the amount of rows in the nodes file (this is why I skip the first column)
    u = np.zeros((dimensions*number_of_nodes))  # sets u as an array of 0's of size 1 by (dimensions * number of nodes)
    force = np.zeros((dimensions*number_of_nodes))  # sets force as an array of 0's of size 1 by (dimensions * number of nodes)
    global_stiffness = np.zeros((dimensions*number_of_nodes, dimensions*number_of_nodes))  # sets global stiffness as an array of 0's of size (dimensions * number of nodes) by (dimensions * number of nodes)
    force_external = np.zeros((dimensions*number_of_nodes))  # sets external force as an array of 0's of size (dimensions * number of nodes) by 1
    force_internal = np.zeros((number_of_elements))  # sets internal force as an array of 0's of size number of elements by 1
    axial_strain = np.zeros((number_of_elements))  # sets axial strain as an array of 0's of size number of elements by 1
    stress = np.zeros((number_of_elements))  # sets stress as an array of 0's of size number of elements by 1

    # create global node dof matrix
    global_conn = np.zeros((number_of_nodes, dimensions), dtype=int) # sets the global node dof matrix to be zeros and of size node number by dimensions
    for i in range(number_of_nodes):
        for j in range(dimensions):
            global_conn[i, j] = 3*i + j  #sets each value as (previous value) + 1

    # create and fill the displacement vectors
    node_with_displacement = disp[:, 0].astype(int)-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
    displacement_dof = disp[:, 1].astype(int)-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
    displacement_value = disp[:, 2] # read in all displacement values
    displacement_dof = global_conn[node_with_displacement, displacement_dof]  # set the displacement degree of freedom using the global positions
    u[displacement_dof] = displacement_value # update u to include the initial known (given) displacements



    # Assign Force conditions
    node_with_force = forces[:, 0].astype(int)-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
    force_dof = forces[:, 1].astype(int)-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
    force_value = forces[:, 2]  # read in all force values
    force_dof = global_conn[node_with_force, force_dof]  # set the force degree of freedom using its global positions
    force[force_dof] = force_value #update force to include the given forces

    for element_num, row in enumerate(elements): # grabs element number and each row's data
        n1 = int(row[0])-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
        n2 = int(row[1])-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
        E_of_element = row[2]  # set E of each element using data from elements
        A_of_element = row[3]  # set A of each element using data from elements
        I_of_element = row[4]
        EA_of_element = E_of_element*A_of_element
        EI_of_element = E_of_element*I_of_element
        # EI_of_element = 0.001

        # store the x and y variable of both nodes of the truss
        x_1 = nodes[n1, 0]
        y_1 = nodes[n1, 1]
        x_2 = nodes[n2, 0]
        y_2 = nodes[n2, 1]

        truss_length = math.sqrt(pow(x_2-x_1,2) + pow(y_2-y_1,2))  # calculates the length of the truss using the previously recorded coordinates.

        # calculate both sin and cosine of the angle theta using small angle calculations
        cos = (x_2-x_1)/truss_length
        sin = (y_2-y_1)/truss_length
        # print(cos)
        EAL = EA_of_element/truss_length  # define (E*A)/L
        EIL = EI_of_element /truss_length
        EIL2 = EI_of_element/pow(truss_length, 2)
        EIL3 = EI_of_element/pow(truss_length, 3)
        # print(EI_of_element)
        R = np.array([[cos, sin, 0, 0, 0, 0],
                     [-sin, cos, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, cos, sin, 0],
                     [0, 0, 0, -sin, cos, 0],
                     [0, 0, 0, 0, 0, 1],
                     ])

        # RT = R.T

        # Local stiffness for element using the provided sin cosine convention
        local_stiffness = np.array([
            [EAL, 0, 0, -EAL, 0, 0],
            [0, 12*EIL3, 6*EIL2, 0, -12*EIL3, 6*EIL2],
            [0, 6*EIL2, 4*EIL, 0, -6*EIL2, 2*EIL],
            [-EAL, 0, 0, EAL, 0, 0],
            [0, -12*EIL3, -6*EIL2, 0, 12*EIL3, -6*EIL2],
            [0, 6*EIL2, 2*EIL, 0, -6*EIL2, 4*EIL]
        ])
        # print("local = \n", local_stiffness)
        # print(R)
        local_stiffness = R.T@local_stiffness@R
        # print(global_stiffness)
        # set up global stiffness and force
        # print(force)
        # print(global_stiffness)
        for node_counter in range(2):  # loop over all rows
            for dof_counter in range(dimensions):
                local_dof_counter = dimensions*node_counter+dof_counter
                global_node_counter = int(elements[element_num, node_counter])-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
                global_dof_counter = global_conn[global_node_counter, dof_counter]

                # Check if the value of global_dof_counter is contained withing the array displacement_dof
                # if global_dof_counter in displacement_dof:
                #     # print(global_dof_counter)
                #     # global_stiffness[global_dof_counter, :] = 0
                #     # global_stiffness[:, global_dof_counter] = 0
                #     global_stiffness[global_dof_counter, global_dof_counter] = 1  # update the global stiffness matrix
                #     force[global_dof_counter] = u[global_dof_counter]
                #     continue

                # Loop over all columns
                for node_counter_2 in range(2):
                    for dof_counter_2 in range(dimensions):
                        local_dof_counter_2 = dimensions*node_counter_2+dof_counter_2
                        global_node_counter_2 = int(elements[element_num, node_counter_2])-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
                        global_dof_counter_2 = global_conn[global_node_counter_2, dof_counter_2]
                        # print(global_dof_counter)
                        global_stiffness[global_dof_counter,global_dof_counter_2] += local_stiffness[local_dof_counter,local_dof_counter_2] # update the global stiffness matrix

                        # Check if the value of global_dof_counter2 is not contained withing the array displacement_dof
                        # if global_dof_counter_2 not in displacement_dof:
                        #     global_stiffness[global_dof_counter,global_dof_counter_2] += local_stiffness[local_dof_counter,local_dof_counter_2] # update the global stiffness matrix
                        # else:
                        #     force[global_dof_counter] -= local_stiffness[local_dof_counter,local_dof_counter_2]*u[global_dof_counter_2]

    # print(global_stiffness)
    global_stiffness1 = global_stiffness
    # print(global_stiffness)
    force1 = force
    # print(force)
    for element_num, row in enumerate(elements):
        for node_counter in range(2):  # loop over all rows
            for dof_counter in range(dimensions):
                local_dof_counter = dimensions * node_counter + dof_counter
                global_node_counter = int(elements[element_num, node_counter]) - 1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
                global_dof_counter = global_conn[global_node_counter, dof_counter]
                # Check if the value of global_dof_counter is contained withing the array displacement_dof
                if global_dof_counter in displacement_dof:
                    # print(global_dof_counter)
                    global_stiffness[global_dof_counter, :] = 0
                    global_stiffness[:, global_dof_counter] = 0

                    global_stiffness[global_dof_counter, global_dof_counter] = 1  # update the global stiffness matrix
                    force[global_dof_counter] = u[global_dof_counter]
                    # print("stiff \n",force1)
                    continue

                for node_counter_2 in range(2):
                    for dof_counter_2 in range(dimensions):
                        local_dof_counter_2 = dimensions*node_counter_2+dof_counter_2
                        global_node_counter_2 = int(elements[element_num, node_counter_2])-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
                        global_dof_counter_2 = global_conn[global_node_counter_2, dof_counter_2]
                        # Check if the value of global_dof_counter2 is not contained withing the array displacement_dof
                        if global_dof_counter_2 in displacement_dof:
                            # print(global_dof_counter_2)
                            # print(u[global_dof_counter_2])
                            force[global_dof_counter] -= local_stiffness[local_dof_counter,local_dof_counter_2]*u[global_dof_counter_2]



    # print(force)
    # print(global_stiffness)
    u = slv(global_stiffness, force)  # use scipy.linalg to solve the linear system
    external = global_stiffness1 @ u
    # print(global_stiffness)
    v = np.zeros((number_of_elements))  # sets axial strain as an array of 0's of size number of elements by 1
    m1 = np.zeros((number_of_elements))
    m2 = np.zeros((number_of_elements))
    # New Strain and force solver
    for el_num, row in enumerate(elements):
        # print(row)
        n1 = int(row[0])-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
        n2 = int(row[1])-1  # using your in class pseudo code, you used indexing starting at 1 so i subtract each by 1 to get 0 indexing
        E_element = row[2]  # set E of each element using data from elements
        A_element = row[3]  # set A of each element using data from elements
        I_of_element = row[4]
        EI_of_element = E_of_element*I_of_element
        # EI_of_element = 0.001

        # store the x and y variable of both nodes of the truss
        x_1 = nodes[n1, 0]
        y_1 = nodes[n1, 1]
        x_2 = nodes[n2, 0]
        y_2 = nodes[n2, 1]

        # store the u and v with respect to their global positions
        u_1 = u[global_conn[n1, 0]]
        v_1 = u[global_conn[n1, 1]]
        u_2 = u[global_conn[n2, 0]]
        v_2 = u[global_conn[n2, 1]]
        theta_1 = u[global_conn[n1, 2]]
        theta_2 = u[global_conn[n2, 2]]


        # print(n1)
        # print(n2)
        truss_length = math.sqrt(pow(x_2-x_1,2) + pow(y_2-y_1,2))  # calculates the length of the truss using the previously recorded coordinates.

        # calculate both sin and cosine of the angle theta using small angle calculations
        cos = (x_2 - x_1) / truss_length
        sin = (y_2 - y_1) / truss_length

        ua1 = u_1*cos+v_1*sin
        ua2 = u_2*cos+v_2*sin
        ut1 = -u_1*sin+v_1*cos
        ut2 = -u_2*sin+v_2*cos
        # print(truss_length)
        # calculate axial strain of each element
        axial_strain[el_num] = (ua2 - ua1)/truss_length# + (v_2 - v_1) * sin/truss_length

        # calculate the stress of each element
        stress[el_num] = axial_strain[el_num] * E_element
        n = stress
        # calculate the internal force at each element
        force_internal[el_num] = stress[el_num]*A_element
        # print(force_internal)
        v[el_num] = (12*EI_of_element/pow(truss_length, 3)*(ut1-ut2)) + (6*EI_of_element/pow(truss_length, 2) * (theta_1+theta_2))
        m1[el_num] = (6*EI_of_element/pow(truss_length,2)*(ut2-ut1)-(4*EI_of_element/truss_length*theta_1) - (2*EI_of_element/truss_length)*theta_2)
        m2[el_num] = (6*EI_of_element/pow(truss_length,2)*(ut1-ut2)+(4*EI_of_element/truss_length*theta_2) + (2*EI_of_element/truss_length)*theta_1)



        # calculate the external forces using the internal forces with respect to their global positions
        force_external[global_conn[n1, 0]] = force_external[global_conn[n1, 0]] - n[el_num] * cos - v[el_num] * sin
        force_external[global_conn[n1, 1]] = force_external[global_conn[n1, 1]] - n[el_num] * sin + v[el_num] * cos
        force_external[global_conn[n1, 2]] = force_external[global_conn[n1, 2]] - m1[el_num]
        force_external[global_conn[n2, 0]] = force_external[global_conn[n2, 0]] + n[el_num] * cos + v[el_num] * sin
        force_external[global_conn[n2, 1]] = force_external[global_conn[n2, 1]] + n[el_num] * sin - v[el_num] * cos
        force_external[global_conn[n2, 2]] = force_external[global_conn[n2, 2]] + m2[el_num]
    # # return force_internal, force_external, u, axial_strain, stress

    return u, force_external, n, v, m1, m2

# all input functions
def read_nodes_from_text(nodes_text):
    nodes = np.genfromtxt(nodes_text, skip_header=1)  # the first row is not used in my code
    nodes = nodes[:, 1:]  # removes the first column since I do not use the node number column of nodes that you provide
    return nodes

def read_elements_from_text(elements_text):
    elements = np.genfromtxt(elements_text, skip_header=1)  # the first row is not used in my code
    elements = elements[:, 1:]  # removes the first column since I do not use the element number column of elements that you provide
    return elements

def read_forces_from_text(forces_text):
    forces = np.genfromtxt(forces_text, skip_header=1)  # the first row of input file is not used in my code
    return forces

def read_displacements_from_text(displacements_text):
    displacements = np.genfromtxt(displacements_text, skip_header=1)  # the first row of input file is not used in my code
    return displacements

# run the main function
if __name__=="__main__":
    main()