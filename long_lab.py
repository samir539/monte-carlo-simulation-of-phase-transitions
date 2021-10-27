## imports ##
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import math
from math import exp
from random import randrange
import copy
import tqdm as tqdm 


#########################################
    # G L O B A L  V A L U E S #
#########################################
cellA = 0 # Matrix atom
cellB = 1 # Alloy atom


def orderRandom(Z,f):
    """
    OrderRandom gives the distribution function of unlike neighbours for a completely random alloy.
    The order parameter here is the NUMBER OF AB BONDS around a lattice site.
    The binomial distribution is used to calculate the distribution fucntion.

    Input Arguments
        Z => The number of neighbouring sites per site
        f => The fraction of alloying atoms

    Output Arguments
        N_Rand => List of possible number of neighbours
        P_Rand => The probabilty distribution of the order parameter
    
    """
    # ==========================================================================================================================================
    
    Z = 4
    prob1 = 0
    P_Rand = np.linspace(0,0,Z+1)
    N_Rand = np.linspace(0,Z,Z+1)

    # Generate distribution based on binomial expression
    for n in range(5):
        prob1 = (math.factorial(Z)/((math.factorial(n))*( math.factorial(Z-n) ) ))*     ((f)*(f**(Z-n))*((1-f)**n) + (1-f) *(f**n)* ((1-f)**(Z-n)) )
        P_Rand[n] = prob1
        N_Rand[n] = n

    return(P_Rand, N_Rand)

#---------------------------------------------------------------------------------------------------------------------------------

def order2D(eqb_lattice):
    """
    ORDER2D produces a distribtion function of the orderparameter. 
    The order parameter is just the number of AB bondsaround a site.

    Input arguments
        config => configuration of the system at eqb

    Output arguments
        P_Alloy => List of possible number of neighbours
        N_Alloy => The probabilty disitribution of the order parameter
        mean_unlike_neigh => The mean number of unlike neighbours for a given configuration

    """
    total_unlike = 0
    zero_unlike = 0
    one_unlike = 0
    two_unlike = 0
    three_unlike = 0
    four_unlike = 0

    

    P_Alloy = np.linspace(0,0,5)
    N_Alloy = np.linspace(0,4,5)

    # total number of atoms under consideration 
    dimensions = eqb_lattice.shape
    total_atoms = (dimensions[0]-2) * (dimensions[1] - 2)


    #loop through the lattice 
    for i in range (1, dimensions[0]-1):
        for j in range(1,dimensions[1]-1): 

            if eqb_lattice[i,j] != eqb_lattice[i, j-1]:
                total_unlike = total_unlike + 1
            if eqb_lattice[i,j] != eqb_lattice[i-1,j]:
                total_unlike = total_unlike + 1
            if eqb_lattice[i,j] != eqb_lattice[i, j+1]:
                total_unlike = total_unlike + 1
            if eqb_lattice[i,j] != eqb_lattice[i+1, j]:
                total_unlike = total_unlike + 1
            
            if total_unlike == 4:
                four_unlike = four_unlike + 1
            elif total_unlike == 3:
                three_unlike = three_unlike + 1
            elif total_unlike == 2:
                two_unlike = two_unlike + 1
            elif total_unlike == 1:
                one_unlike = one_unlike + 1
            elif total_unlike == 0:
                zero_unlike = zero_unlike + 1
            
            total_unlike = 0



    # list with the number of atoms which have N_unlike neighbours
    unlike = [zero_unlike, one_unlike, two_unlike, three_unlike, four_unlike]

    #generating the distribution
    #Generating the distribution
    distr = [] 
    for i in unlike:
        prob = i/total_atoms
        distr.append(prob)

    #assigning the distribution
    P_Alloy = np.array(distr)

    # Mean unlike number of neighbours
    sum_unlike = 0
    for i in range(len(unlike)):
        sum_unlike = sum_unlike +  i*unlike[i]

    mean_unlike_neigh = sum_unlike/total_atoms


    return(P_Alloy, N_Alloy, mean_unlike_neigh)


# ------------------------------------------------------------------------------------------------------------------------
# This function is extra to the script
def EnergyCalc (Atom_iIndex, Atom_jIndex, Energy_AB, lattice):
    """
    This function calculates the energy of the four surrounding bonds of a given atom

    Inputs
        Atom_iIndex => The i index of the chosen element in the numpy array
        Atom_jIndex => The j index of the chosen element in the numpy array
        Energy_AB => Energy of a bond between an alloying and host atom
        lattice => numpy array representing the atoms 

    Outputs
        Energy => Energy of the four bonds surrounding the inputted atom

    """
    # Energy before

    # left bond
    if lattice[Atom_iIndex, Atom_jIndex] == lattice[Atom_iIndex, Atom_jIndex - 1]:
        Bond_One = 0
    else:
        Bond_One = Energy_AB
    # top bond
    if lattice[Atom_iIndex, Atom_jIndex] == lattice[Atom_iIndex - 1, Atom_jIndex]:
        Bond_Two = 0
    else:
        Bond_Two = Energy_AB
    # right bond
    if lattice[Atom_iIndex, Atom_jIndex] == lattice[Atom_iIndex, Atom_jIndex + 1]:
        Bond_Three = 0
    else:
        Bond_Three = Energy_AB
    # bottom bond
    if lattice[Atom_iIndex, Atom_jIndex] == lattice[Atom_iIndex + 1, Atom_jIndex]:
        Bond_Four = 0
    else:  
        Bond_Four = Energy_AB
    Energy = Bond_One + Bond_Two + Bond_Three + Bond_Four
    return(Energy)
    
# -------------------------------------------------------------------------------------------------------------------------------

def swapInfo(FirstAtom_iIndex, FirstAtom_jIndex, dab, lattice, Energy_AB):
    """
    This function calculates the difference in energy between a the swapped and unswapped state and returns the energy change and the coordinates of the 
    neighbouring atom

    Inputs
        FirstAtom_iIndex => This is the i index of the chosen atom (will be chosen randomly by the simulation)
        FirstAtom_jIndex => This is the j index of the chosen atom 
        dab => Direction of swap (selected at random)
        lattice => numpy array representing all the atoms
        Energy_AB => Energy of bond between two unlike atoms

    Returns
        SecondAtom_iIndex => The i coordinate of the atom chosen in the respective direction
        SecondAtom_jIndex => The j coordinate of the atom chosen in the respective direction
        energy_change => The energy change as a result of a swap
      

    """
    temp_lattice = copy.deepcopy(lattice)
    # print("This is lat from function \n", lattice)
    
    
    energy_before  = 0
    energy_after = 0
    a = getNeighbour(FirstAtom_iIndex, FirstAtom_jIndex, dab)
    SecondAtom_iIndex = a[0]
    SecondAtom_jIndex = a[1]

    # calculating the energy before
    energy_site_one = EnergyCalc(FirstAtom_iIndex, FirstAtom_jIndex, Energy_AB, temp_lattice)         # energy of chosen atom
    energy_site_two = EnergyCalc(SecondAtom_iIndex, SecondAtom_jIndex, Energy_AB, temp_lattice)     # energy of neighbouring atom
 

    # swapping the atoms to see
    temp_lattice[FirstAtom_iIndex,FirstAtom_jIndex], temp_lattice[SecondAtom_iIndex, SecondAtom_jIndex] = temp_lattice[SecondAtom_iIndex, SecondAtom_jIndex], temp_lattice[FirstAtom_iIndex, FirstAtom_jIndex]

    energy_site_oneSWAP =  EnergyCalc(FirstAtom_iIndex, FirstAtom_jIndex, Energy_AB, temp_lattice)
    energy_site_twoSWAP =  EnergyCalc(SecondAtom_iIndex, SecondAtom_jIndex, Energy_AB, temp_lattice)


    energy_before = (energy_site_one + energy_site_two - Energy_AB) 
    energy_after = (energy_site_oneSWAP + energy_site_twoSWAP - Energy_AB)
    # Calculate energy change
    energy_change = energy_after - energy_before
 
    return (SecondAtom_iIndex, SecondAtom_jIndex, energy_change)

#-------------------------------------------------------------------------------------------------------------------------

def getNeighbour(Atom_iIndex, Atom_jIndex, d12):
    """
    Returns to the user the coordinates of the neighbouring atom to the atom refereced in the function call.

    Inputs 
        Atom_iIndex => The i component of the atom chosen
        Atom_jIndex => The j componenet of the atom chosen 
    
    Outputs
        NeighbourAtom_iIndex => The i compoenent of the atom chosen  
        NeighbourAtom_jIndex => The j compoenent of the atom chosen

    """
    # inital empty array which will go on to contain the indicies of the chosen neighbour
    index = np.array([0,0])

    # Calculating the indicies of each possible neighbour for the atom inputed into the function
    a = np.array([Atom_iIndex, Atom_jIndex -1])
    b = np.array([Atom_iIndex - 1, Atom_jIndex])
    c = np.array([Atom_iIndex, Atom_jIndex + 1])
    d = np.array([Atom_iIndex + 1, Atom_jIndex])

    if d12 == "Alpha":
        index = a
    elif d12 == "Beta":
        index = b
    elif d12 == "Delta":
        index = c
    elif d12 == "Gamma":
        index = d

    NeighbourAtom_iIndex = index[0]
    NeighbourAtom_jIndex = index[1]

    return(NeighbourAtom_iIndex, NeighbourAtom_jIndex)
#-----------------------------------------------------------------------------------------------------------


def alloy2D(nBox, fAlloy, nSweeps, nEquil, Temp, Energy_AB, job):

    """
        This function generates the lattice, runs the monte carlo simulation and then subsequently analyses the data

        Inputs
                nBox => Dimensions of the (square) lattice
                fAlloy => Fraction of alloying atoms present in the lattice
                nSweeps => The number of steps for the monte carlo simulation
                Temp => The temperature chosen to run the simulation at
                Energy_AB => The bond energy of unlike bonds
                Job => Job
        

        Outputs 
                nBar => The average number of unlike neighbours
                Ebar => The average energy 
                C => The heat capacity
    """
    #################################
    ### GENERATE LATTICE PROPERLY ###
    #################################
    lattice = np.zeros((nBox, nBox))
    count = 0
    a = lattice.shape
    num_atoms = a[0]* a[1]
    while count < math.ceil(fAlloy*num_atoms):
        rand_row = randrange(0, a[0])
        rand_coloumn = randrange(0, a[1])
        if lattice[ rand_row, rand_coloumn] == 0:
            lattice[rand_row, rand_coloumn] = 1
            count = count + 1
    
    ####################################
    ### APPLYING BOUNDARY CONDITIONS ###
    ####################################
    shp = np.array(lattice.shape)
    top = lattice[0,:]   # coloumn to be reshaped
    left = lattice[:,0]
    bottom = lattice[shp[0] - 1,:] 
    right = lattice[:, shp[1] -1]   # coloumn to be reshaped

    lattice = np.vstack((bottom, lattice))
    lattice = np.vstack((lattice, top))

    left = np.append(left, 0)
    left = np.insert(left, 0, 0)
    left = left.reshape(-1,1)

    right = np.append(right, 0)
    right = np.insert(right, 0, 0)
    right = right.reshape(-1,1)

    lattice = np.hstack((lattice,left))
    lattice = np.hstack((right, lattice))

    #################################
    ### ENERGY OF INITAL  LATTICE ###
    #################################

    initial_lat_energy = 0
    b = lattice.shape

    for i in range(b[0]):
        for j in range(b[1] - 1):
            if lattice[i,j] == lattice[i, j + 1]:
                initial_lat_energy = initial_lat_energy + 0
            else:
                initial_lat_energy = initial_lat_energy + Energy_AB

    for k in range(b[0]-1):
        for l in range(b[1]):
            if lattice[k,l] == lattice[k + 1, l]:
                initial_lat_energy = initial_lat_energy + 0
            else:
                initial_lat_energy = initial_lat_energy + Energy_AB

    
    print("Initial energy of lattice is", initial_lat_energy)




    ###########################################
    ###  Carrying out the simulation to eqb ###
    ###########################################
    values = ["Alpha", "Beta", "Delta", "Gamma"]
    probability = [0.25, 0.25, 0.25, 0.25]
    energy = initial_lat_energy
    energy_list = []
    R = np.random.random_sample()

    for i in range(nEquil):


        shape_array = lattice.shape
        rand_row = randrange(2, shape_array[0]-2)                                 # Generate lattice point
        rand_column = randrange(2, shape_array[1]-2)                            # Generate lattice point
        direction = np.random.choice(values, p = probability)                   # Generate random direction

        a = swapInfo(rand_row, rand_column, direction, lattice, Energy_AB)
        if a[2] <= 0:
            lattice[rand_row,rand_column], lattice[a[0], a[1]] = lattice[a[0], a[1]], lattice[rand_row, rand_column]
            energy = energy + a[2]
            energy_list.append(energy)
        elif a[2] > 0:
            R = np.random.random_sample()
            if(exp(-(a[2])/ ((0.00008617332)*(Temp)))) > R: 
                lattice[rand_row,rand_column], lattice[a[0], a[1]] = lattice[a[0], a[1]], lattice[rand_row, rand_column]
                energy = energy + a[2]
                energy_list.append(energy)
        

    eqb_lattice = lattice
    

    ########################################
    ####### Data analysis at eqb ###########
    ######################################## 

    distr_analysis = order2D(eqb_lattice)
    rand_dist = orderRandom(4, fAlloy)


    # Plot the configuration at eqb
    # CONFIG AT EQB
    config_plot = np.zeros((nBox+2, nBox+2))
    config_plot[0:nBox + 2, 0:nBox + 2] = lattice
    plt.figure(0)
    plt.pcolor(config_plot)
    plt.title('Configuration at equilibrium at Eam {}, Temp {}, and composition {}'.format(Energy_AB, Temp, fAlloy))
    plt.savefig(str(job)+'Configuration at equilibrium at Eam {}, Temp {}, and composition {}.png'.format(Energy_AB, Temp, fAlloy) )
    # plt.show()
    plt.clf()

    # ENERGY PLOT (TO EQB)
    iteration_list = list(range(nEquil))
    x  = np.linspace(0,len(energy_list),len(energy_list)) 
    y = energy_list   
    plt.figure(1)
    plt.plot(x,y, color = "r")
    plt.title('Plot of energy at Eam {}, Temp {}, and composition {}'.format(Energy_AB, Temp, fAlloy))
    plt.ylabel('Energy')
    plt.xlabel('Steps')
    plt.savefig(str(job)+ 'Plot of energy at Eam {}, Temp {}, and composition {}.png'.format(Energy_AB, Temp, fAlloy))
    # plt.show()
    plt.clf()

    # DISTRIBUTION OF UNLIKE NEIGHBOURS
    num_unlike = ('0 unlike', '1 unlike', '2 unlike', '3 unlike', '4 unlike')
    y_pos = np.arange(len(num_unlike))
    plt.bar(y_pos + 0.00, distr_analysis[0], color= "k", width = 0.25, label = "neighbour distribution at eqb" )
    plt.bar(y_pos + 0.25, rand_dist[0], color = "r", width = 0.25, label = "neighbour distribution for random lattice")
    plt.title("Distribution of unlike neighbours at Eam {}, Temp {}, and composition {}".format(Energy_AB, Temp, fAlloy), fontsize = 10)
    plt.xticks(y_pos, num_unlike)
    plt.legend(fontsize = 10)
    plt.savefig(str(job)+ "Distribution of unlike neighbours at Eam {}, Temp {}, and composition {}.png".format(Energy_AB, Temp, fAlloy))
    # plt.show()
    plt.clf()
    
    ##############################
    ##### CONTINUING AT EQB ######
    ##############################

    Energy_eqb_lat = 0
    n_bar = 0
    Total_energy = 0
    Total_energy_squared = 0
    

    # WE LOOK TO OBTAIN:
                        # 1.) mean nBar 
                        # 2.) Ebar
                        # 3.) C

    for i in range(nSweeps - nEquil):
        shape_array = eqb_lattice.shape
        rand_row = randrange(2, shape_array[0]-2)                               # Generate lattice point
        rand_column = randrange(2, shape_array[1]-2)                            # Generate lattice point
        direction = np.random.choice(values, p = probability)                   # Generate random direction

        a = swapInfo(rand_row, rand_column, direction, eqb_lattice, Energy_AB)
        if a[2] <= 0:
            eqb_lattice[rand_row,rand_column], eqb_lattice[a[0], a[1]] = eqb_lattice[a[0], a[1]], eqb_lattice[rand_row, rand_column]
            energy = energy + a[2]
            energy_list.append(energy)
            
        elif a[2] > 0:
            R = np.random.random_sample()
            
            if(exp(-(a[2])/ ((0.00008617332)*(Temp)))) > R: 
                eqb_lattice[rand_row,rand_column], eqb_lattice[a[0], a[1]] = eqb_lattice[a[0], a[1]], eqb_lattice[rand_row, rand_column]
                energy = energy + a[2]
                energy_list.append(energy)
                
        
        #Calculating the average value for nBar
        new_n_bar = order2D(eqb_lattice)
        n_bar = n_bar + new_n_bar[2]

        #Calculating Ebar (the average energy) [loop through each lattice site to calculate energy of system]
        for i in range(b[0]):
            for j in range(b[1] - 1):
                if eqb_lattice[i,j] == eqb_lattice[i, j + 1]:
                    Energy_eqb_lat = Energy_eqb_lat + 0
                else:
                    Energy_eqb_lat = Energy_eqb_lat + Energy_AB

        for k in range(b[0]-1):
            for l in range(b[1]):
                if eqb_lattice[k,l] == eqb_lattice[k + 1, l]:
                    Energy_eqb_lat = Energy_eqb_lat + 0
                else:
                    Energy_eqb_lat = Energy_eqb_lat + Energy_AB     
                    

        Total_energy = Total_energy + Energy_eqb_lat
        Total_energy_squared = Total_energy_squared + Energy_eqb_lat**2   
        Energy_eqb_lat = 0
    
    mean_n_bar = n_bar/(nSweeps - nEquil)
    Ebar = Total_energy/(nSweeps - nEquil)      # mean of energy
    E2bar = Total_energy_squared/(nSweeps - nEquil)
    print("No. of eqb sweeps", nSweeps- nEquil)
    print("the total energy is", Total_energy)
    print("the total energy squared is", Total_energy_squared)
    print("the mean energy is", Ebar)
    print("the mean total energy squared is", E2bar)
    C = (E2bar - Ebar*Ebar)/((Temp**2)*(0.0008617332))
    print("the heat capacity is", C)

    # return(eqb_lattice, distr_analysis[2], mean_n_bar,Temp)
    return(mean_n_bar, Ebar, C)

#--------------------------------------------------------------------------------------------------------------------------------------------------------

#####################
### MAIN FUNCTION ###
#####################


def main():

    # SIMULATION PARAMETERS
    nBox = 10
    nEquil = 20000
    nSweeps = 75000
    fAlloy_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    T_list = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800 , 1900, 2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4300, 4600, 4900, 5100 ]
    Eam_list = [0.1, 0.0, -0.1]

    # Open file to save data
    file = open("data.csv", "w")
    file.write('Job number, Alloy fraction, Temperature(K), Unlike bond energy (eV), Average number of unlike neighbours, Average energy (eV), Heat capacity (kB) \n')
    
    #Looping over values
    count = 0
    for fAlloy in fAlloy_list:
        for Temp in T_list:
            for Energy_AB in Eam_list:
                count = count + 1
                job = '{:04d}'.format(count)

                # Echos parameters back to the user
                print("")
                print("Simulation", job)
                print("---------------------")
                print("Cell size                =", nBox)
                print("Alloy fraction           =", fAlloy)
                print("Total number of moves    =", nSweeps) 
                print("Number of equilibration moves  =", nEquil)
                print("Temperature                    =", Temp, "K")
                print("Bond energy                    =", Energy_AB, "eV")

                # Run the simulation
                mean_n_bar, Ebar, C = alloy2D(nBox, fAlloy, nSweeps, nEquil, Temp, Energy_AB, job)
                # Write out the statistics
                file.write('{0:4d}, {1:6.4f}, {2:8.2f}, {3:5.2f}, {4:6.4f}, {5:14.7g}, {6:14.7g} \n'.format(count, fAlloy, Temp, Energy_AB, mean_n_bar, Ebar, C))

    # close the file
    file.close()

    # sign off
    print('')
    print('Simulations completed')

# Ensure main is invoked
if __name__ == "__main__":
    main()           



