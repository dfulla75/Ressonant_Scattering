######################################################################

# Software: Resonant Scattering and Diffraction Visualisation
# Daniel Fulla Marsa 
# DESY - Petra III Extension
# Email: daniel.fulla.marsa@desy.de

#### Under Development ####

######################################################################

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

import matplotlib.gridspec as gridspec


def file_open(file_name, num_intensity):
    
    data_file = open(file_name, 'r').readlines()
    only_intensities = data_file[num_intensity].split('  ')

    return data_file, only_intensities


def data_extractor(data_file):

    data_line_9 = data_file[8].split('Dimension 1 scale=')[1]
    data_energy = data_line_9.split(' ')

    data_angle = data_file[11].split('Dimension 2 scale=')[1]
    data_angle = data_angle.split(' ')

    print 'Energy range, from%s eV to %s eV with %i points'%(data_energy[0], data_energy[-1], len(data_energy))
    print 'Angle range, from %s deg to %s deg with %i points'%(data_angle[0], data_angle[-1], len(data_angle))



    return data_energy, data_angle

'''

def plot_several(number_of_plots):

    #number_of_plots = 1
    line_start_intensity = 49
    
    for i in range(number_of_plots):
        only_data, data_line_49 = file_open('071016_0108.txt', line_start_intensity + i)
        plt.plot(data_line_49)
        
    print 'number of scans: %i'%len(only_data)
    print 'number of points per scan: %i'%len(data_line_49)

    plt.show()


#plot_several()

'''

def plot_with_angle(file_name):
    
        #number_of_plots = 341
        #number_of_plots = 581
        line_start_intensity = 49
        
        
        data_file ,only_intensities = file_open(file_name, 49) # wtf ... ineficiency
        data_energy, data_angle, = data_extractor(data_file)

        #print 'You have %i energy scans'%len(data_energy)

        number_of_plots = len(data_energy)

        data_all_intensities = []
        
        for i in range(number_of_plots):
            
            data_file, only_intensities = file_open(file_name, line_start_intensity + i)
            data_all_intensities.append(only_intensities[1:]) # attention first pixel being cut ??????????there is an artifact at the first point---all go to 0??????????????
            #plt.plot(data_angle, only_intensities[0:-1])

        #plt.show()

        return data_angle, data_all_intensities, data_energy



    
def cut_plot(name_file, start_energy, end_energy, angle_low, angle_high, times_delta_e):

    print name_file, start_energy, end_energy, angle_low, angle_high, times_delta_e
    
    data_angle, data_all_intensities, data_energy = plot_with_angle(name_file)


    print 'Introduce cutting angle between %s and %s deg'%(data_angle[0], data_angle[-1])

    if start_energy < float(data_energy[0]):

        print 'Lowest energy is %s ' %data_energy[-1]

    if end_energy > float(data_energy[-1]):

        print 'Lowest energy is %s ' %data_energy[0]


    low =  data_angle.index(angle_low)
    high = data_angle.index(angle_high)

    #energy_cut = data_energy[low:high]
    intensities_cut = []

    #intensities_cut_filtered = []

    se = str(start_energy)
    ee = str(end_energy)
    
    energy_index_start =  data_energy.index(se)
    energy_index_end =  data_energy.index(ee)

    energy_cut = data_energy[energy_index_start:energy_index_end]


    increment = energy_index_end - energy_index_start
    
    delta_e = (float(end_energy)-float(start_energy))/float(increment)

    integrate_delta = times_delta_e * delta_e
    intensities_to_add = []
    averaged_intensities = []

    factor_delta = int(integrate_delta / delta_e)

    j = energy_index_start
    
    angle = data_angle[low:high]

    all_intensities_averaged = []

    averaged_list_to_plot = []

    while j < energy_index_end:


        for i in range(factor_delta):

            intensities_to_add.append(map(float,data_all_intensities[j + i][low:high]))

        j = j + factor_delta
        averaged_list = sum(map(np.array, intensities_to_add))/factor_delta

        intensities_to_add = []
        
        
        averaged_intensities.append(averaged_list)
    

    print 'given low energy: %s corresponds to intensity index %i'%(start_energy, energy_index_start)
    print 'given high energy: %s corresponds to intensity index %i'%(end_energy, energy_index_end)

    for i in range(increment):

        all_intensities_averaged.append(data_all_intensities[energy_index_start + i][low:high])
        intensities_cut.append(data_all_intensities[energy_index_start + i][low:high])
    
    return data_angle, intensities_cut, all_intensities_averaged, angle, energy_cut, averaged_intensities, energy_index_start, energy_index_end, delta_e



def create_file(file_data_output, angle_low, angle_high):

    data_angle, intensities_cut, energy_cut  = cut_plot(angle_low, angle_high)
    #data_angle, intensities_cut, energy_cut  = cut_plot('-13.50857', '-10.08000')
 
    with open('cut_' + file_data_output, 'w') as the_file:

        the_file.write('Angle [deg]')
        the_file.write('\t')

        for column in range(len(intensities_cut[0])):

                #the_file.write(energy_cut[column])
                the_file.write(data_angle[column])
                the_file.write('\t')
                
        the_file.write('\n')

        for index, intensity in enumerate(intensities_cut):

            #print intensity

            the_file.write('Intensity_%i'%(index+1))
            the_file.write('\t')
            
            for column in range(len(intensity)):

                the_file.write(intensity[column])
                the_file.write('\t')
                
            the_file.write('\n')

            
    the_file.close()


def create_file_cut(name_file_output, intensities_cut, angle, energy_cut):

     
     with open('cut_' + name_file_output, 'w') as the_file:
         
         #the_file.write('Energy [eV] / Angle [deg]')
         #the_file.write('\n\t')

         for column in range(len(angle)): # write angles in horizontal

                the_file.write(angle[column])
                the_file.write('\t')
                
         the_file.write('\n')

         for index,energy in enumerate(energy_cut): # write energies in vertical

            the_file.write(energy_cut[index])
            the_file.write('\t')

            #for each in intensities_cut:
                #print each
            try: # bug to solve
                for intensity in intensities_cut[index]:
                    #print intensity
                
                    the_file.write(intensity)
                    the_file.write('\t')
            

                the_file.write('\n')
            except:
                pass
                
            #the_file.write('\n')   

     the_file.close()

     myarray = np.asarray(all_intensities_averaged)
     myarray_transposed = np.transpose(myarray)


     with open('cut_transposed_' + name_file_output, 'w') as the_file:

         #the_file.write('Angle [deg] / Energy [eV]')
         #the_file.write('\t')

         for column in range(len(energy_cut)):

                #the_file.write(energy_cut[column])
                the_file.write(energy_cut[column])
                the_file.write('\t')

         the_file.write('\n')

         for index, energy in enumerate(angle):

            the_file.write(angle[index])
            the_file.write('\t')

            for intensity in myarray_transposed[index]:
                    #print intensity
                    the_file.write(intensity)
                    the_file.write('\t')

            the_file.write('\n')

     the_file.close()

     #plt.show()
     with open('all_intensities_' + name_file_output, 'w') as the_file:

         #the_file.write('Energy [eV] / Angle [deg]')
         #the_file.write('\n\t')

         for column in range(len(data_angle)): # write angles in horizontal

                the_file.write(data_angle[column])
                the_file.write('\t')

         the_file.write('\n')

         for index,energy in enumerate(data_energy): # write energies in vertical

            the_file.write(data_energy[index])
            the_file.write('\t')

            #for each in intensities_cut:
                #print each
            try: # bug to solve
                for intensity in data_all_intensities[index]:
                    #print intensity

                    the_file.write(intensity)
                    the_file.write('\t')


                the_file.write('\n')
            except:
                pass

            #the_file.write('\n')

     the_file.close()

     myarray = np.asarray(data_all_intensities)
     myarray_transposed = np.transpose(myarray)


     with open('transposed_all_intensities_' + name_file_output, 'w') as the_file:

         #the_file.write('Angle [deg] / Energy [eV]')
         #the_file.write('\t')

         for column in range(len(data_energy)):

                #the_file.write(energy_cut[column])
                the_file.write(data_energy[column])
                the_file.write('\t')

         the_file.write('\n')

         for index, energy in enumerate(data_angle):

            the_file.write(data_angle[index])
            the_file.write('\t')

            for intensity in myarray_transposed[index]:
                    #print intensity
                    the_file.write(intensity)
                    the_file.write('\t')

            the_file.write('\n')

     the_file.close()







def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        
       import numpy as np
       from math import factorial
       
       try:
           window_size = np.abs(np.int(window_size))
           order = np.abs(np.int(order))
       except ValueError, msg:
           raise ValueError("window_size and order have to be of type int")
       if window_size % 2 != 1 or window_size < 1:
           raise TypeError("window_size size must be a positive odd number")
       if window_size < order + 2:
           raise TypeError("window_size is too small for the polynomials order")
       order_range = range(order+1)
       half_window = (window_size -1) // 2
       # precompute coefficients
       b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
       m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
       # pad the signal at the extremes with
       # values taken from the signal itself
       firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
       lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
       y = np.concatenate((firstvals, y, lastvals))

       return np.convolve( m[::-1], y, mode='valid')

def find_closest(data_angle, data_energy, point_low, point_high, start_energy, end_energy):

    if point_low not in data_angle:

        point_low = min(data_angle, key=lambda x:abs(float(x)-float(point_low)))

    if point_high not in data_angle:

        point_high = min(data_angle, key=lambda x:abs(float(x)-float(point_high)))

    if start_energy not in data_energy:

        start_energy = min(data_energy, key=lambda x:abs(float(x)-float(start_energy)))

    if end_energy not in data_energy:

        end_energy = min(data_energy, key=lambda x:abs(float(x)-float(end_energy)))

    return point_low, point_high, start_energy, end_energy
    

def plot_all_intensities_and_filtering(intensities_cut,window_size, order, averaged_list, angle, delta_e, times_delta_e, name_of_file):

    #gs = gridspec.GridSpec(2,2) to be seen

    plt.subplot(222)

    for intensity in intensities_cut:

            intensity = np.array(intensity)
            intensity = intensity.astype(float)
            intensity_filtered = savitzky_golay(intensity,window_size, order )
            plt.plot(angle, intensity_filtered)

    plt.xlabel('Angle (deg)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Filtered Savitzky Golay\nStart angle: %s End angle: %s\nInitial Energy: %s eV Final Energy: %s eV'%(point_low, point_high,start_energy.split('.')[0], end_energy.split('.')[0]))

    plt.subplot(221)

    for i, element in enumerate(intensities_cut):

            plt.plot(angle, intensities_cut[i])


    plt.xlabel('Angle (deg)')
    plt.ylabel('Intensity (a.u.)')

    plt.title('Start angle: %s End angle: %s\nInitial Energy: %s eV Final Energy: %s eV'%(point_low, point_high,start_energy.split('.')[0], end_energy.split('.')[0]))


    plt.subplot(223)

    #ax = fig.add_subplot(111)

    for i,intensity in enumerate(data_all_intensities):
                plt.plot(data_angle, data_all_intensities[i])

    plt.xlabel('Angle (deg)')
    plt.ylabel('Intesity (a.u.)')


    plt.axvline(x=angle[0], ls='dashed')
    plt.axvline(x=angle[-1], ls='dashed')

    plt.axhline(y=data_all_intensities[energy_index_start][1], ls='dashed')
    plt.axhline(y=data_all_intensities[energy_index_end][1], ls='dashed')
    #plt.annotate('%s'%data_all_intensities[energy_index_start][1], xy=(1,1), xytext=(2,2))


    plt.xlabel('Angle (deg)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('All intensities. File %s'%name_of_file)

    plt.subplot(224)

    for i,averaged in enumerate(averaged_list):
        plt.plot(angle, averaged_list[i], '-o')

    plt.title('delta_e: %s integrate_delta: times_delta_e:%s'%(delta_e, times_delta_e))
    plt.xlabel('Angle(deg)')
    plt.ylabel('Intensity (a.u)')



def plot_image_all_intensities_and_cut(data_all_intensities,intensities_cut ):

    plt.figure(2)

    plt.subplot(211)

    im_intensities = np.array(data_all_intensities)
    im_intensities = im_intensities.astype(float)

    im_intensities_reverted = im_intensities[::-1]

    plt.imshow(im_intensities_reverted, extent = [a[0], a[-1], b[0], b[-1]])

    plt.gca().invert_yaxis()

    plt.title('Image Intensity Complete (Energy vs Angle)')
    plt.xlabel('Angle(deg)')
    plt.ylabel('Energy(eV)')

    plt.subplot(212)

    im_intensities_cut = np.array(intensities_cut)
    im_intensities_cut = im_intensities_cut.astype(float)

    plt.colorbar(orientation = 'horizontal')

    im_intensities_cut_reverted = im_intensities_cut[::-1]

    plt.imshow(im_intensities_cut_reverted, extent = [d[0], d[-1], e[0], e[-1]])
    plt.gca().invert_yaxis()
    plt.xlabel('Angle(deg)')
    plt.ylabel('Energy(eV)')
    plt.title('Start energy:%s End energy: %s'%(start_energy, end_energy))

    plt.figure(3)

    plt.subplot(211)

    im_intensities = np.array(data_all_intensities)
    im_intensities = im_intensities.astype(float)

    im_intensities_reverted = im_intensities[::-1]

    #plt.imshow(im_intensities_reverted, extent = [a[0], a[-1], b[0], b[-1]], cmap = 'Reds')
    plt.imshow(im_intensities_reverted, extent = [a[0], a[-1], b[0], b[-1]])

    #plt.gca().invert_yaxis()

    plt.title('Image Intensity Complete (Energy vs Angle)')
    plt.xlabel('Angle(deg)')
    plt.ylabel('Energy(eV)')

    plt.subplot(212)

    im_intensities_cut = np.array(intensities_cut)
    im_intensities_cut = im_intensities_cut.astype(float)

    plt.colorbar(orientation = 'horizontal')

    im_intensities_cut_reverted = im_intensities_cut[::-1]

    #plt.imshow(im_intensities_cut_reverted, extent = [d[0], d[-1], e[0], e[-1]], cmap = 'Reds')
    plt.imshow(im_intensities_cut_reverted, extent = [d[0], d[-1], e[0], e[-1]])

    #plt.gca().invert_yaxis()
    plt.xlabel('Angle(deg)')
    plt.ylabel('Energy(eV)')
    plt.title('Start energy:%s End energy: %s'%(start_energy, end_energy))


    plt.figure(4)

    plt.subplot(211)

    im_intensities = np.array(data_all_intensities)
    im_intensities = im_intensities.astype(float)

    im_intensities_reverted = im_intensities[::-1]

    plt.imshow(im_intensities_reverted, extent = [a[0], a[-1], b[0], b[-1]], cmap = 'hot')

    plt.gca().invert_yaxis()

    plt.title('Image Intensity Complete (Energy vs Angle)')
    plt.xlabel('Angle(deg)')
    plt.ylabel('Energy(eV)')

    plt.subplot(212)

    im_intensities_cut = np.array(intensities_cut)
    im_intensities_cut = im_intensities_cut.astype(float)

    plt.colorbar(orientation = 'horizontal')

    im_intensities_cut_reverted = im_intensities_cut[::-1]

    plt.imshow(im_intensities_cut_reverted, extent = [d[0], d[-1], e[0], e[-1]], cmap = 'hot')
    plt.gca().invert_yaxis()
    plt.xlabel('Angle(deg)')
    plt.ylabel('Energy(eV)')
    plt.title('Start energy:%s End energy: %s'%(start_energy, end_energy))


def plot_3D(a,b,c, d,e,int_cut):

        plt.figure(9)

        a1 = plt.gca(projection = '3d')
        Z = np.array(c).reshape(a.size, b.size)
        X,Y = np.meshgrid(a,b)

        a1.plot_surface(X, Y, np.transpose(Z), cmap = cm.coolwarm)

        a1.set_xlabel('angle [deg]')
        a1.set_ylabel('energy [eV]')
        a1.set_zlabel('Intensity')



        plt.figure(10)

        a2 = plt.gca(projection = '3d')
        Z = np.array(int_cut).reshape(d.size, e.size)
        X,Y = np.meshgrid(d,e)
        a2.plot_surface(X, Y, np.transpose(Z), cmap = cm.coolwarm)

        a2.set_xlabel('angle [deg]')
        a2.set_ylabel('energy [eV]')
        a2.set_zlabel('Intensity')

        plt.figure(11)

        scaled = []
        scale = 500

        for i,intensity in enumerate(int_cut):

            scaled.append(intensity + i*scale)


        a3 = plt.gca(projection = '3d')
        Z = np.array(scaled).reshape(d.size, e.size)
        X,Y = np.meshgrid(d,e)
        a3.plot_surface(X, Y, np.transpose(Z), cmap = cm.coolwarm)

        a3.set_xlabel('angle [deg]')
        a3.set_ylabel('energy [eV]')
        a3.set_zlabel('Intensity')




def plot_3D_scaled(d,e,int_cut, scale):


        scaled_intensities = []

        for i,intensity in enumerate(int_cut):

            scaled_intensities.append(intensity + i*scale)

        plt.plot(scaled_intensities, e)

        plt.show()



if __name__ == '__main__':
    

    continue_y = 'y'

    #parameters

    name_of_file = '071016_0108.txt'
    #name_of_file = '071016_0681.txt'
    # angles
    
    #point_low = '-6.60571'
    #point_high = '-4.73143'

    #point_low = '-5.00571'
    point_low = '-15'
    point_high = '15'
    # energies
    
    start_energy = '152.00000'
    end_energy = '155.00000'

    times_delta_e = 5
    
    #data_angle, data_all_intensities, data_energy = plot_with_angle(name_of_file)
    
    cut_file_output = 'output_cut.dat'

    # there are the parameters for the smoothing savitzky_golay

    window_size = 11
    order = 4


    while continue_y == 'y':


        graph_mode = ''

        if len(sys.argv) == 10:
            #print 'arguments detected'
            name_of_file, point_low, point_high, start_energy, end_energy, times_delta_e,cut_file_output,  window_size, order = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]
            times_delta_e = float(times_delta_e)

        if len(sys.argv) == 11:
            #print 'arguments detected'
            name_of_file, point_low, point_high, start_energy, end_energy, times_delta_e, cut_file_output,  window_size, order, graph_mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10]
            times_delta_e = float(times_delta_e)


        print name_of_file, point_low, point_high, start_energy, end_energy, times_delta_e,cut_file_output,  window_size, order

        option = '2'

        data_angle, data_all_intensities, data_energy = plot_with_angle(name_of_file)
        point_low, point_high, start_energy, end_energy = find_closest(data_angle, data_energy, point_low, point_high, start_energy, end_energy)
        data_angle, intensities_cut, all_intensities_averaged, angle, energy_cut, averaged_list, energy_index_start, energy_index_end, delta_e = cut_plot(name_of_file, start_energy, end_energy, point_low, point_high, times_delta_e)

        a = np.array(data_angle)
        a = a.astype(float)

        b = np.array(data_energy)
        b = b.astype(float)

        c = np.array(data_all_intensities)
        c = c.astype(float)

        d = np.array(angle)
        d = d.astype(float)

        e = np.array(energy_cut)
        e = e.astype(float)

        int_cut = np.array(intensities_cut)
        int_cut = int_cut.astype(float)




        if graph_mode == '':

            plt.figure(1)

            plot_all_intensities_and_filtering(intensities_cut,window_size, order, averaged_list, angle, delta_e, times_delta_e, name_of_file)
            
            #plt.figure(2)

            plot_image_all_intensities_and_cut(data_all_intensities,intensities_cut )




            plot_3D(a,b,c,d,e,int_cut)



            plt.show()


        if graph_mode == 'all_intensities':

            plot_all_intensities_and_filtering(intensities_cut,window_size, order, averaged_list, angle, delta_e, times_delta_e, name_of_file)
            plt.show()

        if graph_mode == 'image_intensities':

            plot_image_all_intensities_and_cut(data_all_intensities,intensities_cut )
            plt.show()

        if graph_mode == '3_D':

            plot_3D(a,b,c,d,e,int_cut)
            plt.show()

        if graph_mode == 'save_data':

            create_file_cut(cut_file_output, intensities_cut, angle, energy_cut)

        continue_y = 'n'

