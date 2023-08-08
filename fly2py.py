#toolkit to extract data from Flytracker output for JAABA
#Currently can only extract data from the JAABA trx file and any of the perframe directory files
#Can also plot tracks from the x and y data in the trx file

#importing modules
import re
import scipy.io as spio
import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
#networkx can cause some problems, to avoid these networkx is optional for this program to run
try:
    import networkx as nx
    if not hasattr(nx, 'draw'):
        raise AttributeError("The draw() function is not available in your version of the NetworkX library.\nThe network() method is therefore not available.\nPlease use pip to install: `pip install netowrkx`.")
except ImportError:
    print("Warning: NetworkX module not found. The network() method is therefore not available.\nPlease use pip to install: `pip install netowrkx`.")
except AttributeError as e:
    print("Warning:", str(e))









#helper functions
def _dict2list_of_dicts(input_dict):
    """
    Private function used in the class `struct2df` to transpose a dictionary of lists to a list of dictionaries.
    This is for converting the data in mat 7.3 files which are opened with the module mat73. This modules returns a dictioary where each key represents a column.
    However, this module expects a sepearte dictioary for each row, hence the need for transposition. The module converts dictioaries as follows:
    e.g. {'a': [1, 2, 3], 'b': [4, 5, 6]} --> [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]
    """

    #list of input dictionary keys
    keys = list(input_dict.keys())
    #finding the length of the list values of the input dictionary (note: all values are lists of the same length, this line finds the length of the first value)
    length = len(input_dict[keys[0]])

    output_list = []

    #looping through number of items in the list values to make a dictionary for each item (row)
    for i in range(length):
        #generating temp dictionary by iterating through keys and grabbing items for that row
        temp_dict = {key: input_dict[key][i] for key in keys}
        #appending row dictionary to list
        output_list.append(temp_dict)

    return output_list


def _listofdicts_clean_scalar_arrays(listofdicts):
    """
    Private function used in the class `struct2df` to clean the the converted list of dictioaries from `_dict2list_of_dicts`.
    The mat73 module places numeric values into numpy arrays of shape ().
    This function iterates through values in the list of dictioaries and converts the () arrays into numeric values.
    If the value is a whole number this function also converts it to an integer.
    e.g. [{'a':np.array(1.0), 'b':np.array([1, 2, 3])}] --> [{'a':1, 'b':np.array([1, 2, 3])}]
    """
    
    def _scalar_array2num(value):
        """
        Helper function that converts an array of shapr () into a numeric and converts whole number floats to integers.
        Items other than np arrays of shapr () are left alone.
        """
        if isinstance(value, np.ndarray) and value.shape == ():
            scalar_value = value.item()
            if isinstance(scalar_value, float) and scalar_value.is_integer():
                return int(scalar_value)
            return scalar_value
        return value

    output_list = []

    #looping through dictionaries in list
    for dictionary in listofdicts:
        output_dict = {}

        #looping through key,value pairs in dictionary
        for key, value in dictionary.items():
            #direct assignment to output dictioary with converted values (if conversion was needed)
            output_dict[key] = _scalar_array2num(value)
        #appending converted dictionary to list
        output_list.append(output_dict)

    return output_list









#class for extracting matlab structure type data
class struct2df():

    def __init__(self, matfile, separate_chambers=None):
        """
        This class takes in a .mat structure file from the Flytracker for JAABA output and extracts the data.
        First the structure is converted to a multidimensional dictionary using the scipy.io module.
        For mat7.3 files the mat73 module is used to convert the data. Note that this only works on trx files currently.
        The dictionary is parsed to ether extract specific data or extract all the data depending on the type of file.
        With trx files, queried data can be extracted or a dataframe for each fly can be exported.
        With perframe files, the parameter of the file selected is extracted into a dataframe.
        The optional parameter `separate_chambers` is for flytracker data with multiple arenas. Default is None.
        To differentiate between the arenas pass a dictionary of chamber number keys and list of ids as values.
        The key must be a string and the value must be a list of integers representing the fly id.
        e.g. {'1':[1,2,3,4,5,6,7], 'B':[8,9,10,11,12,13,14]}
        """

        #structure to dictionary
        try:
            self.mat_dict = spio.loadmat(matfile, simplify_cells=True)

        except NotImplementedError:
            print("\nWARNING: The input file is mat version 7.3 which must be converted to the scipy.loadmat output format.\nCurrently, only trx files can be converted. Other mat7.3 data files are not currently supported by this module.\n")
            
            #opening mat file with mat73
            self.mat_dict = mat73.loadmat(matfile)

            #transposing the dictionary of lists to list of dictionaries
            temp_ls_of_dicts = _dict2list_of_dicts(self.mat_dict['trx'])

            #cleaning scalar arrays by direct assignment
            self.mat_dict['trx'] = _listofdicts_clean_scalar_arrays(temp_ls_of_dicts)



        #struct2df objects
        self.trx_ls = []
        self.chambers = separate_chambers
        self.param_df = pd.DataFrame()
        self.scores = pd.DataFrame()
        self.processed_scores = pd.DataFrame()
        self.dtype = ''
        self.param_name = ''
        self.behavior_name = ''


        #finding the type of file that was input and loading the relevant objects
        if 'trx' in self.mat_dict.keys():
            self.dtype = 'trx'

            #df for each row of struct of trx file
            for idx in range(len(self.mat_dict['trx'])):

                seriesls = []
                for k, v in self.mat_dict['trx'][idx].items():
                    seriesls.append(pd.Series(v, name=k))

                self.trx_ls.append(pd.concat(seriesls, axis=1))




        elif 'allScores' in self.mat_dict.keys():
            self.dtype = 'scores'

            self.behavior_name = (matfile.split('/')[-1]).replace('.mat', '').replace("scores_", "")

            #making dictionaries for the perframe scores
            scores_dict = {}
            postprocessed_dict = {}

            for idx in range(len(self.mat_dict['allScores']['scores'])):
                scores_dict.update({idx+1 : self.mat_dict['allScores']['scores'][idx]})
                postprocessed_dict.update({idx+1 : self.mat_dict['allScores']['postprocessed'][idx]})

            #making dataframes for the perframe scores
            self.scores = pd.DataFrame(scores_dict)
            self.processed_scores = pd.DataFrame(postprocessed_dict)




        else:
            self.dtype = 'perframe'
            self.param_name = (matfile.split('/')[-1]).replace('.mat', '')

            #making dictionary for the perframe parameter
            new_perframe = {}
            for idx in range(len(self.mat_dict['data'])):
                new_perframe.update({idx+1 : self.mat_dict['data'][idx]})

            #making dataframe for the perframe parameter
            self.param_df = pd.DataFrame(new_perframe)
    



    #methods
    def extract_trx_param(self, param, savefile=True, name=''):
        """
        Method takes in a parameter name as a string (e.g. 'x', 'dt', etc.) or names (e.g. ['x', 'y']) as a list 
        from the trx file and loads the .param_df with a dataframe of that per frame parameter for each fly.
        If the savefile argument is True by default. If it is set to False a csv file will not be saved.
        There is an optional name argument that will add to the begining of the filename and can be used to save file to different path.
        """

        if self.dtype == 'trx':

            #formatting parameters
            paramls = []
            if isinstance(param, str):
                paramls.append(param)
            else:
                paramls = param

            #changing param name
            self.param_name = '_'.join(paramls)

            #making extracted param dataframe
            new_d = {}

            for p in paramls:
                for i in self.trx_ls:
                    l = i[p].to_list()
                    new_d.update({p + '_' + str(int((i['id'].to_list()[0]))) : l})

            self.param_df = pd.DataFrame(new_d)

            if savefile == True:
                self.param_df.to_csv('{nme}_'.format(nme=name) + '_'.join(paramls) + '.csv', index=False)

        else:
            print("Method does not support this data. Make sure data is from the trx file.")



    def save_all_trx(self, name=''):
        """
        Method to save a trx csv for each fly. the optional name argument will add to the begining of the filename and can be used to save file to different path.
        """

        if self.dtype == 'trx':

            for idx, i in enumerate(self.trx_ls):
                i.to_csv('{nme}_'.format(nme=name) + '_{fly}.csv'.format(fly=str(int(idx+1))), index=False)

        else:
            print("Method does not support this data. Make sure data is from the trx file.")



    def save_perframe_or_behavior(self, persecond=False, framerate=30, name=''):
        """
        Method saves .param_df, a dataframe of a feature perframe for each fly, to a csv file.
        There is an optional name argument that will add to the begining of the filename and can be used to save file to different path.
        """

        if self.dtype == 'perframe':
            if persecond == True:
                df_perf = self.param_df.groupby(np.arange(len(self.param_df))//framerate).mean()
                df_perf.to_csv('{nme}_persecond_'.format(nme=name) + self.param_name + ".csv", index=False)
            else:
                self.param_df.to_csv('{nme}_'.format(nme=name) + self.param_name + ".csv", index=False)

        elif self.dtype == 'scores':
            if persecond == True:
                df_scores = self.scores.groupby(np.arange(len(self.scores))//framerate).mean()
                df_proc = self.processed_scores.groupby(np.arange(len(self.processed_scores))//framerate).mean()
                df_scores.to_csv('{nme}_persecond_'.format(nme=name) + self.behavior_name + "_scores.csv", index=False)
                df_proc.to_csv('{nme}_persecond_'.format(nme=name) + self.behavior_name + "_processed_scores.csv", index=False)
            else:
                self.scores.to_csv('{nme}_'.format(nme=name) + self.behavior_name + "_scores.csv", index=False)
                self.processed_scores.to_csv('{nme}_'.format(nme=name) + self.behavior_name + "_processed_scores.csv", index=False)

        else:
            print("Method does not support this data. Make sure data is from the perframe directory.")

    

    def plot_tracks(self, bysex=False, burnin=0, plottitle='', saveplot=True, filename='', showplot=False):
        """
        Method plots tracks of flies using the x,y coordinates (by pixels or mm).
        The plot will be made using the mm data if both pixel and mm data have been extracted but will plot the pixel data if mm x and/or y data have not been extracted
        The optional argument bysex is a boolean argument that indicates whether to color the tracks by the sex of the fly
        burnin is the starting frame for which the plotting starts. it defaults to zero, the first frame.
        If `struct2df` was instanciated with a `separate_chambers` dictionary, multiple plots will be generated.
        """

        if 'x_1' in self.param_df.columns and 'y_1' in self.param_df.columns or ('x_mm_1' in self.param_df.columns and 'y_mm_1' in self.param_df.columns):

            #getting the parameter to plot
            if 'x_mm_1' and 'y_mm_1' in self.param_df.columns:
                measure = 'mm_'
                units = 'mm'
            elif 'x_1' and 'y_1' in self.param_df.columns:
                measure = ''
                units = 'pixels'


            #single chamber data or multi-chamber data in one plot
            if self.chambers == None:

                #setting up figure
                fig = plt.figure(figsize=(9,9))
                ax = fig.add_subplot()

                if bysex == True:

                    #plotting x and y coordinates as a line plot
                    for idx, i in enumerate(self.trx_ls):

                        x = self.param_df['x_{unit}{id}'.format(unit=measure, id=str(int(idx+1)))].to_list()[burnin:]
                        y = self.param_df['y_{unit}{id}'.format(unit=measure, id=str(int(idx+1)))].to_list()[burnin:]

                        if 'm' in i['sex'].to_list():
                            sex = 'm'
                            colr = 'blue'
                        elif 'f' in i['sex'].to_list():
                            sex = 'f'
                            colr = 'red'
                        plt.plot(x, y, label = sex, color=colr, alpha=0.7)

                    #formating and showing the plot
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 0.5), loc="center left")

                else:

                    #plotting x and y coordinates as a line plot
                    for idx, i in enumerate(self.trx_ls):

                        x = self.param_df['x_{unit}{id}'.format(unit=measure, id=str(int(idx+1)))].to_list()[burnin:]
                        y = self.param_df['y_{unit}{id}'.format(unit=measure, id=str(int(idx+1)))].to_list()[burnin:]

                        plt.plot(x, y, alpha=0.7)

                ax.set_aspect('equal', adjustable='box')
                plt.xlabel('X ({})'.format(units))
                plt.ylabel('Y ({})'.format(units))
                plt.title(plottitle)

                if saveplot:
                    plt.savefig('{name}{unit}x_y_tracks.png'.format(name=filename, unit=measure))
                
                if showplot:
                    plt.show()
            
            
            #separate plots for separate chambers
            else:

                """ #setting up figure
                fig = plt.figure(figsize=(9,9))
                ax = fig.add_subplot() """

                if bysex == True:

                    #plotting x and y coordinates as a line plot
                    for cham in list(self.chambers.keys()):
                        #setting up figure
                        fig = plt.figure(figsize=(9,9))
                        ax = fig.add_subplot()
                        for indv in self.chambers[cham]:

                            x = self.param_df['x_{unit}{id}'.format(unit=measure, id=str(int(indv)))].to_list()[burnin:]
                            y = self.param_df['y_{unit}{id}'.format(unit=measure, id=str(int(indv)))].to_list()[burnin:]

                            if 'm' in self.trx_ls[int(indv-1)]['sex'].to_list():
                                sex = 'm'
                                colr = 'blue'
                            elif 'f' in self.trx_ls[int(indv-1)]['sex'].to_list():
                                sex = 'f'
                                colr = 'red'
                            plt.plot(x, y, label = sex, color=colr, alpha=0.7)

                        #formating and showing the plot
                        handles, labels = plt.gca().get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 0.5), loc="center left")
                        ax.set_aspect('equal', adjustable='box')
                        plt.xlabel('X ({})'.format(units))
                        plt.ylabel('Y ({})'.format(units))
                        plt.title(plottitle + " Chamber {ch}".format(ch=cham))

                        if saveplot:
                            plt.savefig('{name}_Chamber_{ch}_{unit}x_y_tracks.png'.format(name=filename, ch=cham, unit=measure))
                        
                        if showplot:
                            plt.show()


                else:

                    #plotting x and y coordinates as a line plot
                    for cham in list(self.chambers.keys()):
                        #setting up figure
                        fig = plt.figure(figsize=(9,9))
                        ax = fig.add_subplot()
                        for indv in self.chambers[cham]:

                            x = self.param_df['x_{unit}{id}'.format(unit=measure, id=str(int(indv)))].to_list()[burnin:]
                            y = self.param_df['y_{unit}{id}'.format(unit=measure, id=str(int(indv)))].to_list()[burnin:]

                            plt.plot(x, y, alpha=0.7)

                        ax.set_aspect('equal', adjustable='box')
                        plt.xlabel('X ({})'.format(units))
                        plt.ylabel('Y ({})'.format(units))
                        plt.title(plottitle + " Chamber {ch}".format(ch=cham))

                        if saveplot:
                            plt.savefig('{name}_Chamber_{ch}_{unit}x_y_tracks.png'.format(name=filename, ch=cham, unit=measure))
                        
                        if showplot:
                            plt.show()



        else:
            print("Method does not support this data. Make sure data is from the trx file and x and y corrdinates have been extracted using the .extract_trx_param() method.")
        


    def plot_density(self, resolution=5, burnin=0, plottype="heatmap", chamber="all", plottitle='', showplot=False, saveplot=True, filename=''):
        """
        Method plots the frequency that locations on the arena were occupied by flies using the x_mm and y_mm parameters.
        A temparary dataframe is made using these parameters and transformed into a 2D histogram. This histogram plots the density either as a heatmap or 3D surface map.
        The `resolution` defaults to 5 which represents 2D partitions of the arena of size 5mmx5mm. This can be changed to desired resolution.
        The `burnin` defaults to 0 and can be set to remove the desired number of frames from the beginning of the data. Units are FRAMES!
        The `plottype` defaults to "heatmap" but can be changed to "3D" for a 3D surface plot. Note that the surface plot is only shown but not automatically saved.
        The `chamber` defaults to all and will plot all arenas if there are multiple. This can be set to a string which is the name of the chamber you wish to plot.
        There is no `plottitle` by default but one can be set. The `filename` defaults to "_density_heatmap.png" but additional text can be added to the beginning using the `filename` parameter.
        The `showplot` and `saveplot` parameters can be set as a bool. Please note that the 3D plot is shown but cannot be saved.
        """

        if 'x_mm_1' in self.param_df.columns and 'y_mm_1' in self.param_df.columns:

            #filter x and y mm data out of self.param_df
            cols = self.param_df.filter(regex='^(x_mm_|y_mm_)').columns.tolist()

            #selecting chamber
            if chamber != "all":
                cols = [col for col in self.param_df.columns if int(col.split('_')[-1]) in self.chambers[chamber]]

            #filtering the df
            df = self.param_df[cols]
            df = df[burnin:]


            #number of flies
            if chamber != "all":
                num_individuals = len(self.chambers[chamber])
            else:
                num_individuals = len(self.trx_ls)

            #list for individual dfs
            individual_dfs = []


            #making dfs for each individual
            if chamber != "all":
                for i in self.chambers[chamber]:
                    #get column name
                    x_col_name = 'x_mm_' + str(i)
                    y_col_name = 'y_mm_' + str(i)
                    
                    #making df
                    individual_df = pd.DataFrame({'x': df[x_col_name], 'y': df[y_col_name]})
                    
                    #appending to df list
                    individual_dfs.append(individual_df)
            else:
                for i in range(num_individuals):
                    #get column name
                    x_col_name = 'x_mm_' + str(i+1)
                    y_col_name = 'y_mm_' + str(i+1)
                    
                    #making df
                    individual_df = pd.DataFrame({'x': df[x_col_name], 'y': df[y_col_name]})
                    
                    #appending to df list
                    individual_dfs.append(individual_df)


            #combining dataframes
            all_df = pd.concat(individual_dfs)
            all_df.dropna(inplace=True)

            #making 2D histogram
            x_bins = int((all_df['x'].max() - all_df['x'].min()) / resolution)
            y_bins = int((all_df['y'].max() - all_df['y'].min()) / resolution)

            num_bins = (x_bins, y_bins)

            hist, xedges, yedges = np.histogram2d(all_df['x'], all_df['y'], bins=num_bins)

            #normalize histogram
            hist_norm = hist/(np.sum(hist))

            #making histogram and plotting
            if plottype == "heatmap":
                
                #Plot the heatmap Note: must be transposed because np.histogram2d does not follow normal cartesian convention
                plt.imshow(hist_norm.T, cmap='plasma', interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
                cbar = plt.colorbar()
                cbar.set_label('Frequency')

                #labels
                plt.xlabel('X (mm)')
                plt.ylabel('Y (mm)')
                plt.title(plottitle)

                if saveplot:
                    plt.savefig('{}_density_heatmap.png'.format(filename))

                if showplot:
                    plt.show()



            elif plottype == "3D":
                #creating 3D figure
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Create the X, Y meshgrid with reversed x-axis limits
                x_, y_ = np.meshgrid(xedges[:-1], yedges[:-1])

                # Plot the 3D surface  Note: must be transposed because np.histogram2d does not follow normal cartesian convention
                ax.plot_surface(x_, y_, hist_norm.T, cmap='plasma')

                # Set the axis labels
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Frequency')
                plt.title(plottitle)

                #Show plot
                plt.show()
                

            else:
                print('Incorrect plottype input. Please use either "heatmap" or "3D".')

        else:
            print("Method does not support this data. Make sure data is from the trx file and x and y corrdinates have been extracted using the .extract_trx_param() method.")
        




    def plot_timeseries(self, fly='all', persecond=True, framerate=30, scorethreshold=None, burnin=0, plottitle='', saveplot=True, filename='', showplot=False):
        """
        Plots a line graph of a perframe feature or behavior score. Can plot lines for all flies or select flies.
        If the type of data is JAABA behavior data, the method outputs a scores and processed scores plots.
        persecond argment defaults to true to plot data averaged per second. Set to False to plot per frame.
        framerate defaults to 30 fps. Check the framerate of your experiement and change accordingly.
        The fly argument defaults to all, but this can be changed to a fly id as an integer, a list of fly ids as integers, or the name of a fly chamber as a string.
        plottitle and filename are optional arguments. filename adds to the beginning of the filename, there is a default name for the file.
        Optional arguments to save the plot and show the plot.
        scorethreshold defaults to None, but change to a float to set a lower limit to the processed behavior score
        burnin is the starting frame at which the plotting should start. If the plotting is set to seconds the method converts the frame to seconds.
        """


        #formatting fly parameter
        flyls = []
        if isinstance(fly, list):
            flyls = fly
        else:
            if isinstance(fly, int):
                flyls.append(fly)
            elif fly != "all":
                flyls = self.chambers[fly]
            else:
                flyls = None
            


        
        #setting burnin
        if persecond == True:
            bi = int(burnin / framerate)
        else:
            bi = burnin

        
        #getting unit
        if persecond == True:
            unit = "Seconds"
        else:
            unit = "Frames"


        if self.dtype == 'perframe':
            plt.figure(figsize=(15,5))

            if fly == 'all':
                #plotting x and y coordinates as a line plot
                for idx, i in enumerate(self.mat_dict['data']):
                    if persecond == True:
                        modi = np.append(i, [0 for j in range((framerate - (len(i) % framerate)))])
                        ls = np.average(modi.reshape(-1, framerate), axis=1)
                    else:
                        ls = i
                    plt.plot(ls, label=idx+1)

                #formating and showing the plot
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

            elif len(flyls) == 1:
                for i in flyls:
                    if persecond == True:
                        modls = np.append(self.mat_dict['data'][int(i)-1], [0 for j in range((framerate - (len(self.mat_dict['data'][int(i)-1]) % framerate)))])
                        ls = np.average(modls.reshape(-1, framerate), axis=1)
                    else:
                        ls = self.mat_dict['data'][int(i)-1]
                    plt.plot(ls)

            else:
                for i in flyls:
                    if persecond == True:
                        modls = np.append(self.mat_dict['data'][int(i)-1], [0 for j in range((framerate - (len(self.mat_dict['data'][int(i)-1]) % framerate)))])
                        ls = np.average(modls.reshape(-1, framerate), axis=1)
                    else:
                        ls = self.mat_dict['data'][int(i)-1]
                    plt.plot(ls, label=i)

                #formating and showing the plot
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

            plt.xlim(left=bi, right=len(ls))
            plt.xlabel(unit)
            plt.ylabel(self.param_name)
            plt.title(plottitle)

            if saveplot:
                plt.savefig('{name}{default}_perframe_plot.png'.format(name=filename, default=self.param_name))
            
            if showplot:
                plt.show()

        



        elif self.dtype == 'scores':
            for thing2plot in ['scores', 'postprocessed']:
                plt.figure(figsize=(15,5))

                if fly == 'all':
                    #plotting x and y coordinates as a line plot
                    for idx, i in enumerate(self.mat_dict['allScores'][thing2plot]):
                        if persecond == True:
                            modi = np.append(i, [0 for j in range((framerate - (len(i) % framerate)))])
                            ls = np.average(modi.reshape(-1, framerate), axis=1)
                        else:
                            ls = i
                        plt.plot(ls, label=idx+1, alpha=0.5)
                        if thing2plot == 'postprocessed':
                            plt.fill_between([j for j in range(len(ls))], ls, alpha=0.5)
                            if scorethreshold != None:
                                plt.ylim(bottom=scorethreshold, top=1)
                            else:
                                plt.ylim(top=1)

                    #formating and showing the plot
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    leg = plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
                    for obj in leg.get_lines():
                        obj.set_linewidth(5)

                elif len(flyls) == 1:
                    for i in flyls:
                        if persecond == True:
                            modls = np.append(self.mat_dict['allScores'][thing2plot][int(i)-1], [0 for j in range((framerate - (len(self.mat_dict['allScores'][thing2plot][int(i)-1]) % framerate)))])
                            ls = np.average(modls.reshape(-1, framerate), axis=1)
                        else:
                            ls = self.mat_dict['allScores'][thing2plot][int(i)-1]
                        plt.plot(ls)

                        if thing2plot == 'postprocessed':
                            plt.fill_between([i for i in range(len(ls))], ls)
                            if scorethreshold != None:
                                plt.ylim(bottom=scorethreshold, top=1)
                            else:
                                plt.ylim(top=1)
                                
                        

                else:
                    for i in flyls:
                        if persecond == True:
                            modls = np.append(self.mat_dict['allScores'][thing2plot][int(i)-1], [0 for j in range((framerate - (len(self.mat_dict['allScores'][thing2plot][int(i)-1]) % framerate)))])
                            ls = np.average(modls.reshape(-1, framerate), axis=1)
                        else:
                            ls = self.mat_dict['allScores'][thing2plot][int(i)-1]
                        plt.plot(ls, label=i, alpha=0.5)
                        if thing2plot == 'postprocessed':
                            plt.fill_between([i for i in range(len(ls))], ls, alpha=0.5)
                            if scorethreshold != None:
                                plt.ylim(bottom=scorethreshold, top=1)
                            else:
                                plt.ylim(top=1)

                    #formating and showing the plot
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    leg = plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
                    for obj in leg.get_lines():
                        obj.set_linewidth(5)

                #scores or postprocessed
                if thing2plot == 'scores':
                    ylabelname = ' score'
                else:
                    ylabelname = ' processed score'

                plt.xlim(left=bi, right=len(ls))
                plt.xlabel(unit)
                plt.ylabel(self.behavior_name + ylabelname)
                plt.title(plottitle)

                if saveplot:
                    plt.savefig('{name}{default}_{flies}_{thing}_plot.png'.format(name=filename, default=self.param_name, flies=str(fly), thing=thing2plot))
                
                if showplot:
                    plt.show()





#class for organizing a fly experiment using struct2df instances
class fly_experiment():

    def __init__(self, structdfls):
        """
        Note that this class cannot be imported without also importing struct2df.
        This class takes in a list of instances of struct2df from the same fly experiment.
        The instantiation of the class needs the trx.mat file and any other .mat file you want to include.
        This class will load data into multiple lists and dictionaries which can be referenced and are referenced by methods.
        The ethogram method needs processed behavior score mat files.
        The network method needs the dcenter parameter and an optional behavior processed score file.
        """

        #objects that can be referenced
        self.trx_ls = []
        self.trxs = {}
        self.chambers = None
        self.perframes = {} #also includes any extracted parameters from trx
        self.jaaba_scores = {}
        self.jaaba_processed = {}
        self.sex = {}

        #loading data into objects
        for i in structdfls:

            if i.dtype == 'trx':
                self.trx_ls = i.trx_ls
                self.chambers = i.chambers

                for idx, j in enumerate(i.trx_ls):
                    self.trxs.update({idx+1: j})
                    self.sex.update({idx+1: j['sex'].to_list()[0]})

                if i.param_df.empty:
                    continue
                else:
                    self.perframes.update({'trx_' + i.param_name: i.param_df})
                

            elif i.dtype == 'perframe':
                self.perframes.update({i.param_name: i.param_df})


            else:
                self.jaaba_scores.update({i.behavior_name: i.scores})
                self.jaaba_processed.update({i.behavior_name: i.processed_scores})

        #another object that can be referenced (list of fly ids)
        self.flies = list(self.trxs.keys())

    

    #methods
    def stack_timeseries(self, params="all", behavior_scores="all", behavior_processed="all", persecond=False, framerate=30, savefile=False, name=''):
        """
        The default behavior of this method is to put every perframe feature including behavior scores into one dataframe that is returned.
        The params, behavior_scores, and behavior_processed arguments can be set to the name of one or a few (str or list) features instead of all features.
        These can also be set to None if no parameters from that category are desired.
        If the savefile argument is False by default. If it is set to True a csv file will be saved.
        There is an optional name argument that will add to the begining of the filename and can be used to save file to different path.
        """

        #getting lists of features to extract
        paramls = []
        scoresls = []
        processedls = []

        if not isinstance(params, list) and params != 'all' and params != None:
            paramls.append(params)
        elif params == 'all':
            paramls = list(self.perframes.keys())
        elif params == None:
            paramls.append("")
        else:
            paramls = params

        
        if not isinstance(behavior_scores, list) and behavior_scores != 'all' and behavior_scores != None:
            scoresls.append(behavior_scores)
        elif behavior_scores == 'all':
            scoresls = list(self.jaaba_scores.keys())
        elif behavior_scores == None:
            scoresls.append("")
        else:
            scoresls = behavior_scores


        if not isinstance(behavior_processed, list) and behavior_processed != 'all' and behavior_processed != None:
            processedls.append(behavior_processed)
        elif behavior_processed == 'all':
            processedls = list(self.jaaba_processed.keys())
        elif behavior_processed == None:
            processedls.append("")
        else:
            processedls = behavior_processed

        
        #stacking data
        stackdf = pd.DataFrame()

        if params != None:
            for i in paramls:
                df = self.perframes[i]

                if 'trx' not in i:
                    df = df.add_prefix(i + '_')

                if stackdf.empty:
                    stackdf = df
                else:
                    stackdf = pd.merge(stackdf, df, left_index=True, right_index=True)

        if behavior_scores != None:
            for i in scoresls:
                df = self.jaaba_scores[i]
                df = df.add_prefix(i + '_score_')

                if stackdf.empty:
                    stackdf = df
                else:
                    stackdf = pd.merge(stackdf, df, left_index=True, right_index=True)

        if behavior_processed != None:
            for i in processedls:
                df = self.jaaba_processed[i]
                df = df.add_prefix(i + '_processed_')

                if stackdf.empty:
                    stackdf = df
                else:
                    stackdf = pd.merge(stackdf, df, left_index=True, right_index=True)

        #per frame or per second
        if persecond == True:
            stackdf = stackdf.groupby(np.arange(len(stackdf))//framerate).mean()

        #saving df
        if savefile == True:
            superlist = paramls + scoresls + processedls
            stackdf.to_csv('{nme}_'.format(nme=name) + '_'.join(superlist) + '.csv', index=False)
        
        return stackdf

        

    def ethogram(self, burnin=0, scorethreshold=None, fly="all", framerate=30, plottitle="", showplot=False, saveplot=True, filename=""):
        """
        Method to plot a pseudo-ethogram of all loaded behaviors for all flies, subset of flies, or single fly.
        The burnin can be set to the SECOND to start the plot at. Note that this is different from the struct2df method which takes the frame to start at.
        The average behavior score threshold can also be set, but defaults to zero.
        The flies defaults to a plot of all flies but can be set to a subset of flies which is input as a list of fly ids as integers or a single fly id as an integer.
        'm' or 'f' can also be passed to select just male or female flies.
        You may also pass in the name of a chamber as a string if you wish to plot all flies in one chmaber and if the trx `struct2df` instance contains a separate_chambers dictionary.
        """

        #selecting flies
        flyls = []

        if fly == 'all':
            for idx in range(len(self.trx_ls)):
                flyls.append(idx+1)
            opacity = 0.5

        elif isinstance(fly, list):
            flyls = fly
            opacity = 0.5

        elif fly == 'm':
            for i in self.sex.keys():
                if self.sex[i] == 'm':
                    flyls.append(i)
            flyls.sort()
            opacity = 0.5

        elif fly == 'f':
            for i in self.sex.keys():
                if self.sex[i] == 'f':
                    flyls.append(i)
            flyls.sort()
            opacity = 0.5

        elif self.chambers != None and fly in list(self.chambers.keys()):
            flyls = list(self.chambers[fly])
            opacity = 0.5

        else:
            flyls.append(fly)
            opacity = 1


        #getting behaviors
        behaviors = list(self.jaaba_processed.keys())

        #ethogram can only plot with more than one behavior, thus this if, else condition
        if len(behaviors) > 1:


            #plotting
            fig, ax = plt.subplots(len(behaviors), 1, sharex='col', figsize=(20,7))

            for i in range(len(behaviors)):

                #averaging dataframe per second
                framesdf = self.jaaba_processed[behaviors[i]]
                persec = framesdf.groupby(np.arange(len(framesdf))//framerate).mean()

                for id in flyls:

                    ax[i].plot(persec[id].to_list(), label=id, alpha=opacity)
                    ax[i].fill_between([i for i in range(len(persec[id].to_list()))], persec[id].to_list(), alpha=opacity)
                    ax[i].set_ylabel(behaviors[i])

                    if scorethreshold != None:
                        ax[i].set_ylim(bottom=scorethreshold, top=1)
                    else:
                        ax[i].set_ylim(top=1)

            plt.xlim(left=burnin, right=len(persec[id].to_list()))
            plt.xlabel("seconds")
            plt.suptitle(plottitle)

            #formating and showing the plot
            if len(flyls) != 1:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                leg = plt.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=len(flyls))
                for obj in leg.get_lines():
                    obj.set_linewidth(5)

            if showplot:
                plt.show()

            if saveplot:
                plt.savefig('{name}_ethogram_{flies}.png'.format(name=filename, flies=str(fly)))


        else:
            print("\nWARNING: Only one behavior is present in this instance of `fly_experiment`. `.ethogram` cannot plot with only one behavior.\nPlease use the `.plot_timeseries` method on the `struct2df` instance instead.\n")





    def network(self, dist_threshold=float('inf'), behavior=None, behavior_threshold=0.5, burnin=0, framerate=30, chamber="all", plottitle="", showplot=False, saveplot=True, filename=""):
            """
            can now pass the chamber name as a string to `chamber`
            """

            #function for sorted() function key
            def sort_key(lsitem):

                number = re.findall(r'\d+', lsitem)[0]

                return int(number)


            #warning message
            if dist_threshold == float('inf') and behavior == None:
                print("WARNING: it is not recommended to create a proximity network without a distance threshold set.")

            #getting pairs
            colnames = list(self.perframes['dcenter'].columns)

            #condition for removing unwanted chambers
            if chamber == "all":
                colnames = ['dcenter_' + str(i) for i in colnames]
            else:
                colnames = ['dcenter_' + str(i) for i in self.chambers[chamber]]



            pairs = list(itertools.combinations(colnames, 2))
            connections = {}
            for i in pairs:
                connections.update({i:0})

            
            #stacking timeseries of dcenter and behavior
            if behavior == None:
                stack = self.perframes['dcenter']
                stack = stack.add_prefix('dcenter_')

            else:
                stack = self.stack_timeseries(params='dcenter', behavior_scores=[], behavior_processed=behavior)

            #making stack per second
            stack = stack.groupby(np.arange(len(stack))//framerate).mean()
            stack = stack[burnin:]


            #interactions list
            interactions_raw = []


            #single chamber or chamber selection
            if chamber == "all":
                #looping through flies
                for i in self.flies:

                    #empty df
                    df = pd.DataFrame()

                    #filtering for behavior unless no behavior parameter is given
                    if behavior != None:
                        column_name = "{b}_processed_{f}".format(b=behavior,f=str(i))
                        df = pd.concat([df, stack.loc[stack[column_name] >= behavior_threshold]])
                    else:
                        df = pd.concat([df, stack])

                    #new interaction column
                    df['interactions'] = df.apply(lambda row: [col for col in df.columns if col.startswith('dcenter_') and row[col] == row['dcenter_{f}'.format(f=str(i))] and row[col] <= dist_threshold], axis=1)

                    #adding interactions to list
                    interactions_raw += df['interactions'].to_list()

            else:
                #looping through flies
                for i in self.chambers[chamber]:

                    #empty df
                    df = pd.DataFrame()

                    #filtering for behavior unless no behavior parameter is given
                    if behavior != None:
                        column_name = "{b}_processed_{f}".format(b=behavior,f=str(i))
                        df = pd.concat([df, stack.loc[stack[column_name] >= behavior_threshold]])
                    else:
                        df = pd.concat([df, stack])

                    #generating new interaction column
                    df['interactions'] = df.apply(lambda row: [col for col in df.columns if col.startswith('dcenter_') and row[col] == row['dcenter_{f}'.format(f=str(i))] and row[col] <= dist_threshold], axis=1)

                    #adding interactions to list
                    interactions_raw += df['interactions'].to_list()






            #cleaning interactions list
            interactions = [tuple(sorted(item, key=sort_key)) for item in interactions_raw if len(item) == 2]

            #getting weights based on frequency of interactions (pairwise)
            for i in interactions:
                connections[i] += 1



            #making network df
            nw_df = pd.DataFrame.from_dict(connections, orient='index', columns=['weight'])
            nw_df = nw_df.reset_index().rename(columns={'index': 'tuples'})

            

            #nw_df = nw_df.drop(nw_df[nw_df['weight'] == 0].index)

            nw_df[['node1', 'node2']] = nw_df['tuples'].apply(lambda x: pd.Series([x[0], x[1]]))
            nw_df = nw_df.drop('tuples', axis=1)
            nw_df = nw_df.replace(to_replace='dcenter_', value='', regex=True)
            nw_df = nw_df.loc[:,['node1','node2','weight']]
            


            #formatting plot
            plt.figure(figsize=(8,8))
            ax = plt.gca()
            ax.set_title(plottitle)


            ###making network
            # create an empty undirected graph
            G = nx.Graph()

            #node colors
            colors = {}
            for k in self.sex.keys():
                if self.sex[k] == 'm':
                    colors.update({k:'blue'})
                else:
                    colors.update({k:'red'})

            # add nodes with their respective colors, ordering nodes by fly id
            nodes = list(set(nw_df['node1']).union(set(nw_df['node2'])))
            nodes = [int(i) for i in nodes]
            nodes.sort()
            nodes = [str(i) for i in nodes]

            for node in nodes:
                col = colors.get(node, 'gray') # use the color from the dictionary, or default to gray
                G.add_node(node, color=col)
            

            # add edges with their respective weights
            for index, row in nw_df.iterrows():
                G.add_edge(row['node1'], row['node2'], weight=row['weight'])

            # draw the graph
            pos = nx.circular_layout(G)
            weights = [G[u][v]['weight'] for u, v in G.edges()]

            try:
                nx.draw(G, pos, width=[(w/nw_df['weight'].max())*15 for w in weights], edge_color='gray', node_color=[colors[int(node)] for node in G.nodes()])
                nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')

                if saveplot:
                    plt.savefig('{name}_{d}mm_behavior_{b}_network.png'.format(name=filename, d=str(dist_threshold), b=behavior))

                if showplot:
                    plt.show()

            except ZeroDivisionError:
                plt.close()
                print("Thresholds set are too strict. No interactions found with these thresholds. Could not generate network.")







