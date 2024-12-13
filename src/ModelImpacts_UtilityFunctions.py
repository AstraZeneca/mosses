# Author - Vigneshwari Subramanian
#Code relevant for time weighted score plots contributed by Nafiz Abeer

# A script will all the utility functions required for running the model impacts notebook; Please make sure to have this file in the same folder as your notebook

#Load the relevant packages and set the directory to the path where you have your files
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import colorama
import seaborn as sns
from math import sqrt
from colorama import Fore
from datetime import date,datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr,spearmanr
from scipy.stats import gmean

from scipy.signal import savgol_filter
from IPython.core.display import display_markdown, display_html
import warnings
warnings.filterwarnings('ignore')

plt.ioff()

def is_in_notebook():
    return 'ipykernel' in sys.modules

def print_note(message):
    if is_in_notebook():
        display_markdown(message, raw=True)
    else:
        print(message)

def print_metrics_table(pearson_r2, r2, rmse):
    if is_in_notebook():
        display_markdown(f"""
| Metric | Value |
| ------ | ----- |
| Experimental vs Predicted correlation (Coefficient of determination, R2) | {r2} |
| Root Mean Squared Error (RMSE) | {rmse} |

""", raw=True)
    else:
        print(f"Experimental vs Predicted correlation (Coefficient of determination, R2): {r2}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

def print_cpds_info_table(TotalCpds, TrainingSet, TestSet,Compounds_BelowThresh, Compounds_AboveThresh, ratio_GoodCpds):
    if is_in_notebook():
        display_markdown(f"""
|  | No. of Compounds |
| ------ | ----- |
| Compounds with measured values| {TotalCpds}|
| Training Set| {TrainingSet} |
| Prospective Validation Set| {TestSet} |
| Below Selected Experimental Threshold | {Compounds_BelowThresh} |
| Above Selected Experimental Threshold | {Compounds_AboveThresh} |
| Ratio of good compounds made so far | {ratio_GoodCpds} |        
""", raw=True)
    else:
        print(f"Compounds below the desired project threshold: {Compounds_BelowThresh}")
        print(f"Compounds above the desired project threshold: {Compounds_AboveThresh}")
        print(f"Ratio of good compounds made so far: {ratio_GoodCpds}")
        print("\n")
           
#Generate a table for biased metrics
def print_PPV_FOR_table(PreSelectedThreshold, PPV, FOR,RecommendedThreshold,RecPPV,RecFOR):
    if is_in_notebook():
        display_markdown(f"""
|  | Predicted Threshold | PPV % | FOR % |
| ------ | ----- | ----- | ----- |
| Selected Experiemental Threshold | {PreSelectedThreshold} | {PPV} | {FOR} |
| Recommended Threshold | {RecommendedThreshold} | {RecPPV} | {RecFOR} |     
""", raw=True)
    else:
        print(f"Threshold Type: {'Selected Experiemental Threshold'}")
        print(f"Threshold: {PreSelectedThreshold}")
        print(f"PPV at the selected threshold: {PPV}")
        print(f"FOR at the selected threshold: {FOR}")
        print("\n")

        print(f"Threshold Type: {'Recommended Threshold'}")
        print(f"Threshold: {RecommendedThreshold}")
        print(f"PPV at the selected threshold: {RecPPV}")
        print(f"FOR at the selected threshold: {RecFOR}")
        print("\n")


#Generate a table for unbiased metrics
def print_Unbiased_PPV_FOR_table(RecommendedThreshold,RecPPV,RecFOR):
    if is_in_notebook():
        display_markdown(f"""
| Experimental = Predicted threshold | PPV % | FOR % |
| ----- | ----- | ----- |
| {RecommendedThreshold} | {RecPPV} | {RecFOR} |     
""", raw=True)
    else:
        print(f"Threshold Type: {'Recommended Threshold'}")
        print(f"Threshold: {RecommendedThreshold}")
        print(f"PPV at the selected threshold: {RecPPV}")
        print(f"FOR at the selected threshold: {RecFOR}")
        print("\n")


# helper function to set new x ticks 
def Reset_x_ticks(Threshold,ax):
        xlims = np.arange(min(np.log10(Threshold))-0.5,max(np.log10(Threshold))+0.5,step=0.5)
        
        ax.set_xlim(min(np.log10(Threshold))-0.5,max(np.log10(Threshold))+0.5)
        ax.xaxis.set_ticks(xlims)
        
        #Replace log values with actual values to make the plots much more intuitive
        x_ticks = ax.get_xticks()
        if (x_ticks<0).any():
            new_x_ticks = [round((10 ** x),2) for x in x_ticks]
        else:
            new_x_ticks = [int(10 ** x) for x in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(new_x_ticks,rotation = 45)
        return ax

# helper function to set new y ticks        
def Reset_y_ticks(ax):
    ax.set_ylim(0,100)
    y_ticks = ax.get_yticks()
    new_y_ticks = [int(y) for y in y_ticks]
    ax.set_yticks(new_y_ticks)
    ax.set_yticklabels(new_y_ticks, rotation = 45)   
    return ax

#helper function to compute rmse
def rmse_score(obs,pred,scale):
    if scale=='log':
        rmse = round(math.sqrt(mean_squared_error(np.log10(obs),np.log10(pred))),2)
        
    else:
        rmse = round(math.sqrt(mean_squared_error(obs,pred)),2)
        
    return rmse
    

#Function to select minimum and maximum thresholds relevant for PPV and FOR calculations
def Thresh_Selection(Preds,DesiredProjectThreshold,scale):
    if scale == "log":
        #transform to log, do the selection of thresholds, transform back to non logged values
        min_thresh = np.log10(min(Preds))
        max_thresh = np.log10(max(Preds))
        increment_factor = (max_thresh-min_thresh)/50
        Thresholds_selection = np.append([(10 ** x) for x in np.arange(min_thresh,max_thresh,increment_factor)],[DesiredProjectThreshold])
    else:
        min_thresh = min(Preds)
        max_thresh = max(Preds)
        increment_factor = (max_thresh-min_thresh)/50
        Thresholds_selection = np.append(np.arange(min_thresh,max_thresh,increment_factor),[DesiredProjectThreshold])

    return min_thresh,max_thresh,Thresholds_selection


#Finding the longest arrow based on unbiased PPVs and FORs and 
# provide a recommended threshold fo extracting as many good compound as possible
def LongestArrow(Thresh,PPV,FOR):
    #Predicted pos and neg likelihoods can have Nan due to smoothing; Remove them, prior to max distance calculations
    Relevant_DistCalc_cols = pd.DataFrame({'Thresh': Thresh,'PPV': PPV, 'FOR': FOR}, columns=['Thresh', 'PPV','FOR'])
    Relevant_DistCalc_cols = Relevant_DistCalc_cols.dropna().reset_index(drop=True)

    if (len(Relevant_DistCalc_cols) >0):
        # Calculate the absolute differences between the PPV and FOR curves
        distances = np.array(np.abs(Relevant_DistCalc_cols.PPV - Relevant_DistCalc_cols.FOR))

        # Find the maximum distance and the corresponding index
        max_distance = np.max(distances)
        ind = np.argmax(distances)

        # Find the threshold that corresponds to the maximum distance between the 2 curves
        Thresh_max_distance = Relevant_DistCalc_cols.Thresh[ind]

        return max_distance, Thresh_max_distance, Relevant_DistCalc_cols.PPV[ind],Relevant_DistCalc_cols.FOR[ind]

    else:
        return -1,-1,-1,-1   #Setting an arbitrary value, if PPV and FOR values are NAN for all the predicted thresholds


# Scoring functions relevant for time dependant MPO optimization
# Calculating similarity scores based on Euclidean distance
def similarity_score(x,y,x2,y2,d):
    '''
    x : ground truth for training data
    y : predictions for training data
    x2 : ground truth for prospective/test data
    y2 : predictions for prospective/test data
    
    d : (d>0) smoothing factor/parameter, smaller value of d will penalize the distance more.
    
    '''
    
    ref_pairs = np.vstack((x,y)).T
    test_pairs = np.vstack((x2,y2)).T
    
    #Scaled to accomodate the penalty values that range between 0 and 1
    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref_pairs)
    X_test = scaler.transform(test_pairs)
    
    # Euclidean distance used to compute distances at the moment; Absolute values are expected to change depending on the distance metric, but the trends would remain the same
    pairwise_dist = cdist(X_ref, X_test)
    
    #Find the closest training value and convert that to a similarity score
    Sim_score = np.exp(-pairwise_dist.min(axis = 0)/d).mean()
    return Sim_score


#Calclating Spearmann correlation scores
def correlation_score(x,y,x2,y2, d, return_nbr_idx = False):
    
    '''
    x : ground truth for training data
    y : predictions for training data
    x2 : ground truth for prospective/test data
    y2 : predictions for prospective/test data
    
    d : (d>0) smoothing factor/parameter, smaller value of d will put more penalty on the difference between correlation.
    
    return_nbr_idx : used only for visualization, not needed for the scoring 
    
    '''
    
    ref_pairs = np.vstack((x,y)).T
    test_pairs = np.vstack((x2,y2)).T
    
    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref_pairs)
    X_test = scaler.transform(test_pairs)
    pairwise_dist = cdist(X_ref, X_test)
    
    nbr_idx = np.argmin(pairwise_dist, axis = 0) # Finding the indices of the closest neighbours
    nbr_r = (spearmanr(X_ref[nbr_idx,0],X_ref[nbr_idx,1]).statistic+1)/2 #Spearmann correlation between the actual and the predicted training values; (Statistics +1)/2 done to rescale values from 0 to 1
    test_r = (spearmanr(X_test[:,0],X_test[:,1]).statistic+1)/2 #Spearmann correlation between the actual and the predicted test values
    
        
    #Calculate the final scores based on the differences between training and test Spearmann correlations
    if return_nbr_idx:
        Corr_score =np.exp(-np.abs(nbr_r-test_r)/d), nbr_idx
        return Corr_score 
    else:
        #Corr_score =np.exp(-np.log(2)*np.abs(nbr_r-test_r)/d)
        Corr_score =np.exp(-np.abs(nbr_r-test_r)/d)
        return Corr_score



#A function to calulate time dependant scores that rely on the first predicted values of a compound; Both training and test data would be used for this assessment
def time_weighted_score_plot(df,discount_factor,PlotTitle):
    '''
    df : dataframe with all the observed and predicted values of both the training and test sets
    
    discount factor: smaller values (>0) put more weight on recent score. 
                    setting it to 1 is same as taking average with uniform weights.
    
    
    Outputs:
    
    t : array of unique timepoints (in chronological order) excluding the first one.
    
    score : 2D array, where each column contains the score for len(t) timepoints
    
    '''
    
    #Extarct the months and years from the date
    df['ModelVersionDate'] = pd.DataFrame(df['ModelVersion'].apply(lambda x: x.split("-")[0]))
    df['ModelVersionDate'] = pd.to_datetime(df['ModelVersionDate'])
    df_sorted = df.sort_values(by = 'ModelVersionDate')


    #Make an array of time points and navigate through every timepoint to calculate the MPO scores
    t_arr, _ = np.unique(df_sorted['ModelVersionDate'], return_counts = True)
    t_arr_month_year = []
    scores = []
    t_all=[]
    
    for t in t_arr[1:]: #Excluding the first time point, as it doesn't add much value to the analysis
        train_mask = df['ModelVersionDate']<t
        test_mask = df['ModelVersionDate']==t
        train_df = df[train_mask]
        prospective_df = df[test_mask]
        train_x, train_y = train_df['Observed'].to_numpy(), train_df['Predicted'].to_numpy()
        test_x, test_y = prospective_df['Observed'].to_numpy(), prospective_df['Predicted'].to_numpy()

        #Consider only those train and test dataframes with atleast 10 datapoints to avoid any bias created by 1 or 2 highly similar or dissimilar datapoints
        if ((len(train_df) >= 10) & (len(prospective_df) >= 10)):
            sim_score = similarity_score(train_x, train_y,test_x, test_y ,0.8)
            corr_score = correlation_score(train_x, train_y,test_x, test_y,0.2, return_nbr_idx=False)
            scores.append([sim_score, corr_score])
            t_all.append(t)
    
    if (len(scores)>0): #Compute weighted scores only if the scores matrix is not null
        score = np.vstack(scores)

        #Generating a triangular matrix with discount factors multiplied by arange of time indices; Oldest timepoint with a few training data points weighted much less than the recent ones
        x = discount_factor**np.arange(len(t_all)-1, -1,-1) 
        W = np.tril(x)

        w_score = W@score/W.sum(-1).reshape(-1,1)

        for date in t_all:
            date = date.astype("datetime64[D]")
            t_arr_month_year.append(date.astype(datetime).strftime('%b %Y'))
        
        if (len(t_arr_month_year) > 1):
            fig,ax = plt.subplots(figsize=(5,5))
            fig.canvas.header_visible = False

            plt.plot(t_arr_month_year,score)
            plt.plot(t_arr_month_year,w_score)

            ax.set_xlabel('Model version',fontweight='bold')
            ax.set_ylabel('Scores',fontweight='bold')       
            ax.set_xticklabels(t_arr_month_year,rotation = 90)
            plt.rc('xtick',labelsize=8)
            plt.rc('ytick',labelsize=8)

            plt.legend(['Similarity of data', 'Similarity of correlation']+ ['Similarity of data (Time-weighted)', 'Similarity of correlation (Time-weighted)'],fontsize=7)
            plt.title(PlotTitle)
            plt.ylim([0,1.1])
            plt.show()
        else:
            print(Fore.RED+"No sufficient datapoints to generate plots!"+Fore.RESET)

    else:
            print(Fore.RED+"No sufficient datapoints to generate plots!"+Fore.RESET)

#A function to compute Positive Predictive Value (PPV), % of good compounds discarded, sensitivity, specificity & Balanced accuracy
#I/P: A data frame with experimental and predicted labels (Labels assigned based on the thresholds defined by the user)

#O/P: A dataframe with all the metrics
def Calculate_allMetrics(Parameter_df):
        tn, fp, fn, tp = confusion_matrix(Parameter_df.Observed_Binaries,Parameter_df.Predicted_Binaries,labels=[0, 1]).ravel()
        
        #Return PPV and PercentCompoundsDiscard, only if the total number of positives or negatives are more than 5; Otherwise, return Nan
        PPV = precision_score(Parameter_df.Observed_Binaries,Parameter_df.Predicted_Binaries,zero_division=0)*100 if((tp+fp)>10) else np.nan
        Sensitivity = tp/(tp+fn)
        Specificity = tn/(tn+fp)
        PercentGoodCpds_Discard = (fn/(tn+fn))*100 if((tn+fn)>10) else np.nan
        #PercentGoodCpds_Discard = (fn/(tn+fn))*100
        BA = balanced_accuracy_score(Parameter_df.Observed_Binaries,Parameter_df.Predicted_Binaries)
        Metrics_df = pd.DataFrame([[PPV,PercentGoodCpds_Discard]]).round(1)
        return Metrics_df
 

#A plot to show how PPV,proportion of good compounds discarded and no. of compounds tested vary with respect to different thresholds

#I/P: Different threshold ranges, corresponding PPVs/NPVs and desired project threshold

def LinePlot(Threshold,Obs,Metric1,Metric2,DesiredProjectThreshold,Compounds_TestSet,min_thresh,max_thresh,class_annotation,Desired_Threshold_df,scale,PlotTitle):

    fig,ax = plt.subplots(figsize=(5,5))
    fig.canvas.header_visible = False
    #PPV_text = "PPV: Likelihood that compounds are true positives at each predicted threshold (higher = better)" +"\n"
    #CpdsDiscarded_text = "FOR: Likelihood to discard good compounds at each predicted threshold (lower = better)"+"\n"
    #TP_text = "True positive: Experimental value is " +class_annotation+" the corresponding predicted threshold"
    
    PPV_value = 'No True Positives were found at the selected threshold' if(math.isnan(Desired_Threshold_df.PPV)) else str(int(Desired_Threshold_df.PPV))+'%'
    CpdsDiscarded_value = 'No True Negatives were found at the selected threshold' if(math.isnan(Desired_Threshold_df.CompoundsDiscarded)) else str(int(Desired_Threshold_df.CompoundsDiscarded))+'%'
    #PPV_SelectedThresh_text = 'PPV at selected predicted threshold: '+ PPV_value
    #CpdsDiscarded_SelectedThresh_text = 'Likelihood to discard good compounds at selected predicted threshold: '+ CpdsDiscarded_value
    
    #print_note(f"* {PPV_text}")
    #print_note(f"* {CpdsDiscarded_text}")
    #print_note(f"* {TP_text}")
            
    # PPV & FOR values fluctuate a lot; It's important to smoothen the curves to see trends
    # Apply Savitzky-Golay filter with window size 5 and polynomial order 2
    Metric1 = savgol_filter(Metric1, window_length=5, polyorder=2)
    Metric2 = savgol_filter(Metric2, window_length=5, polyorder=2)
    Obs = savgol_filter(Obs, window_length=5, polyorder=2)

    if scale == 'log':

        ax.plot(np.log10(Threshold),Metric1,color='blue',marker="o")
        ax.plot(np.log10(Threshold),Metric2,color='orange',marker="o")
        
        ax = Reset_x_ticks(Threshold,ax)

        if ((len(Metric1)!=0) & (len(Metric2)!=0)):
            #call the function to calculate the longest arrow and display them on the plots
            Logged_Thresh = np.array(np.log10(Threshold))
            Max_Dist,Max_Thresh,Max_PPV,Max_FOR = LongestArrow(Logged_Thresh,Metric1,Metric2)

            plt.annotate(text='',xy=(Max_Thresh,Max_FOR), xytext=(Max_Thresh,Max_PPV),
                arrowprops=dict(arrowstyle='<->',color='plum'))
        
        ax2=ax.twinx()
        ax2.plot(np.log10(Threshold),Obs,color="grey",marker="o")
        ax2.set_xlim(min(np.log10(Threshold)),max(np.log10(Threshold))+0.5)

        #Display PPV/FOR metrics at the recommended thresholds in a table
        Max_PPV = 'N/A' if Max_PPV == -1 else int(Max_PPV)
        Max_FOR = 'N/A' if Max_FOR == -1 else int(Max_FOR)

        print_Unbiased_PPV_FOR_table(str(int(10**Max_Thresh)), str(Max_PPV), str(Max_FOR))
        
    else:  
        ax.plot(Threshold,Metric1,color='blue',marker="o")
        ax.plot(Threshold,Metric2,color='orange',marker="o")

        if ((len(Metric1)!=0) & (len(Metric2)!=0)):

            #call the function to calculate the longest arrow and display them on the plots
            Max_Dist,Max_Thresh,Max_PPV,Max_FOR = LongestArrow(Threshold,Metric1,Metric2)
            plt.annotate(text='',xy=(Max_Thresh,Max_FOR), xytext=(Max_Thresh,Max_PPV),
                arrowprops=dict(arrowstyle='<->',color='plum'))
        
        ax2=ax.twinx()
        ax2.plot(Threshold,Obs,color="grey",marker="o")

        #Display PPV/FOR metrics at the recommended thresholds in a table
        Max_PPV = 'N/A' if Max_PPV == -1 else int(Max_PPV)
        Max_FOR = 'N/A' if Max_FOR == -1 else int(Max_FOR)

        print_Unbiased_PPV_FOR_table(str(int(Max_Thresh)), str(Max_PPV), str(Max_FOR))
    
    ax.set_xlabel('Predicted threshold',fontweight='bold')
    ax.set_ylabel('PPV & FOR (unbiased) - Likelihood% ',fontweight='bold')
    ax2.set_ylabel('% of compounds tested',fontweight='bold')

    ax = Reset_y_ticks(ax)
    ax2 = Reset_y_ticks(ax2)

    ax.set_title(PlotTitle)

    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    
    #If the project has a very few compounds, highlight it with yellow, so that the users can treat the plots and statistics with caution; 20 is an arbitrary number - Can be changed depending on the needs
    if Compounds_TestSet < 20:
        ax.set_facecolor('lemonchiffon')
        
    myHandle = [Line2D([], [],  color='blue',linestyle='solid'),
                Line2D([], [],  color='orange',linestyle='solid'),
                Line2D([], [], color='grey', linestyle='solid')]
                
    PPV_label = 'Likelihood to extract good compounds at each predicted threshold'
    FOR_label = 'Likelihood to discard good compounds at each predicted threshold'
    ax.legend(handles=myHandle, labels = [PPV_label,FOR_label,'% of compounds tested (cumulative)'], bbox_to_anchor=(0.5, -0.2),loc='upper center',fontsize=7)
    fig.tight_layout()
    plt.show()
    
    
#A plot to show the classical Predicted vs Experimental correlations
#The threshold desired for a specific project is highlighted by an orange dotted line

#I/P: Experimental values, Predicted values, Desired project threshold and a title for the plot

def ScatterPlot(df,DesiredProjectThreshold,scale,PlotTitle):

    if ((len(df.Observed)>0) and (len(df.Predicted)>0)):

        if scale == 'log':
            #Discard rows, whose experimental or predicted values are below 1 - Done to avoid issues with negative logs in scatter plots
            #df = df[(df.Observed >= 1) & (df.Predicted >= 1)]
            fig,ax = plt.subplots(figsize=(5,5))
            fig.canvas.header_visible = False
            #plt.scatter(np.log10(df.Predicted),np.log10(df.Observed),color='grey')
            df['logPredicted']= np.log10(df.Predicted)
            df['logObserved'] = np.log10(df.Observed)
            sns.regplot(data = df,x = "logPredicted",y = "logObserved",color='grey',ci = None)
            plt.axhline(y=np.log10(DesiredProjectThreshold),color='orangered',linestyle='dotted')
            plt.axvline(x=np.log10(DesiredProjectThreshold),color='orangered',linestyle='dotted')
            plt.xlim(0,max(np.log10(df.Predicted)))
            plt.ylim(0,max(np.log10(df.Observed)))

            #Replace log values with actual values to make the plots much more intuitive
            ax = Reset_x_ticks(df.Predicted,ax)
            
            
            ylims = np.arange(min(np.log10(df.Observed))-0.5,max(np.log10(df.Observed))+0.5,step=0.5)
            ax.set_ylim(min(np.log10(df.Observed))-0.5,max(np.log10(df.Observed))+0.5)
            ax.yaxis.set_ticks(ylims)
            
            #Replace log values with actual values to make the plots much more intuitive
            y_ticks = ax.get_yticks()
            if (y_ticks<0).any():
                new_y_ticks = [round((10 ** y),2) for y in y_ticks]
            else:
                new_y_ticks = [int(10 ** y) for y in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(new_y_ticks, rotation = 45)
            
            #Compute observed vs Predicted correlations and RMSEs
            if len(df)>= 10:
                pearson_r2 = str(round((pearsonr(np.log10(df.Observed),np.log10(df.Predicted)).statistic)**2,2))
                r2 = str(round(r2_score(np.log10(df.Observed),np.log10(df.Predicted)),2))
                rmse = str(round(math.sqrt(mean_squared_error(np.log10(df.Observed),np.log10(df.Predicted))),2))
                print_metrics_table(pearson_r2, r2, rmse)
            else:
                print(Fore.RED+"Less than 10 compounds to compute R2 and RMSEs!"+Fore.RESET)

  
        else:
            fig,ax = plt.subplots(figsize=(5,5))
            fig.canvas.header_visible = False
            sns.regplot(data = df,x = "Predicted",y = "Observed",color='grey',ci = None)
            plt.axhline(y=DesiredProjectThreshold,color='orangered',linestyle='dotted')
            plt.axvline(x=DesiredProjectThreshold,color='orangered',linestyle='dotted')
            
            #Compute observed vs Predicted correlations and RMSEs
            if len(df)>= 10:
                pearson_r2 = str(round((pearsonr(df.Observed,df.Predicted).statistic)**2,2))
                r2 = str(round(r2_score(df.Observed,df.Predicted),2))
                rmse = str(round(math.sqrt(mean_squared_error(df.Observed,df.Predicted)),2))
                print_metrics_table(pearson_r2, r2, rmse)
            else:
                print(Fore.RED+"Less than 10 compounds to compute R2 and RMSEs!"+Fore.RESET)

            
        plt.xlabel('Predicted',fontweight='bold')
        plt.ylabel('Experimental',fontweight='bold')
        plt.title(PlotTitle)
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        #plt.axline((0,0),(0,0),color='grey')
        #plt.axline((0,0),slope=1,color='grey')
        
        
        #plt.annotate("r-squared = {:.2f}".format(r2), (0, 1), ha='left', va='top')
        #plt.annotate("rmse = {:.2f}".format(rmse), (0, 1), ha='right', va='top')
        plt.tight_layout()
        plt.show()

    else:
        print(Fore.RED+"No sufficient datapoints to generate plots!"+Fore.RESET)


    
    
#A plot to show the likelihood to predict below or above a certain experimental threshold

#Purpose: What should be the predicted threshold, if the users want to be sure about acquiring maximum proportion of compounds below the experimental threshold?

#I/P: Different threshold ranges, corresponding predicted likelihoods and desired experimental project threshold

def LikelihoodPlot(Threshold,Obs,Pred_Pos_Likelihood,Pred_Neg_Likelihood,DesiredProjectThreshold,Compounds_TestSet,min_thresh,max_thresh,class_annotation,Pos_class,Desired_Threshold_df,scale,PlotTitle):
    fig,ax = plt.subplots(figsize=(5,5))
    fig.canvas.header_visible = False
    
    Threshold_label =  'Selected Experimental Threshold: '+ Pos_class +  str(DesiredProjectThreshold)
    
    # PPV & FOR values fluctuate a lot; It's important to smoothen the curves to see trends
    # Apply Savitzky-Golay filter with window size 5 and polynomial order 2
    Pred_Pos_Likelihood = savgol_filter(Pred_Pos_Likelihood, window_length=5, polyorder=2)
    Pred_Neg_Likelihood = savgol_filter(Pred_Neg_Likelihood, window_length=5, polyorder=2)
    Obs = savgol_filter(Obs, window_length=5, polyorder=2)


    if scale == 'log':
        ax.plot(np.log10(Threshold),Pred_Pos_Likelihood,color='turquoise',marker="o")
        ax.plot(np.log10(Threshold),Pred_Neg_Likelihood,color='indigo',marker="o")
        ax = Reset_x_ticks(Threshold,ax)
   
        if ((len(Pred_Pos_Likelihood)!=0) & (len(Pred_Neg_Likelihood)!=0)):
            #call the function to calculate the longest arrow and display them on the plots
            Logged_Thresh = np.array(np.log10(Threshold))
            Max_Dist,Max_Thresh,Max_PPV,Max_FOR = LongestArrow(Logged_Thresh,Pred_Pos_Likelihood,Pred_Neg_Likelihood)
            
            plt.annotate(text='',xy=(Max_Thresh,Max_FOR), xytext=(Max_Thresh,Max_PPV),
                arrowprops=dict(arrowstyle='<->',color='plum'))
            #If PPV at the desired threshold is 0, no arrows pointing to PPV can be drawn
            if (~np.isnan(Desired_Threshold_df.Pred_Pos_Likelihood[0])):
                plt.annotate(text='',xy=(np.log10(DesiredProjectThreshold),Desired_Threshold_df.Pred_Neg_Likelihood[0]), xytext=(np.log10(DesiredProjectThreshold),Desired_Threshold_df.Pred_Pos_Likelihood[0]),
                    arrowprops=dict(arrowstyle='<->',color='green'))
 
        ax2=ax.twinx()
        ax2.plot(np.log10(Threshold),Obs,color="grey",marker="o")
        ax2.set_xlim(min(np.log10(Threshold)),max(np.log10(Threshold))+0.5)
        
       
        #Display PPV/FOR metrics at pre-selected and recommended thresholds in a table
        Desired_Threshold_df.Pred_Pos_Likelihood = 'N/A' if (math.isnan(Desired_Threshold_df.Pred_Pos_Likelihood)) else  int(Desired_Threshold_df.Pred_Pos_Likelihood)
        Desired_Threshold_df.Pred_Neg_Likelihood = 'N/A' if (math.isnan(Desired_Threshold_df.Pred_Neg_Likelihood)) else  int(Desired_Threshold_df.Pred_Neg_Likelihood)
        Max_PPV = 'N/A' if Max_PPV == -1 else int(Max_PPV)
        Max_FOR = 'N/A' if Max_FOR == -1 else int(Max_FOR)

        print_PPV_FOR_table(DesiredProjectThreshold, str(Desired_Threshold_df.Pred_Pos_Likelihood[0]), str(Desired_Threshold_df.Pred_Neg_Likelihood[0]),
                        str(round(10**Max_Thresh)), str(Max_PPV), str(Max_FOR))

    else:
        ax.plot(Threshold,Pred_Pos_Likelihood,color='turquoise',marker="o")
        ax.plot(Threshold,Pred_Neg_Likelihood,color='indigo',marker="o")

        if ((len(Pred_Pos_Likelihood)!=0) & (len(Pred_Neg_Likelihood)!=0)):
            #call the function to calculate the longest arrow and display them on the plots
            Max_Dist,Max_Thresh,Max_PPV,Max_FOR = LongestArrow(Threshold,Pred_Pos_Likelihood,Pred_Neg_Likelihood)

            plt.annotate(text='',xy=(Max_Thresh,Max_FOR), xytext=(Max_Thresh,Max_PPV),
                arrowprops=dict(arrowstyle='<->',color='plum'))

            plt.annotate(text='',xy=(DesiredProjectThreshold,Desired_Threshold_df.Pred_Neg_Likelihood), xytext=(DesiredProjectThreshold,Desired_Threshold_df.Pred_Pos_Likelihood),
                arrowprops=dict(arrowstyle='<->',color='green'))
        
        ax2=ax.twinx()
        ax2.plot(Threshold,Obs,color="grey",marker="o")
    
        #Display PPV/FOR metrics at pre-selected and recommended thresholds in a table
        Desired_Threshold_df.Pred_Pos_Likelihood = 0 if (math.isnan(Desired_Threshold_df.Pred_Pos_Likelihood)) else  int(Desired_Threshold_df.Pred_Pos_Likelihood)
        Desired_Threshold_df.Pred_Neg_Likelihood = 0 if (math.isnan(Desired_Threshold_df.Pred_Neg_Likelihood)) else  int(Desired_Threshold_df.Pred_Neg_Likelihood)
        Max_PPV = 'N/A' if Max_PPV == -1 else int(Max_PPV)
        Max_FOR = 'N/A' if Max_FOR == -1 else int(Max_FOR)

        print_PPV_FOR_table(DesiredProjectThreshold, str(Desired_Threshold_df.Pred_Pos_Likelihood[0]), str(Desired_Threshold_df.Pred_Neg_Likelihood[0]),
                        str(round(Max_Thresh)), str(Max_PPV), str(Max_FOR))

    ax.set_xlabel('Predicted threshold',fontweight='bold')
    ax.set_ylabel('PPV & FOR (using SET) -  Likelihood% ',fontweight='bold')


    ax2.set_ylabel('% of compounds tested',fontweight='bold')
    ax = Reset_y_ticks(ax)
    ax2 = Reset_y_ticks(ax2)
    

    PlotTitle_ExpThresh = PlotTitle
    ax.set_title(PlotTitle_ExpThresh)

    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    
    #If the project has a very few compounds, highlight it with yellow, so that the users can treat the plots and statistics with caution; 20 is an arbitrary number - Can be changed depending on the needs
    if Compounds_TestSet < 20:
        ax.set_facecolor('lemonchiffon')
    
    
    myHandle = [Line2D([], [],  color='white'),
                Line2D([], [],  color='turquoise',linestyle='solid'),
                Line2D([], [],  color='indigo',linestyle='solid'),
                Line2D([], [], color='grey', linestyle='solid')]
    
    #Rec_Threshold_label =  'Recommended Experimental Threshold: '+ Pos_class +  str(Max_Thresh)
    Likelihood_Pos_label = 'Likelihood to extract good compounds at the pre-selected experimental threshold'
    Likelihood_Neg_label = 'Likelihood to discard good compounds at the pre-selected experimental threshold'
    ax.legend(handles=myHandle, labels = [Threshold_label,Likelihood_Pos_label,Likelihood_Neg_label,'% of compounds tested (cumulative)'], bbox_to_anchor=(0.5, -0.2),loc='upper center',fontsize=7)
    plt.tight_layout()
    plt.show()

#A plot to show model stability over time

#Purpose: Is the model performance improving by adding new chemistry from the project

#I/P: Model versions together with the corresponding experimental and predicted values

def ModelStabilityPlot(df,scale,PlotTitle):
    #Extarct the months and years from the date
    df['ModelVersionDate'] = pd.DataFrame(df['ModelVersion'].apply(lambda x: x.split("-")[0]))
    df['ModelVersionDate'] = pd.to_datetime(df['ModelVersionDate'])
    df['Model_Month_year'] = df['ModelVersionDate'].apply(lambda x: x.strftime("%b %Y"))
    #Group by Month/Year and compute RMSEs
    ModelPerf_df = pd.DataFrame()
    ModelPerf_df['RMSE']= df.groupby('Model_Month_year')[['Observed','Predicted']].apply(lambda x: rmse_score(x['Observed'],x['Predicted'],scale))
    ModelPerf_df['NoOfCpds']= df.groupby('Model_Month_year')['Observed'].count()
    ModelPerf_df['ModelVersion']=  ModelPerf_df.index
    
    ModelPerf_df = ModelPerf_df.set_index(pd.to_datetime(ModelPerf_df['ModelVersion']).rename('datetime'))
    
    ModelPerf_df_sorted = ModelPerf_df.sort_index()
    ModelPerf_df_sorted = ModelPerf_df_sorted[ModelPerf_df_sorted.NoOfCpds >=5] #Consider only those RMSES computed based on atleast 5 compounds
    #print(len(ModelPerf_df_sorted))
    
    #Plot RMSEs / No. of compounds against the Model versions
    if len(ModelPerf_df_sorted) >1:
    
        # RMSEs & No. of compounds fluctuate a lot; It's important to smoothen the curves to see trends
        # Apply Savitzky-Golay filter with window size 5 and polynomial order 2
        #ModelPerf_df_sorted['RMSE'] = savgol_filter(ModelPerf_df_sorted['RMSE'], window_length=5, polyorder=2).round(2)
        #ModelPerf_df_sorted['NoOfCpds'] = savgol_filter(ModelPerf_df_sorted['NoOfCpds'], window_length=5, polyorder=2).astype(int)
        
        fig,ax = plt.subplots(figsize=(5,5))
        fig.canvas.header_visible = False
        ax2=ax.twinx()
        ax.plot(ModelPerf_df_sorted.ModelVersion,ModelPerf_df_sorted.RMSE,color='deeppink',marker="o")
        ax2.plot(ModelPerf_df_sorted.ModelVersion,ModelPerf_df_sorted.NoOfCpds,color='grey',marker="o")
    
        ax.set_ylim(0,2.5)
        ax.set_xlabel('Model Version',fontweight='bold')
        ax.set_ylabel('RMSE (log scale)',fontweight='bold')
        ax2.set_ylabel('No. of compounds',fontweight='bold')
        
    
        ax.set_xticklabels(ModelPerf_df_sorted.ModelVersion,rotation = 90,fontsize=8)
        #ax.set_yticklabels(ModelPerf_df_sorted.RMSE,fontsize=8)
        #ax2.set_yticklabels(ModelPerf_df_sorted.NoOfCpds.astype(int),fontsize=8)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
        PlotTitle_Stability = PlotTitle + ' - Model performance over time'
        ax.set_title(PlotTitle_Stability)
    
        myHandle = [Line2D([], [],  color='deeppink'),
                Line2D([], [],  color='grey',linestyle='solid')]
        RMSE_label = 'RMSEs over time'
        Compounds_label = 'No. of compounds considered for prediction each month'
        ax.legend(handles=myHandle, labels = [RMSE_label,Compounds_label], bbox_to_anchor=(0.5 ,-0.3),loc='upper center',fontsize=7)
        fig.tight_layout()
        plt.show()
        
    else:
        print('\n')
        print(Fore.RED+"No sufficient data to track model performances for "+ PlotTitle +" over time!"+Fore.RESET)
        
    
        
#A plot to show the distribution of experimental values corresponding to an end point for a project

#Purpose: Is the project focusing on measuring those experimental values desired to be within certain limits

#I/P: First sample registration date and the experimental values

def Exp_Values_Dist(df,DesiredProjectThreshold,scale,PlotTitle): 
    #df = df[(df.Observed >= 1)]
    df2 = pd.DataFrame()
    #Group by Month/Year and compute mean experimental values
    df['Month_Year'] = pd.DataFrame(df['SampleRegDate'].apply(lambda x: datetime.strptime(x, '%d-%b-%Y').strftime('%b %Y')))
    df2['NoOfCpds'] = df.groupby('Month_Year')['Observed'].agg(total='count')['total']
    df2['Median_Exp'] = df.groupby('Month_Year')['Observed'].agg(Median='median')['Median']
    #df2['Mean_Exp'] = df.groupby('Month_Year')['Observed'].apply(lambda x: gmean(x))
    df2['Min'] = df.groupby('Month_Year')['Observed'].agg(minimum='min')['minimum']
    df2['Max'] = df.groupby('Month_Year')['Observed'].agg(maximum='max')['maximum']
    df2 = df2.dropna(how='any')
    df2['RegDate_Month_Year'] = pd.to_datetime(df2.index)
    df2_sorted = df2.sort_values(by='RegDate_Month_Year').reset_index()
    df2_sorted = df2_sorted[df2_sorted['RegDate_Month_Year'] >= '01-01-2021'] # Consider only the data corresponding to last 3 years
    
    df2_sorted = df2_sorted[df2_sorted.NoOfCpds >=5] #Consider only those means and SDs computed based on atleast 5 compounds

    
    #Plot Median experimental values / No. of compounds against the Sample Registration Date
    if scale == 'log':
        #df2_sorted = df2_sorted[((np.log10(df2_sorted.Median_Exp)-np.log10(df2_sorted['Min'])) > 0) & ((np.log10(df2_sorted['Max'])-np.log10(df2_sorted.Median_Exp)) >0)]#Make sure to check if the difference between the mean and min/max values are positive; Negative values are not accepted, while plotting error bars
        if len(df2_sorted) >1:
            fig,ax = plt.subplots(figsize=(5,5))
            fig.canvas.header_visible = False
            ax.plot(df2_sorted.Month_Year,np.log10(df2_sorted.Median_Exp),marker='o',color='dodgerblue')
            ylims = np.arange(min(np.log10(df.Observed))-0.5,max(np.log10(df.Observed))+0.5,step=0.5)
            ax.set_ylim(min(np.log10(df.Observed))-0.5,max(np.log10(df.Observed))+0.5)

            #ax.errorbar(df2_sorted.Month_Year,np.log10(df2_sorted.Mean_Exp),yerr= np.stack([(np.log10(df2_sorted.Mean_Exp)-np.log10(df2_sorted['Min']).to_numpy()),((np.log10(df2_sorted['Max'])-np.log10(df2_sorted.Mean_Exp)).to_numpy())]), color='maroon',ecolor= 'lightcoral',fmt='-o',capsize=5)
            #ax2.plot(df2_sorted.Month_Year,df2_sorted.NoOfCpds,color='grey',marker="o")
                
            ax.set_xlabel('Sample Registration Date',fontweight='bold')
            ax.set_ylabel('Experimental values - Median',fontweight='bold')
            #ax2.set_ylabel('No. of compounds',fontweight='bold')

            
            ax.set_xticklabels(df2_sorted.Month_Year,rotation = 90,fontsize=8)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_ticks(ylims)
          
            #ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                
            #Replace log values with actual values to make the plots much more intuitive
            y_ticks = ax.get_yticks()
            new_y_ticks = [int(10 ** y) for y in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(new_y_ticks, rotation = 45,fontsize=8)
            ax.axhline(y=np.log10(DesiredProjectThreshold),color='orangered',linestyle='dotted')
            
            PlotTitle_Stability = PlotTitle + ' - Experimental values over time'
            ax.set_title(PlotTitle_Stability)
            plt.rc('xtick',labelsize=8)
            plt.rc('ytick',labelsize=8)

            
            myHandle = [Line2D([], [],  color='dodgerblue'),
            Line2D([], [],  color='orange',linestyle=':')]
            #Exp_label = 'Variation in experimental values over time (Since Jan 2021)'
            MeanExp_label = 'Median experimental values during each time period'
            Threshold_label = 'Desired project threshold'
            ax.legend(handles=myHandle, labels = [MeanExp_label,Threshold_label], bbox_to_anchor=(0.5 ,-0.3),loc='upper center',fontsize=7)
            fig.tight_layout()
            plt.show()
            
        else:
            print('\n')
            print(Fore.RED+"No sufficient data beyond Jan 2021 to track experimental values for " + PlotTitle + " over time!"+Fore.RESET)
    else:
        #df2_sorted = df2_sorted[((df2_sorted.Median_Exp-df2_sorted['Min']) > 0) & ((df2_sorted['Max']-df2_sorted.Median_Exp) >0)]#Make sure to check if the difference between the mean and min/max values are positive; Negative values are not accepted, while plotting error bars
        if len(df2_sorted) >1:
            fig,ax = plt.subplots(figsize=(6,6))
            fig.canvas.header_visible = False
            increment_factor = (max(df.Observed)-min(df.Observed))/5 #Changed to accommodate all linear end points other than logD
            ylims = np.arange(min(df.Observed)-increment_factor,max(df.Observed)+increment_factor,step=increment_factor)
            ax.plot(df2_sorted.Month_Year,df2_sorted.Median_Exp,marker="o")
            ax.set_ylim(min(df.Observed)-increment_factor,max(df.Observed)+increment_factor)
            


            #ax.errorbar(df2_sorted.Month_Year,df2_sorted.Mean_Exp,yerr= np.stack([(df2_sorted.Mean_Exp-df2_sorted['Min']).to_numpy(),(df2_sorted['Max']-df2_sorted.Mean_Exp).to_numpy()]), color='maroon',ecolor='lightcoral',fmt='-o',capsize=5)
            ax.plot(df2_sorted.Month_Year,df2_sorted.Median_Exp,marker="o",color='dodgerblue')
            
            ax.set_xlabel('Sample Registration Date',fontweight='bold')
            ax.set_ylabel('Experimental values - Median',fontweight='bold')
            
            ax.set_xticklabels(df2_sorted.Month_Year,rotation = 90)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_ticks(ylims)        
        
            #Replace log values with actual values to make the plots much more intuitive
            y_ticks = ax.get_yticks()
            new_y_ticks = [round(y,1) for y in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(new_y_ticks, rotation = 45)
            ax.axhline(y=DesiredProjectThreshold,color='orangered',linestyle='dotted')
        
            PlotTitle_Stability = PlotTitle + ' - Experimental values over time'
            ax.set_title(PlotTitle_Stability)
            plt.rc('xtick',labelsize=8)
            plt.rc('ytick',labelsize=8)

            myHandle = [Line2D([], [],  color='dodgerblue'),
                        Line2D([], [],  color='orange',linestyle=':')]
            #Exp_label = 'Variation in experimental values over time (Since Jan 2021)'
            MeanExp_label = 'Median experimental values during each time period'
            Threshold_label = 'Desired project threshold'
            ax.legend(handles=myHandle, labels = [MeanExp_label,Threshold_label], bbox_to_anchor=(0.5 ,-0.3),loc='upper center',fontsize=7)
            fig.tight_layout()
            plt.show()
        else:
            print('\n')
            print(Fore.RED+"No sufficient data beyond Jan 2021 to track experimental values for " + PlotTitle + " over time!"+Fore.RESET)

 
'''    
A function to evaluate a model's predictive potential for an end point considering all compounds in that project
I/P: Data - Dataframe extracted from your D360 query

observed_column - Experimental end point you are interested in

predicted_column - First predicted values from PIP corresponding to your end point; Use the column name that includes 'Earliest'

trainingSet_column - We are interested in assessing project-specific model performances only for compounds not in the training set;Use the column that includes the following term 'traintest' to extract true test set compounds. Please make sure that this column is annotated only with values like train and test. If you have other annotations, 'Observed_Predicted_df[Observed_Predicted_df.CompoundsInTrainingSet=='test']' in the PredictiveValidity function should be replaced.

PosClass - Should your positive class be above or below the threshold; Use '>' for above the threshold and vice-versa

DesiredProjectThreshold - Specify the desired threshold for your project

PlotTitle - Specify the name that you want to appear on your plot
'''

def PredictiveValidity(Data,observed_column,predicted_column,trainingSet_Column,PosClass,DesiredProjectThreshold,ModelVersion,SampleRegDate,scale,PlotTitle):

    #Convert the experimental values to numeric, provided the column is a string

    Observed_Parameter = pd.to_numeric(Data[observed_column].astype(str).str.replace('>|<|NV|;|\?|,','',regex=True),errors='coerce')
    Predicted_Parameter = Data[predicted_column]

    Observed_Predicted_df = pd.concat(
        [Data['Compound Name'], Observed_Parameter, Predicted_Parameter, Data[trainingSet_Column], Data[ModelVersion], Data[SampleRegDate]], axis=1, keys=['Compound Name','Observed','Predicted','CompoundsInTrainingSet','ModelVersion','SampleRegDate']).dropna(subset=['Compound Name','Observed','Predicted','CompoundsInTrainingSet'])

    Observed_Predicted_all = Observed_Predicted_df

    # print(Observed_Predicted_all)
    # print(Observed_Predicted_all.shape)

    #print_note(f"\n ### Overview\n ---")
    Total_Cpds = len(Observed_Predicted_df)
    #print_note(f"##### Compounds with measured values : {Total_Cpds}")
    
    #Extract compounds in the training and test set
    Observed_Predicted_train = Observed_Predicted_df[Observed_Predicted_df.CompoundsInTrainingSet=='train']
    Observed_Predicted_df = Observed_Predicted_df[Observed_Predicted_df.CompoundsInTrainingSet.isin(['test',np.nan])]
    Compounds_TestSet = len(Observed_Predicted_df)
    TrainingSet = Total_Cpds- Compounds_TestSet
    #print_note(f"##### Training Set: {TrainingSet}")
    #print_note(f"##### Prospective Validation Set: {Compounds_TestSet}")

    # print(Observed_Predicted_df)

    if not is_in_notebook():
        print('\n')
        print('*********************************************************************************')
        print('*********************************************************************************')
        print('\n')

    if (len(Observed_Predicted_df)>0):
        
        #Call the thresh function to get the threshold ranges for calculating the various metrics
        min_thresh,max_thresh,Thresholds_selection = Thresh_Selection(Observed_Predicted_df.Predicted,DesiredProjectThreshold,scale)

        print_note("\n --- \n")
        #Print the number of compounds below and above the desired project threshold
        Compounds_BelowThresh = len(Observed_Predicted_df[Observed_Predicted_df.Observed <= DesiredProjectThreshold])
        Compounds_AboveThresh = len(Observed_Predicted_df[Observed_Predicted_df.Observed > DesiredProjectThreshold])
        Compounds_BelowThresh_text = str(len(Observed_Predicted_df[Observed_Predicted_df.Observed <= DesiredProjectThreshold]))
        Compounds_AboveThresh_text = str(len(Observed_Predicted_df[Observed_Predicted_df.Observed > DesiredProjectThreshold]))

        #Estimate the number of good compounds made so far
        if PosClass == '<':
            ratio_GoodCpds = Compounds_BelowThresh / (Compounds_BelowThresh + Compounds_AboveThresh)
        else:
            ratio_GoodCpds = Compounds_AboveThresh / (Compounds_BelowThresh + Compounds_AboveThresh)
            
        ratio_GoodCpds  = str(int(ratio_GoodCpds*100)) +'%'
        print_note(f"\n ### Overview\n ---")
        print_cpds_info_table(Total_Cpds,TrainingSet,Compounds_TestSet,Compounds_BelowThresh_text, Compounds_AboveThresh_text, ratio_GoodCpds) 

        #Plot to show Experimental values over time should be displayed, irrespective of the size of the test set
        print_note(f"\n --- \n ### Experimental values over time")
        Exp_Values_Dist(Observed_Predicted_all,DesiredProjectThreshold,scale,PlotTitle)

        # Model evaluation
        print_note(f"\n --- \n ### Model evaluation")

        # Print a caution statement, if there are a very few compounds in the test set
        if ((Compounds_TestSet !=0) and (Compounds_TestSet < 20)):
            print(Fore.RED + 'Less than 20 compounds in the prospective validation set! Please treat the statistics with caution.' + Fore.RESET)
        
        PlotTitle_Test = PlotTitle + " - Prospective Validation Set"
        print_note(f"\n#### Predicted vs Experimental Values (prospective)")
        ScatterPlot(Observed_Predicted_df,DesiredProjectThreshold,scale,PlotTitle_Test)

    #Scatter plot for training set
    print_note("\n #### Predicted vs Experimental Values (training set)")
    if (TrainingSet > 0):
        PlotTitle_Train = PlotTitle + " - Training Set"
        ScatterPlot(Observed_Predicted_train,DesiredProjectThreshold,scale,PlotTitle_Train)

    else:
        print(Fore.RED + 'Training set is empty - Not possible to generate scatter plots or compute any metrics!' + Fore.RESET)

    if (len(Observed_Predicted_df)>0) and Compounds_TestSet >= 10:
        print_note(f"\n#### Model performance over time")  
        print_note(f"\n##### RMSE")  
        ModelStabilityPlot(Observed_Predicted_df,scale,PlotTitle)

        #Calculating time dependant MPO scores/ Displaying the different similarity/correlation scores(Actual vs time-weighted)        
        print_note(f"\n##### Similarity of prospective data to training set")
        discount_factor = 0.9 #A value set arbitrarily - Might have to be optimized based on a few runs for a couple of pilot projects
        time_weighted_score_plot(Observed_Predicted_all,discount_factor,PlotTitle)

    AllMetrics_df = pd.DataFrame()
        
    if Compounds_TestSet >= 10:    
        for thresh in Thresholds_selection[1:]: #Exclude the first threshold, as there wouldn't be many compounds below the minimal threshold - Generated statistics can be misleading
            #Estimate the number of datapoints at or below each threshold
            NoOfObs =  len(Observed_Predicted_df[Observed_Predicted_df.Predicted<=thresh])
            Compounds_percent = (NoOfObs/len(Observed_Predicted_df))*100
        

            #If the PosClass is '>', anything above that threshold would be considered as positive; If the PosClass is '<', anything below that threshold would fall under the positive class
            if PosClass == '>':
                class_annotation = 'above'
                
                #Mapping to binaries based on different thresholds
                Observed_Predicted_df['Observed_Binaries'] = Observed_Predicted_df['Observed'].map(lambda x: int(x > thresh))
                Observed_Predicted_df['Predicted_Binaries'] = Observed_Predicted_df['Predicted'].map(lambda x: int(x > thresh))
                
                #Identifying the predicted likelihood to extract good compounds at a selected experimental threshold
                Observations_Pos_Extract = Observed_Predicted_df[Observed_Predicted_df.Predicted > thresh]
                if len(Observations_Pos_Extract)>10:
                    Pred_Pos_Likelihood = (len(Observations_Pos_Extract[Observations_Pos_Extract['Observed'] > DesiredProjectThreshold])/len(Observations_Pos_Extract))*100
                else:
                    Pred_Pos_Likelihood = math.nan
                    
                #Identifying the likelihood to remove good compounds at a selected experimental threshold
                Observations_Neg_Extract = Observed_Predicted_df[Observed_Predicted_df.Predicted <= thresh]
                if len(Observations_Neg_Extract)>10:
                    Pred_Neg_Likelihood = (len(Observations_Neg_Extract[Observations_Neg_Extract['Observed'] > DesiredProjectThreshold])/len(Observations_Neg_Extract))*100
                else:
                    Pred_Neg_Likelihood = math.nan
                
                AllMetrics = pd.concat([pd.DataFrame([[date.today(),thresh,Compounds_percent,Pred_Pos_Likelihood,Pred_Neg_Likelihood]]),Calculate_allMetrics(Observed_Predicted_df)],axis=1)
                AllMetrics_df = pd.concat([AllMetrics_df,AllMetrics],axis=0)
            
            else:
                class_annotation = 'below'
                Observed_Predicted_df['Observed_Binaries'] = Observed_Predicted_df['Observed'].map(lambda x: int(x <= thresh))
                Observed_Predicted_df['Predicted_Binaries'] = Observed_Predicted_df['Predicted'].map(lambda x: int(x <= thresh))
                
                Observations_Pos_Extract = Observed_Predicted_df[Observed_Predicted_df.Predicted <= thresh]
                if len(Observations_Pos_Extract)>10:
                    Pred_Pos_Likelihood = (len(Observations_Pos_Extract[Observations_Pos_Extract['Observed'] <= DesiredProjectThreshold])/len(Observations_Pos_Extract))*100
                else:
                    Pred_Pos_Likelihood = math.nan
                    
                Observations_Neg_Extract = Observed_Predicted_df[Observed_Predicted_df.Predicted > thresh]
                if len(Observations_Neg_Extract)>10:
                    Pred_Neg_Likelihood = (len(Observations_Neg_Extract[Observations_Neg_Extract['Observed'] <= DesiredProjectThreshold])/len(Observations_Neg_Extract))*100
                else:
                    Pred_Neg_Likelihood = math.nan
                    
                AllMetrics = pd.concat([pd.DataFrame([[date.today(),thresh,Compounds_percent,Pred_Pos_Likelihood,Pred_Neg_Likelihood]]),Calculate_allMetrics(Observed_Predicted_df)],axis=1)
                AllMetrics_df = pd.concat([AllMetrics_df,AllMetrics],axis=0)
        
        AllMetrics_df.columns=['Calculation Date','Threshold','CompoundsTested','Pred_Pos_Likelihood','Pred_Neg_Likelihood','PPV','CompoundsDiscarded']
        AllMetrics_df = AllMetrics_df.drop_duplicates()#Duplicates exist, if the threshold chosen by the user is one among the 50 thresholds chosen for analysis; Remove them prior to calling the plot functions

        #Printing statistics at desired project threshold
        Desired_Threshold_df = AllMetrics_df[AllMetrics_df.Threshold==DesiredProjectThreshold]
        #print(Desired_Threshold_df)

        #print('\033[1m{:30s}\033[0m'.format('Metrics at Selected Experiment threshold'))
        Thresh_class = PosClass +" "+ str(DesiredProjectThreshold)
        #print('Selected Experimental Threshold: ',Thresh_class)
        #print('Positive class : ',PosClass,' selected threshold')

    
        #Sort the metrics data frame before plotting to have the thresholds in ascending order in x-axis
        AllMetrics_df_sorted = AllMetrics_df.sort_values(by=['Threshold'],ascending = False)


        # Model usage advice
        print_note(f"\n --- \n ### Model usage advice")
        
        print_note(f"\n#### What predicted threshold gives best enrichment?")
        LikelihoodPlot(AllMetrics_df_sorted.Threshold,AllMetrics_df_sorted['CompoundsTested'],AllMetrics_df_sorted.Pred_Pos_Likelihood,AllMetrics_df_sorted.Pred_Neg_Likelihood,DesiredProjectThreshold,Compounds_TestSet,min_thresh,max_thresh,class_annotation,PosClass,Desired_Threshold_df,scale,PlotTitle)

        print_note(f"\n#### Explore other experimental thresholds to aim for")
        LinePlot(AllMetrics_df_sorted.Threshold,AllMetrics_df_sorted['CompoundsTested'],AllMetrics_df_sorted.PPV,AllMetrics_df_sorted.CompoundsDiscarded,DesiredProjectThreshold,Compounds_TestSet,min_thresh,max_thresh,class_annotation,Desired_Threshold_df,scale,PlotTitle)

    else:
        print('\n')
        print_note(f"\n --- \n ### Predicted vs Experimental Values")
        ScatterPlot(Observed_Predicted_df,DesiredProjectThreshold,scale,PlotTitle)
        print(Fore.RED+'Less than 10 compounds with measured values in the prospective validation set! Not possible to compute any metrics!'+Fore.RESET)
    
    return

  
'''
A function to evaluate a model's predictive potential for an end point considering specific series in that project
Parameters to be passed to this function are the same as 'PredictiveValidity' function. The only difference is that an additional column name, where you have information about the series should be passed too.

'''

def PredictiveValidity_Series(Data,observed_column,predicted_column,trainingSet_Column,PosClass,DesiredProjectThreshold,series_column,ModelVersion,SampleRegDate,scale,PlotTitle):
    Observed_Parameter = pd.to_numeric(Data[observed_column].astype(str).str.replace('>|<|NV|;|\?|,','',regex=True),errors='coerce')
    Predicted_Parameter = Data[predicted_column]
    Observed_Predicted_df = pd.concat([Data['Compound Name'],Observed_Parameter,Predicted_Parameter,Data[trainingSet_Column],Data[series_column],Data[ModelVersion],Data[SampleRegDate]],axis=1,keys=['Compound Name','Observed','Predicted','CompoundsInTrainingSet','Series','ModelVersion','SampleRegDate']).dropna(subset=['Compound Name','Observed','Predicted','CompoundsInTrainingSet','Series'])
    # print_note(Observed_Predicted_df.to_markdown())
    
    #Extract compounds in the training and test set
    Total_Cpds = len(Observed_Predicted_df)
    TrainingSet = Observed_Predicted_df[Observed_Predicted_df.CompoundsInTrainingSet=='train']
    Training_Series_Count = TrainingSet.groupby(by='Series')['Compound Name'].count()

    
    Observed_Predicted_all = Observed_Predicted_df
    Observed_Predicted_df = Observed_Predicted_df[Observed_Predicted_df.CompoundsInTrainingSet.isin(['test',np.nan])]
    Test_Series_Count = Observed_Predicted_df.groupby(by='Series')['Compound Name'].count()

    '''
    print_note(f"\n ### Overview\n ---")
    print_note(f"##### Number of Compounds in Series: {len(Observed_Predicted_all)}")
    summary_df = pd.DataFrame({"Prospective Validation Set": Test_Series_Count.values}, index=Test_Series_Count.index)
    summary_df["Training Set"] = Training_Series_Count
    summary_df = summary_df.fillna(0).astype(int)
    summary_df = summary_df[['Training Set', 'Prospective Validation Set']]
    print_note(summary_df.to_markdown())
    '''
    if (len(Test_Series_Count)==0):
        print('\n')
        print(Fore.RED+'No compounds with measured values for this specific series in the prospective validation set! Not possible to compute any metrics!'+Fore.RESET)
        
    if not is_in_notebook():
        print('\n')
        print('*********************************************************************************')
        print('*********************************************************************************')
        print('\n')

    Thresh_class = PosClass +" "+ str(DesiredProjectThreshold)
    #print('Selected Experimental Threshold: ',Thresh_class)
    #print('Positive class : ',PosClass,' selected threshold')

    
    #Compute the metrics corresponding to different thresholds for every series
    for series in Test_Series_Count.index:
        AllMetrics_df = pd.DataFrame()
        Training_Series_df = TrainingSet[TrainingSet['Series']==series]
        Series_all_df = Observed_Predicted_all[Observed_Predicted_all['Series']==series]
        Series_df = Observed_Predicted_df[Observed_Predicted_df['Series']==series]

        TotalCpds_SeriesSpecificCount = len(Series_all_df)
        Training_SpecificSeries_count = len(Training_Series_df)
        SpecificSeries_count = len(Series_df)
        
        PlotTitle_series = PlotTitle+"  (Series: "+str(series)+")"        
              
        if ((SpecificSeries_count!=[]) and (len(Series_df.Predicted)>0)):

            #Print the number of compounds below and above the desired project threshold
            Compounds_BelowThresh = len(Series_df[Series_df.Observed <= DesiredProjectThreshold])
            Compounds_AboveThresh = len(Series_df[Series_df.Observed > DesiredProjectThreshold])
            Compounds_BelowThresh_text = str(len(Series_df[Series_df.Observed <= DesiredProjectThreshold]))
            Compounds_AboveThresh_text = str(len(Series_df[Series_df.Observed > DesiredProjectThreshold]))
        
            #Estimate the number of good compounds made so far
            if PosClass == '<':
                ratio_GoodCpds = Compounds_BelowThresh / (Compounds_BelowThresh + Compounds_AboveThresh)
            else:
                ratio_GoodCpds = Compounds_AboveThresh / (Compounds_BelowThresh + Compounds_AboveThresh)
            
            ratio_GoodCpds_text  = str(int(ratio_GoodCpds*100)) +'%'

            #Print all statistics in a table
            print_note(f"\n ### Overview for Series: {series}")
            print_cpds_info_table(TotalCpds_SeriesSpecificCount,Training_SpecificSeries_count,SpecificSeries_count, Compounds_BelowThresh_text, Compounds_AboveThresh_text, ratio_GoodCpds_text)

            #Plot to show Experimental values over time should be displayed, irrespective of the size of the test set
            print_note(f"\n --- \n ### Experimental values over time for Series: {series}")
            Exp_Values_Dist(Series_all_df,DesiredProjectThreshold,scale,PlotTitle_series)

            # Model evaluation
            print_note(f"\n --- \n ### Model evaluation for Series: {series}")



            
            if (SpecificSeries_count > 10):

                PlotTitle_Test = PlotTitle_series + " - Prospective Validation Set"
                print_note(f"\n#### Predicted vs Experimental Values (prospective) for Series: {series}")
                ScatterPlot(Series_df,DesiredProjectThreshold,scale,PlotTitle_Test)
            
                #Call the thresh function to get the threshold ranges for calculating the various metrics
                min_thresh,max_thresh,Thresholds_selection = Thresh_Selection(Series_df.Predicted,DesiredProjectThreshold,scale)

                for thresh in Thresholds_selection[1:]: #Exclude the first threshold, as there wouldn't be many compounds below the minimal threshold - Generated statistics can be misleading
            
                    #Estimate the number of datapoints at or below each threshold
                    NoOfObs =  len(Series_df[Series_df.Predicted<=thresh])
                    Compounds_percent = (NoOfObs/len(Series_df))*100
                
                    if PosClass == '>':
                        class_annotation = 'above'
                        #Mapping to binaries based on different thresholds
                        Series_df['Observed_Binaries'] = Series_df['Observed'].map(lambda x: int(x > thresh))
                        Series_df['Predicted_Binaries'] = Series_df['Predicted'].map(lambda x: int(x > thresh))
                        
                        #Identifying the predicted likelihood to extract good compounds at a selected experimental threshold
                        Observations_Pos_Extract = Series_df[Series_df.Predicted > thresh]
                        if len(Observations_Pos_Extract)>10:
                            Pred_Pos_Likelihood = (len(Observations_Pos_Extract[Observations_Pos_Extract['Observed'] > DesiredProjectThreshold])/len(Observations_Pos_Extract))*100
                        else:
                            Pred_Pos_Likelihood = math.nan
                            
                        #Identifying the likelihood to remove good compounds at a selected experimental threshold
                        Observations_Neg_Extract = Series_df[Series_df.Predicted <= thresh]
                        if len(Observations_Neg_Extract)>10:
                            Pred_Neg_Likelihood = (len(Observations_Neg_Extract[Observations_Neg_Extract['Observed'] > DesiredProjectThreshold])/len(Observations_Neg_Extract))*100
                        else:
                            Pred_Neg_Likelihood = math.nan
                        
                        AllMetrics = pd.concat([pd.DataFrame([[date.today(),thresh,Compounds_percent,Pred_Pos_Likelihood,Pred_Neg_Likelihood]]),Calculate_allMetrics(Series_df)],axis=1)
                        AllMetrics_df = pd.concat([AllMetrics_df,AllMetrics],axis=0)
                
                    else:
                        class_annotation = 'below'
                        Series_df['Observed_Binaries'] = Series_df['Observed'].map(lambda x: int(x <= thresh))
                        Series_df['Predicted_Binaries'] = Series_df['Predicted'].map(lambda x: int(x <= thresh))
                        
                        Observations_Pos_Extract = Series_df[Series_df.Predicted <= thresh]
                        if len(Observations_Pos_Extract)>10:
                            Pred_Pos_Likelihood = (len(Observations_Pos_Extract[Observations_Pos_Extract['Observed'] <= DesiredProjectThreshold])/len(Observations_Pos_Extract))*100
                        else:
                            Pred_Pos_Likelihood = math.nan
                            
                        Observations_Neg_Extract = Series_df[Series_df.Predicted > thresh]
                        if len(Observations_Neg_Extract)>10:
                            Pred_Neg_Likelihood = (len(Observations_Neg_Extract[Observations_Neg_Extract['Observed'] <= DesiredProjectThreshold])/len(Observations_Neg_Extract))*100
                        else:
                            Pred_Neg_Likelihood = math.nan
                        AllMetrics = pd.concat([pd.DataFrame([[date.today(),thresh,Compounds_percent,Pred_Pos_Likelihood,Pred_Neg_Likelihood]]),Calculate_allMetrics(Series_df)],axis=1)
                        AllMetrics_df = pd.concat([AllMetrics_df,AllMetrics],axis=0)

                AllMetrics_df.columns=['Calculation Date','Threshold','CompoundsTested','Pred_Pos_Likelihood','Pred_Neg_Likelihood','PPV','CompoundsDiscarded']
                AllMetrics_df = AllMetrics_df.drop_duplicates() #Duplicates exist, if the threshold chosen by the user is one among the 50 thresholds chosen for analysis; Remove them prior to calling the plot functions
            
                 #Printing statistics at desired project threshold
                Desired_Threshold_df = AllMetrics_df[AllMetrics_df.Threshold==DesiredProjectThreshold]
                
                #Sort the metrics data frame before plotting to have the thresholds in ascending order in x-axis
                AllMetrics_df_sorted = AllMetrics_df.sort_values(by=['Threshold'],ascending = False) 

                #Training set scatter plots
                print_note(f"\n #### Predicted vs Experimental Values (training set) for Series: {series}")

                if (Training_SpecificSeries_count>0):
                    PlotTitle_Train = PlotTitle_series + " - Training Set"
                    ScatterPlot(Training_Series_df,DesiredProjectThreshold,scale,PlotTitle_Train)
                
                else:
                    print(Fore.RED + 'Training set is empty - Not possible to generate scatter plots or compute any metrics!' + Fore.RESET)

                print_note(f"\n#### Model performance over time for Series: {series}")
                print_note(f"\n##### RMSE for Series: {series}")
                ModelStabilityPlot(Series_df,scale,PlotTitle_series)

                #Calculating time dependant MPO scores/ Displaying the different similarity/correlation scores(Actual vs time-weighted)        
                print_note(f"\n##### Similarity of prospective data to training set for Series: {series}")
                discount_factor = 0.9 #A value set arbitrarily - Might have to be optimized based on a few runs for a couple of pilot projects
                time_weighted_score_plot(Series_all_df,discount_factor,PlotTitle_series)

    
                if ((SpecificSeries_count !=0) and (SpecificSeries_count < 20)):
                    print('\n')
                    print(Fore.RED + 'Less than 20 compounds in the prospective validation set for series: '+  series + '! Please treat the statistics with caution.'+Fore.RESET)

                # Model usage advice
                print_note(f"\n --- \n ### Model usage advice for Series: {series}")

                print_note(f"\n##### What predicted threshold gives best enrichment for Series: {series}")
                LikelihoodPlot(AllMetrics_df_sorted.Threshold,AllMetrics_df_sorted['CompoundsTested'],AllMetrics_df_sorted.Pred_Pos_Likelihood,AllMetrics_df_sorted.Pred_Neg_Likelihood,DesiredProjectThreshold,SpecificSeries_count,min_thresh,max_thresh,class_annotation,PosClass,Desired_Threshold_df,scale,PlotTitle_series)
 
                print_note(f"\n#### Explore other experimental thresholds to aim for Series: {series}")

                LinePlot(AllMetrics_df_sorted.Threshold,AllMetrics_df_sorted['CompoundsTested'],AllMetrics_df_sorted.PPV,AllMetrics_df_sorted.CompoundsDiscarded,DesiredProjectThreshold,SpecificSeries_count,min_thresh,max_thresh,class_annotation,Desired_Threshold_df,scale,PlotTitle_series)
                
                
            else:
                print('\n')
                print_note(f"\n #### Predicted vs Experimental Values (training set) for Series: {series}")
                ScatterPlot(Series_df,DesiredProjectThreshold,scale,PlotTitle_series)
                print(Fore.RED+'Less than 10 compounds with measured values in the prospective validation set for series: '+ series + '! Not possible to compute any metrics!'+Fore.RESET)

    return
