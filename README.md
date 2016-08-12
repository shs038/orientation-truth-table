import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from collections import Counter
%matplotlib inline
#read in data
motif_orientation_df = pd.read_csv('/home/jtao/for_shengnan/motif_strand_C57BL6J.tsv', sep='\t')
motif_orientation_df.index = motif_orientation_df['ID'].values
del motif_orientation_df['ID']
#seperate data into veh only, kla only, and unchanged
veh_indices=motif_orientation_df[motif_orientation_df['Factors'].str.contains('atac_veh')].index.values
kla_incices=motif_orientation_df[motif_orientation_df['Factors'].str.contains('atac_kla')].index.values
veh_indices=set(veh_indices)
kla_incices=set(kla_incices)
veh_only=veh_indices-kla_incices
kla_only=kla_incices-veh_indices
unchanged=veh_indices.intersection(kla_incices)
veh_only=np.array(list(veh_only))
kla_only=np.array(list(kla_only))
unchanged=np.array(list(unchanged))
veh_only_df=motif_orientation_df.loc[motif_orientation_df.index.isin(veh_only)]
del veh_only_df['Factors']
del veh_only_df['chr']
kla_only_df=motif_orientation_df.loc[motif_orientation_df.index.isin(kla_only)]
del kla_only_df['Factors']
del kla_only_df['chr']
unchanged_df=motif_orientation_df.loc[motif_orientation_df.index.isin(unchanged)]
del unchanged_df['Factors']
del unchanged_df['chr']
def count_orientation(motif_orientation):
    '''
    input:a pandas dataframe contains motifs orientation data
    output:a 3 rows pandas dataframe contains counts of each orientation of each motifs
    '''
    motifs = motif_orientation.columns.values #save motifs identity 
    #creaty a zeros matrix to sotre future orientation count data
    zero_data = np.zeros((3,motif_orientation.shape[1]),dtype=np.int)
    # conver zeros matrix to zeros dataframe
    count_frame = pd.DataFrame(zero_data, columns=motifs)
    count_frame.index=['+','-','?']
    for i in range (motif_orientation.shape[1]):#loop to count the orientations of all motifs 
        one_motif=motif_orientation.ix[:,i].values#retrieve orientations of one motif
        one_motif=list(one_motif)#convert to list
        c=Counter(one_motif)#count each orientation
        c=dict(c)#convert to dictionary
        c=pd.DataFrame.from_dict(c,orient='index')#convert to dataframe
        count_frame.ix[:,i]=c.ix[:,0]#store orientation count in zeros dataframe 
    return count_frame
#count orientation
veh_orientation=count_orientation(veh_only_df)
kla_orientation=count_orientation(kla_only_df)
unchanged_orientation=count_orientation(unchanged_df)
#transpose the count dataframe 
veh_orientation = veh_orientation.T
kla_orientation = kla_orientation.T
unchanged_orientation = unchanged_orientation.T
veh_orientation=veh_orientation.fillna(value=0)
kla_orientation=kla_orientation.fillna(value=0)
unchanged_orientation=unchanged_orientation.fillna(value=0)
#plot orientation counts
sns.distplot(veh_orientation['+'])
plt.ylabel('Frequency')
plt.xlabel('+')
plt.title('veh + count') 
plt.show()
sns.distplot(veh_orientation['-'])
plt.ylabel('Frequency')
plt.xlabel('-')
plt.title('veh - count') 
plt.show()
sns.distplot(kla_orientation['+'])
plt.ylabel('Frequency')
plt.xlabel('+')
plt.title('kla + count') 
plt.show()
sns.distplot(kla_orientation['-'])
plt.ylabel('Frequency')
plt.xlabel('-')
plt.title('kla - count') 
plt.show()
sns.distplot(unchanged_orientation['+'])
plt.ylabel('Frequency')
plt.xlabel('+')
plt.title('unchanged + count') 
plt.show()
sns.distplot(unchanged_orientation['-'])
plt.ylabel('Frequency')
plt.xlabel('-')
plt.title('unchanged - count') 
#count how many times two motifs that co-occur with each other both have sense orientation 
def count_both_sense(motif_orientation):
    '''
    input:a pandas dataframe contains motifs orientation data
    output:a pandas dataframe contains how many times each pair of motifs both have 
            + orientation.
    '''
    motifs = motif_orientation.columns.values #save motifs identity 
    #creaty a zeros matrix to sotre future orientation count data
    zero_data = np.zeros((motif_orientation.shape[1],motif_orientation.shape[1]),dtype=np.int)
    # conver zeros matrix to zeros dataframe
    count_frame = pd.DataFrame(zero_data, columns=motifs)
    count_frame.index=motifs
    for i in range (motif_orientation.shape[1]-1):
        #find the loci where the motif occur with sense orientation
        logical_col_i=motif_orientation.ix[:,i]=='+' 
        for j in range (i+1,motif_orientation.shape[1]):
            #find the loci where the motif occur with sense orientation
            logical_col_j=motif_orientation.ix[:,j]=='+'
            #find the loci where both of the motifs occur with sense orientation
            logical_input=1*logical_col_i+1*logical_col_j==2
            #count how many times both of the motifs occur with sense orientation
            input=np.sum(1*logical_input)
            count_frame.ix[i,j]=count_frame.ix[i,j]+input
    #reshape dataframe
    Pairs=[]
    Count=[]
    #loop in part of count data that contain meaning counting
    for i in range (count_frame.shape[1]-1):
        for j in range (i+1,count_frame.shape[1]):
            #put motif pair and correlation into the empty list
            motif_pairs=(motifs[i],motifs[j])
            Pairs.append(motif_pairs)
            Count.append(count_frame.ix[i,j])
    #reshape the dataframe
    reshaped_frame = pd.DataFrame({'Count': Count}, index=Pairs)
    reshaped_frame['Orientation']='+/+' #add orientation to the dataframe
    reshaped_frame=reshaped_frame[['Orientation','Count']]# put Orientation in front of Count
    return reshaped_frame
#count how many times two motifs that co-occur with each other both have antisense orientation 
def count_both_antisense(motif_orientation):
    '''
    input:a pandas dataframe contains motifs orientation data
    output:a pandas dataframe contains how many times each pair of motifs both have 
            - orientation.
    '''
    motifs = motif_orientation.columns.values #save motifs identity 
    #create a zeros matrix to sotre future orientation count data
    zero_data = np.zeros((motif_orientation.shape[1],motif_orientation.shape[1]),dtype=np.int)
    # conver zeros matrix to zeros dataframe
    count_frame = pd.DataFrame(zero_data, columns=motifs)
    count_frame.index=motifs
    for i in range (motif_orientation.shape[1]-1):
        logical_col_i=motif_orientation.ix[:,i]=='-'
        for j in range (i+1,motif_orientation.shape[1]):
            logical_col_j=motif_orientation.ix[:,j]=='-'
            logical_input=1*logical_col_i+1*logical_col_j==2
            input=np.sum(1*logical_input)
            count_frame.ix[i,j]=count_frame.ix[i,j]+input
    #reshape dataframe
    Pairs=[]
    Count=[]
    #loop in part of count data that contain meaning counting
    for i in range (count_frame.shape[1]-1):
        for j in range (i+1,count_frame.shape[1]):
            #put motif pair and correlation into the empty list
            motif_pairs=(motifs[i],motifs[j])
            Pairs.append(motif_pairs)
            Count.append(count_frame.ix[i,j])
    #reshape the dataframe
    reshaped_frame = pd.DataFrame({'Count': Count}, index=Pairs)
    reshaped_frame['Orientation']='-/-' #add orientation to the dataframe
    reshaped_frame=reshaped_frame[['Orientation','Count']]# put Orientation in front of Count
    return reshaped_frame
#count how many times two motifs co-occur with each other have sense/antisense orientation 
def count_sense_antisense(motif_orientation):
    '''
    input:a pandas dataframe contains motifs orientation data
    output:a pandas dataframe contains how many times each pair of motifs  have 
            +/- orientation.
    '''
    motifs = motif_orientation.columns.values #save motifs identity 
    #create a zeros matrix to sotre future orientation count data
    zero_data = np.zeros((motif_orientation.shape[1],motif_orientation.shape[1]),dtype=np.int)
    # conver zeros matrix to zeros dataframe
    count_frame = pd.DataFrame(zero_data, columns=motifs)
    count_frame.index=motifs
    for i in range (motif_orientation.shape[1]-1):
        logical_col_i=motif_orientation.ix[:,i]=='+'
        for j in range (i+1,motif_orientation.shape[1]):
            logical_col_j=motif_orientation.ix[:,j]=='-'
            logical_input=1*logical_col_i+1*logical_col_j==2
            input=np.sum(1*logical_input)
            count_frame.ix[i,j]=count_frame.ix[i,j]+input
    #reshape dataframe
    Pairs=[]
    Count=[]
    #loop in part of count data that contain meaning counting
    for i in range (count_frame.shape[1]-1):
        for j in range (i+1,count_frame.shape[1]):
            #put motif pair and correlation into the empty list
            motif_pairs=(motifs[i],motifs[j])
            Pairs.append(motif_pairs)
            Count.append(count_frame.ix[i,j])
    #reshape the dataframe
    reshaped_frame = pd.DataFrame({'Count': Count}, index=Pairs)
    reshaped_frame['Orientation']='+/-' #add orientation to the dataframe
    reshaped_frame=reshaped_frame[['Orientation','Count']]# put Orientation in front of Count
    return reshaped_frame
#count how many times two motifs co-occur with each other have antisense/sense orientation 
def count_antisense_sense(motif_orientation):
    '''
    input:a pandas dataframe contains motifs orientation data
    output:a pandas dataframe contains how many times each pair of motifs  have 
            -/+ orientation.
    '''
    motifs = motif_orientation.columns.values #save motifs identity 
    #creaty a zeros matrix to sotre future orientation count data
    zero_data = np.zeros((motif_orientation.shape[1],motif_orientation.shape[1]),dtype=np.int)
    # conver zeros matrix to zeros dataframe
    count_frame = pd.DataFrame(zero_data, columns=motifs)
    count_frame.index=motifs
    for i in range (motif_orientation.shape[1]-1):
        logical_col_i=motif_orientation.ix[:,i]=='-'
        for j in range (i+1,motif_orientation.shape[1]):
            logical_col_j=motif_orientation.ix[:,j]=='+'
            logical_input=1*logical_col_i+1*logical_col_j==2
            input=np.sum(1*logical_input)
            count_frame.ix[i,j]=count_frame.ix[i,j]+input
    #reshape dataframe
    Pairs=[]
    Count=[]
    #loop in part of count data that contain meaning counting
    for i in range (count_frame.shape[1]-1):
        for j in range (i+1,count_frame.shape[1]):
            #put motif pair and correlation into the empty list
            motif_pairs=(motifs[i],motifs[j])
            Pairs.append(motif_pairs)
            Count.append(count_frame.ix[i,j])
    #reshape the dataframe
    reshaped_frame = pd.DataFrame({'Count': Count}, index=Pairs)
    reshaped_frame['Orientation']='-/+' #add orientation to the dataframe
    reshaped_frame=reshaped_frame[['Orientation','Count']]# put Orientation in front of Count
    return reshaped_frame
#dataframe of veh orientation count
veh_sense_sense=count_both_sense(veh_only_df)
veh_antisense_antisense=count_both_antisense(veh_only_df)
veh_sense_antisense=count_sense_antisense(veh_only_df)
veh_antisense_sense=count_antisense_sense(veh_only_df)
#concatenate data for all four orientation
veh_frames = [veh_sense_sense,veh_antisense_antisense,
          veh_sense_antisense,veh_antisense_sense]
veh_cooccur_orientation = pd.concat(veh_frames)
#dataframe of kla orientation count
kla_sense_sense=count_both_sense(kla_only_df)
kla_antisense_antisense=count_both_antisense(kla_only_df)
kla_sense_antisense=count_sense_antisense(kla_only_df)
kla_antisense_sense=count_antisense_sense(kla_only_df)
#concatenate data for all four orientation
kla_frames = [kla_sense_sense,kla_antisense_antisense,
          kla_sense_antisense,kla_antisense_sense]
kla_cooccur_orientation = pd.concat(kla_frames)
#dataframe of unchanged orientation count
unchanged_sense_sense=count_both_sense(unchanged_df)
unchanged_antisense_antisense=count_both_antisense(unchanged_df)
unchanged_sense_antisense=count_sense_antisense(unchanged_df)
unchanged_antisense_sense=count_antisense_sense(unchanged_df)
#concatenate data for all four orientation
unchanged_frames = [unchanged_sense_sense,unchanged_antisense_antisense,
          unchanged_sense_antisense,unchanged_antisense_sense]
unchanged_cooccur_orientation = pd.concat(unchanged_frames)
#normalize orientation count for veh
veh_df_add=veh_sense_sense['Count'].add(veh_antisense_antisense['Count'], fill_value=0)
veh_df_add=veh_df_add.add(veh_sense_antisense['Count'], fill_value=0)
veh_df_add=veh_df_add.add(veh_antisense_sense['Count'], fill_value=0)
veh_cooccur_orientation['Total']=veh_df_add
veh_cooccur_orientation = veh_cooccur_orientation[veh_cooccur_orientation.Total != 0]
veh_cooccur_orientation['Normalized_Count']=veh_cooccur_orientation['Count']/veh_cooccur_orientation['Total']
#normalize orientation count for kla
kla_df_add=kla_sense_sense['Count'].add(kla_antisense_antisense['Count'], fill_value=0)
kla_df_add=kla_df_add.add(kla_sense_antisense['Count'], fill_value=0)
kla_df_add=kla_df_add.add(kla_antisense_sense['Count'], fill_value=0)
kla_cooccur_orientation['Total']=kla_df_add
kla_cooccur_orientation = kla_cooccur_orientation[kla_cooccur_orientation.Total != 0]
kla_cooccur_orientation['Normalized_Count']=kla_cooccur_orientation['Count']/kla_cooccur_orientation['Total']
#normalize orientation count for unchanged
unchanged_df_add=unchanged_sense_sense['Count'].add(unchanged_antisense_antisense['Count'], fill_value=0)
unchanged_df_add=unchanged_df_add.add(unchanged_sense_antisense['Count'], fill_value=0)
unchanged_df_add=unchanged_df_add.add(unchanged_antisense_sense['Count'], fill_value=0)
unchanged_cooccur_orientation['Total']=unchanged_df_add
unchanged_cooccur_orientation = unchanged_cooccur_orientation[unchanged_cooccur_orientation.Total != 0]
unchanged_cooccur_orientation['Normalized_Count']=unchanged_cooccur_orientation['Count']/unchanged_cooccur_orientation['Total']
#subset of veh_cooccur_orientation for each orientation pair without Normalized_Count=0
veh_cooccur_sense_sense=veh_cooccur_orientation[veh_cooccur_orientation['Orientation']=='+/+']
veh_cooccur_antisense_antisense=veh_cooccur_orientation[veh_cooccur_orientation['Orientation']=='-/-']
veh_cooccur_sense_antisense=veh_cooccur_orientation[veh_cooccur_orientation['Orientation']=='+/-']
veh_cooccur_antisense_sense=veh_cooccur_orientation[veh_cooccur_orientation['Orientation']=='-/+']
#subset of kla_cooccur_orientation for each orientation pair without Normalized_Count=0
kla_cooccur_sense_sense=kla_cooccur_orientation[kla_cooccur_orientation['Orientation']=='+/+']
kla_cooccur_antisense_antisense=kla_cooccur_orientation[kla_cooccur_orientation['Orientation']=='-/-']
kla_cooccur_sense_antisense=kla_cooccur_orientation[kla_cooccur_orientation['Orientation']=='+/-']
kla_cooccur_antisense_sense=kla_cooccur_orientation[kla_cooccur_orientation['Orientation']=='-/+']
#subset of unchanged_cooccur_orientation for each orientation pair without Normalized_Count=0
unchanged_cooccur_sense_sense=unchanged_cooccur_orientation[unchanged_cooccur_orientation['Orientation']=='+/+']
unchanged_cooccur_antisense_antisense=unchanged_cooccur_orientation[unchanged_cooccur_orientation['Orientation']=='-/-']
unchanged_cooccur_sense_antisense=unchanged_cooccur_orientation[unchanged_cooccur_orientation['Orientation']=='+/-']
unchanged_cooccur_antisense_sense=unchanged_cooccur_orientation[unchanged_cooccur_orientation['Orientation']=='-/+']
#plot normalized count of each pair of orientations for veh
sns.distplot(veh_cooccur_sense_sense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('veh ++') 
plt.show()
sns.distplot(veh_cooccur_antisense_antisense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('veh --') 
plt.show()
sns.distplot(veh_cooccur_sense_antisense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('veh +-') 
plt.show()
sns.distplot(veh_cooccur_antisense_sense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('veh -+') 
#plot normalized count of each pair of orientations for kla
sns.distplot(kla_cooccur_sense_sense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('kla ++') 
plt.show()
sns.distplot(kla_cooccur_antisense_antisense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('kla --') 
plt.show()
sns.distplot(kla_cooccur_sense_antisense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('kla +-') 
plt.show()
sns.distplot(kla_cooccur_antisense_sense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('kla -+') 
#plot normalized count of each pair of orientations for unchanged
sns.distplot(unchanged_cooccur_sense_sense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('unchanged ++') 
plt.show()
sns.distplot(unchanged_cooccur_antisense_antisense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('unchanged --') 
plt.show()
sns.distplot(unchanged_cooccur_sense_antisense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('unchanged +-') 
plt.show()
sns.distplot(unchanged_cooccur_antisense_sense['Normalized_Count'])
plt.ylabel('Frequency')
plt.xlabel('Normalized_Count')
plt.title('unchanged -+') 
#veh orientation array for chi square test
motifpair=veh_cooccur_sense_sense.index.values
vss=veh_cooccur_sense_sense.ix[:,1].values
vsa=veh_cooccur_sense_antisense.ix[:,1].values
vas=veh_cooccur_antisense_sense.ix[:,1].values
vaa=veh_cooccur_antisense_antisense.ix[:,1].values
veh_all_orientation=np.array([vss,vsa,vas,vaa])
veh_all_orientation=veh_all_orientation.T
#kla orientation array for chi square test
kss=kla_cooccur_sense_sense.ix[:,1].values
ksa=kla_cooccur_sense_antisense.ix[:,1].values
kas=kla_cooccur_antisense_sense.ix[:,1].values
kaa=kla_cooccur_antisense_antisense.ix[:,1].values
kla_all_orientation=np.array([kss,ksa,kas,kaa])
kla_all_orientation=kla_all_orientation.T
#unchanged orientation array for chi square test
uss=unchanged_cooccur_sense_sense.ix[:,1].values
usa=unchanged_cooccur_sense_antisense.ix[:,1].values
uas=unchanged_cooccur_antisense_sense.ix[:,1].values
uaa=unchanged_cooccur_antisense_antisense.ix[:,1].values
unchanged_all_orientation=np.array([uss,usa,uas,uaa])
unchanged_all_orientation=unchanged_all_orientation.T
# Given a set of loci L where motif J is in a given orientation O, does that subset of L where 
# motf Z is present ahve a bias towards +/-
# Fisher's  exact test 
# Null hypothesis: orientation of motif J and that of motif Z are independent at loci L.
def Chisquare_test(orientation):
    Chi_array=np.array(np.zeros((1,4),dtype=np.int))
    #store
    P_value=[]
    for i in range(len(orientation)):
        Chi_array=orientation[i,:]
        chisq,p=stats.chisquare(Chi_array)
        P_value.append(p)
    P_df=pd.DataFrame({'P_value': P_value}, index=motifpair)
    return P_df
# Chi squiare test of orientation, corrected by times of compare
veh_p=Chisquare_test(veh_all_orientation)*195
kla_p=Chisquare_test(kla_all_orientation)*195
unchanged_p=Chisquare_test(unchanged_all_orientation)*195
# dataframe for p_value of motif pairs orientation 
orientation_p_df=pd.concat([veh_p, kla_p,unchanged_p], axis=1)
orientation_p_df.columns = ['veh', 'kla','unchanged']
#p difference
orientation_p_df['veh-kla']=orientation_p_df['veh']-orientation_p_df['kla']
orientation_p_df['veh-unchanged']=orientation_p_df['veh']-orientation_p_df['unchanged']
orientation_p_df['kla-unchanged']=orientation_p_df['kla']-orientation_p_df['unchanged']
#plot difference
sns.distplot(orientation_p_df['veh-kla'])
plt.ylabel('Frequency')
plt.xlabel('veh-kla')
plt.title('orientation p_difference under veh and kal') 
#Truth table of veh
veh_truth=1*(orientation_p_df['veh']>=0.05)
veh_truth=veh_truth.to_frame(name=None)
#Truth table of kla
kla_truth=1*(orientation_p_df['kla']>=0.05)
kla_truth=kla_truth.to_frame(name=None)
#Truth table of unchanged
unchanged_truth=1*(orientation_p_df['unchanged']>=0.05)
unchanged_truth=unchanged_truth.to_frame(name=None)
#concatenate Truth able
Truth_table_orientation=pd.concat([veh_truth, kla_truth,unchanged_truth], axis=1)
Truth_table_orientation.to_csv('/home/shs038/veh_kla/Truth_table_orientation.tsv', sep='\t')
