# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
   
    c_ctg = CTG_features.drop([extra_feature], axis=1)
    is_num = c_ctg.applymap(lambda x: isinstance(x, (int, float)))
    c_ctg1 = c_ctg.where(is_num, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False)
    #print(c_ctg1)
    c_ctg3=dict()
    for key in c_ctg1.keys():
        c_ctg2=c_ctg1[key].loc[:].dropna().values.astype(float)
        c_ctg3[key]=c_ctg2

    # --------------------------------------------------------------------------
    return c_ctg3

def rand_sampling(x, var_hist):
    if np.isnan(x):
        rand_idx = np.random.choice(len(var_hist))
        x = var_hist[rand_idx][0]
    return x

def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.drop([extra_feature], axis=1)
    is_num = c_ctg.applymap(lambda x: isinstance(x, (int, float)))
    c_ctg1 = c_ctg.where(is_num, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False)
    c_cdf = pd.DataFrame()
    for key in c_ctg1.keys():
        feat=[key]
        c_ctg2=c_ctg1[feat].loc[:].dropna().values.astype(float) #no Nan's
        new = c_ctg1[feat].applymap(lambda x: rand_sampling(x, c_ctg2))
        new1 = pd.DataFrame.from_dict(new)
        c_cdf[feat] = new1  
    return c_cdf

def sum_stat(c_feat):
    stat_df = pd.DataFrame()
    stat_df = stat_df.append(c_feat.min().transpose(), ignore_index = True)
    stat_df = stat_df.append(c_feat.quantile(0.25, axis = 0).transpose(), ignore_index = True)
    stat_df = stat_df.append(c_feat.quantile(0.5, axis = 0).transpose(), ignore_index = True)
    stat_df = stat_df.append(c_feat.quantile(0.75, axis = 0).transpose(), ignore_index = True)
    stat_df = stat_df.append(c_feat.max().transpose(), ignore_index = True)
    stat_df.index = ['min', 'Q1', 'med', 'Q3', 'max']
    d = stat_df.to_dict()

    return d


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for key in c_feat:
        out_h = d_summary[key]['Q3'] + 1.5*(d_summary[key]['Q3']-d_summary[key]['Q1'])
        out_l = d_summary[key]['Q3'] - 1.5*(d_summary[key]['Q3']-d_summary[key]['Q1'])
        c_new = c_feat[key]

        new = (c_feat[key].loc[:]<out_h) & (c_feat[key].loc[:]>out_l)
        c_no_outlier[key] = c_new.where(new, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False)
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)




def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
   
    filt_feature = []
    new = (c_cdf[feature].loc[:]<thresh) 
    filt_feature_1 = c_cdf[feature].where(new, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False)
    filt_feature=filt_feature_1.loc[:].dropna().values.astype(float) #no Nan's
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    if mode == 'standard':
        result = CTG_features.copy()
        for feature_name in CTG_features.columns:
            mean_value = CTG_features[feature_name].mean()
            std_value = np.std(CTG_features[feature_name])
            result[feature_name] = (CTG_features[feature_name] - mean_value) / std_value

    if mode == 'MinMax':
        result = CTG_features.copy()
        for feature_name in CTG_features.columns:
            max_value = CTG_features[feature_name].max()
            min_value = CTG_features[feature_name].min()
            result[feature_name] = (CTG_features[feature_name] - min_value) / (max_value - min_value)


    if mode == 'mean':
        result = CTG_features.copy()
        for feature_name in CTG_features.columns:
            max_value = CTG_features[feature_name].max()
            min_value = CTG_features[feature_name].min()
            mean_value = CTG_features[feature_name].mean()
            result[feature_name] = (CTG_features[feature_name] - mean_value) / (max_value - min_value)


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    #n, bins, patches = ax1.hist(result[x], bins = 50)
    ax1.hist(result[x], bins = 50)
    ax1.title.set_text(x)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    #n, bins, patches = ax2.hist(result[y], bins = 50)
    ax2.hist(result[y], bins = 50)
    ax2.title.set_text(y)
    
    nsd_res = result
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
