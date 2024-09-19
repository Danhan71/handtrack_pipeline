assert False, "all moved to motifs"

# """ 
# 7/7/22 - for extracting sequences from (e.g) n prims in grid, based on things like:
# - diff prims, but same sequence and location.
# Assumes clean tasks (prims in grid)

# Works using datsegs (either aligned to beh or task strokes)

# See notebook: analy_spatial_prims_motifs_220707

# Is like "motifs in a bag"...
# """

# VARIABLES_KEEP = ["shape_oriented", "gridloc"]

# def generate_dict_of_all_used_motifs(D, nprims=2, 
#     variables_keep = None,
#     WHICH_DATSEGS = "beh_firsttouch", shapes_to_ignore=None):
#     """ Generate dict holding all motifs used in this dataset.
#     PARAMS:
#     - D, Dataset
#     - nprims, number of prims in motif. Currently only coded for 2
#     - variables_keep, list of str keys into datsegs. order of this list
#     defines the feature tuple for each token. e..g, ["shape_oriented", "gridloc"]
#     - WHICH_DATSEGS, str, {'beh', 'beh_firsttouch'}, defines which datsegs to use, ether
#     aligned to beh or beh_firsttouch.
#     - shapes_to_ignore, list of str of shapes_oriented, ignores any motifs that inlcude this
#     shape. None to not apply this
#     RETURNS:
#     - motifs_all_dict, dict, where keys are motifs (tuples of tokens) and itmes are 
#     list of indices, where and index is (trial, starting strokenum)
#     """

#     assert WHICH_DATSEGS in {"beh", "beh_firsttouch"}, "these are the only ones that make sens..."
#     # assert WHICH_DATSEGS == "beh", "this is the only one that maintains mappign between stroke indices across all dat. if use beh_firsttouch, plots will be weird."

#     if variables_keep is None:
#         variables_keep = VARIABLES_KEEP
#     motifs_all_dict = {}
#     def prune_token_variables(token, variables_keep):
#         """
#         keep only the variables you care about.
#         token = datsegs[i]
#         RETURNS:
#         tuple of items, in order of variables_keep
#         """
#     #     token = {k:v for k, v in token.items() if k in variables_keep}
#         token_tuple = tuple([token[k] for k in variables_keep])
#         return token_tuple

#     for ind in range(len(D.Dat)):
#         datsegs = D.taskclass_tokens_extract_wrapper(ind, which_order=WHICH_DATSEGS)
#         # Beh = D.Dat.iloc[ind]["BehClass"]
#         # primlist, datsegs_behlength, datsegs_tasklength = D.behclass_extract_beh_and_task(ind)[:3]
#         # if WHICH_DATSEGS=="task":
#         #     # matching beh strokes first touch
#         #     datsegs = datsegs_tasklength
#         # elif WHICH_DATSEGS=="beh":
#         #     # matching beh strokes,
#         #     datsegs = datsegs_behlength
#         # else:
#         #     assert False
            
#         # print(datsegs)
#         # print(datsegs[0]["Prim"].Stroke())
#         # assert False

#         for i in range(len(datsegs)-(nprims-1)):
#             motif = datsegs[i:i+nprims]
                
#     #         # MOtifs that should obviously not keep
#     #         if motif[0][1]
            
#             # save this motif
#             # use the motif as a key (hash it)
#             # just keep the variables that you care about
#             motif_features = tuple([prune_token_variables(token, variables_keep) for token in motif])
            
#             # This is hashable. use as key in dict
#             index = (motif, (ind, i))
#             if motif_features in motifs_all_dict.keys():
#                 motifs_all_dict[motif_features].append(index)
#             else:
#                 motifs_all_dict[motif_features] = [index]

#             # # Save its tokens and strokes.
#             # motif_tokens = motif

#     # prune motifs
#     if shapes_to_ignore is not None:
#         # First, check you didnt make mistake netering shapes
#         list_shapes = extract_list_shapes_loc(motifs_all_dict)[0]
#         for sh in shapes_to_ignore:
#             if sh not in list_shapes:
#                 print(sh)
#                 print(list_shapes)
#                 assert False

#         # Second, clean up the motifs
#         def _is_bad(motif):
#             for token in motif_features:
#                 if token[0] in shapes_to_ignore:
#                     return True
#             return False

#         motifs_all_dict_new = {}
#         for motif_features, v in motifs_all_dict.items():
#             if _is_bad(motif_features):
#                 print("removing motif: ", motif_features)
#                 continue
#             else:
#                 motifs_all_dict_new[motif_features] = v
#         motifs_all_dict = motifs_all_dict_new

#     print("Found this many motifs: ", len(motifs_all_dict))
#     sorted(list(motifs_all_dict.keys()))
#     print("This many instances per motif_features: ", [len(v) for k, v in motifs_all_dict.items()])
    
#     return motifs_all_dict

# def extract_shapes_in_motif(motif_features):
#     """
#     Get list of shapes (unique, sorted) in this motif_features
#     """
#     shapes = []
#     for token in motif_features:
#         # loc = token[1]
#         shape = token[0]
#         shapes.append(shape)
#     return shapes


# def extract_list_shapes_loc(motifs_all_dict):
#     """ get the list of all unqiue shapes and locations
#     """
#     # 1) Get list of all unique shapes and locations in the dataset.
#     list_shapes = []
#     list_locs = []
#     for motif in motifs_all_dict.keys():
#         for token in motif:
#             loc = token[1]
#             shape = token[0]
#             list_locs.append(loc)
#             list_shapes.append(shape)
#     list_locs = sorted(list(set(list_locs)))
#     list_shapes = sorted(list(set(list_shapes)))
#     return list_shapes, list_locs

# def generate_motifgroup_data(motifs_all_dict, 
#         list_group_kind= None):
#     """ Main extraction of computed data. For each way of grouping (e.g., same prim, same seq, diff location),
#     pull out each "motifgroup" (which is a set of all motifs which are related under this grouping), and
#     for each motifgroup save the trials that have these motifs. 
#     RETURNS:
#     - DatGroups, dict where keys are each kind of grouping (e.g, diff_sequence), and items 
#     are lists of dicts, where each dict is a motifgroup and its data.
#     """
#     if list_group_kind is None:
#         list_group_kind = ["same", "diff_sequence", "diff_location", "diff_prims"]

#     list_shapes, list_locs = extract_list_shapes_loc(motifs_all_dict)
    
#     # 2) Extract data
#     DatGroups = {}
#     for group_kind in list_group_kind:

#         def get_all_possible_motifs_for_this_motifs_group(motif):
#             """
#             RETURNS:
#             - motif_group, tuple of tples, sorted, where one of the
#             tuples if motif, and the rest are the other motifs in this 
#             group for this motif. This works becauseof symmatry -i.e,
#             each motif is only in one possible group
#             """

#             list_motif = []
#             if group_kind=="diff_sequence":
#                 # Get motif and its reverse
#                 list_motif = [motif, motif[::-1]]
#             elif group_kind=="diff_location":
#                 # get this sequence of prims, at all possibel locations
#                 shape1 = motif[0][0]
#                 shape2 = motif[1][0]
#                 for i in range(len(list_locs)):
#                     for ii in range(len(list_locs)):
#                         if i!=ii: # different locations
#                             # This is a unique motif
#                             loc1 = list_locs[i]
#                             loc2 = list_locs[ii]
#                             motif = ((shape1, loc1), (shape2, loc2))
#                             list_motif.append(motif)
#             elif group_kind=="diff_prims":
#                 # Get this sequence of locations, all possible prim location pairs.
#                 loc1 = motif[0][1]
#                 loc2 = motif[1][1]
#                 for shape1 in list_shapes:
#                     for shape2 in list_shapes:
#                         # This is a unique motif
#                         motif = ((shape1, loc1), (shape2, loc2))
#                         list_motif.append(motif)
#             elif group_kind=="same":
#                 # simply: motifs should be identical
#                 list_motif = [motif]
#             else:
#                 assert False

#             # assert len(list_motif)>1, "even if you want same motif, just include it twice."
#             # Convert to sorted tuple
#             list_motif = sorted(list_motif)
#             motif_group = tuple(list_motif)
#             return motif_group

#         def count_num_instances_this_motif(motif):
#             # How often did this motif occur in dataset?
#             if motif in motifs_all_dict.keys():
#                 return len(motifs_all_dict[motif])
#             else:
#                 return 0

#         # Got thru all motifs. check which group it is in
#         dict_motifgroups_all = {}
#         list_motifs_iter = list(motifs_all_dict.keys())
#         for motif in list_motifs_iter:

#             motif_group = get_all_possible_motifs_for_this_motifs_group(motif)

#             if motif_group in dict_motifgroups_all.keys():
#                 # Then skip, since this already done
#                 continue
#             else:
#                 # if group_kind=="same":
#                 #     print(motif_group)
#                 #     assert False
#                 # Then go thru each motif in this group, and see how many instances there are
#                 list_n = [count_num_instances_this_motif(motif_this) for motif_this in motif_group]    
#                 # if motif_group==((('V-4-0', (1, 1)), ('Lcentered-3-0', (0, 0))), ):
#                 #     print(list_n)
#                 #     assert False

#                 dict_motifgroups_all[motif_group] = list_n

        
#         #  Convert to a list of dicts
#         dat = []
#         for k, v in dict_motifgroups_all.items():
#             n_unique_motifs_with_at_least_one_trial = sum([vv>0 for vv in v])
#             n_trials_per_motif = v
#             n_trials_summed_across_motifs = sum(v)
#             dat.append({
#                 "n_unique_motifs_with_at_least_one_trial":n_unique_motifs_with_at_least_one_trial,
#                 "key":k, # motifgroup
#                 "n_trials_per_motif":n_trials_per_motif,
#                 "n_trials_summed_across_motifs":n_trials_summed_across_motifs
#             })
#             # print({
#             #     "n_unique_motifs_with_at_least_one_trial":n_unique_motifs_with_at_least_one_trial,
#             #     "key":k, # motifgroup
#             #     "n_trials_per_motif":n_trials_per_motif,
#             #     "n_trials_summed_across_motifs":n_trials_summed_across_motifs
#             # })
#             # assert False

#         del dict_motifgroups_all # inefficient code, went thru this intermediate
            
#         # sort motifgroups in order from most to least cases 
#         dat = sorted(dat, key=lambda x:(x["n_unique_motifs_with_at_least_one_trial"], x["n_trials_summed_across_motifs"]), reverse=True)
#         DatGroups[group_kind] = dat

#     # Prune, only keep motifgroups that have enough trials to actually use these for an 
#     # experiment.
#     for which_group in list_group_kind:
#         # 1) only keep cases with at least two unique motifs.
#         if which_group!="same":
#             list_dat = []
#             for Dat in DatGroups[which_group]:
#                 if Dat["n_unique_motifs_with_at_least_one_trial"]>1:
#                     list_dat.append(Dat)
#             DatGroups[which_group] = list_dat
        
#         # 2) Only keep cases with >1 trial
#         list_dat = []
#         for Dat in DatGroups[which_group]:
#             if Dat["n_trials_summed_across_motifs"]>1:
#                 list_dat.append(Dat)
#         DatGroups[which_group] = list_dat



#     return DatGroups, list_group_kind
        

# def extract_motifgroup_dict(DatGroups, which_group, motifgroup):
#     """ Help to extract a single data dict, for this motifgroup
#     - DatGroups, dict, where each key is a way of grouping (e..g, "diff_sequence") and
#     each item is DAT the data (list of dicts, each dict corresponding to one group of 
#     motifs (motifgroup))
#     - which_group, string, indexes DatGroups
#     - motifgroup, either:
#     --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
#     --- tuple (motif), tuple of motifs, where motif is a tuple of tokens, where token is
#     a tuple of features (e..g, (circle, [0,0]) for (shape, location)
#     This is a group that indexes DAT
#     RETURNS:
#     - datdict, a dict for this motifgroup
#     Or: None if cant find it
#     """

#     dat = DatGroups[which_group]
    
#     # Get DAT
#     if isinstance(motifgroup, int):
#         # Then get in order of mostcases --> leastcases
#         # Assumes it is already sorted
#         if motifgroup>=len(dat):
#             return None, None
#         datdict = dat[motifgroup]
#     else:
#         # Then search for this motifgroup
#         datdict = [d for d in dat if d["key"]==motifgroup][0]

#     motifgroup = datdict["key"]
#     return datdict, motifgroup



# def get_all_motifs_found_for_this_motifgroup(DatGroups, which_group, motifgroup):
#     """
#     PARAMS:
#     - DatGroups, dict, where each key is a way of grouping (e..g, "diff_sequence") and
#     each item is DAT the data (list of dicts, each dict corresponding to one group of 
#     motifs)
#     - which_group, string, indexes DatGroups
#     - motifgroup, either:
#     --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
#     --- tuple (motif), see above.tuple, a group that indexes DAT
#     RETURNS:
#     - list_motifs, list of motifs for which >0 cases found, 
#     in this motif group
#     - motifgroup, the tuple holding all possible motifs
#     - list_tokens, list of unique tokens across used motifs
#     - list_shapes, list of unique shapes across used motifs
#     """
    
#     datdict, motifgroup = extract_motifgroup_dict(DatGroups, which_group, motifgroup)

#     if datdict is None:
#         return None, None

#     # Get motifs found in this group
#     list_motifs = []
#     for motif, ncases in zip(datdict["key"], datdict["n_trials_per_motif"]):
#         if ncases>0:
#             list_motifs.append(motif)

#     # tokens and shapes
#     list_tokens = list(set([token for motif in list_motifs for token in motif]))
#     list_shapes = list(set([token[0] for token in list_tokens]))

#     return list_motifs, motifgroup, list_tokens, list_shapes
    
# def get_inds_this_motif(motifs_all_dict, motif, random_subset=None):
#     """ REturn list of indices that have this motif
#     PARAMS:
#     - motifs_all_dict, dict where keys are motifs, each vals are list of index where 
#     this is found in dataset, where index is (trialnum, strokenum start)
#     - random_subset, either None or int, if the latter, then extracts maximum this many trials
#     RETURNS:
#     - list_idxs, each is (trial, strokenum)
#     """
#     assert motif in motifs_all_dict.keys(), "asking for motif that was never detected for this dataset..."
#     list_idxs = motifs_all_dict[motif]

#     # Take random subset
#     if random_subset and len(list_idxs)>random_subset:
#         import random
#         list_idxs = random.sample(list_idxs, random_subset)

#     # break out trials and strokenums
#     return list_idxs
    
# def get_inds_this_motifgroup(DatGroups, which_group, motifs_all_dict, motifgroup,
#         ntrials_per_motif=None):
#     """ For this motifgroup, find all indices across all motifs
#     that were found within this group.
#     PARAMS;
#     - DatGroups, see avbove
#     - which_group, str, see above
#     - motifs_all_dict, see above
#     - motifgroup, either:
#     --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
#     --- tuple (motif), see above.
#     - ntrials_per_motif, int, if not None, then only keeps this many trials per motif, i.e., if any
#     motifs have > this many trials, picks random subset
#     RETURNS:
#     - list_trials, list of ints into D.Dat
#     - list_strokenums, list of strokenums taht start each motif instance,
#     paired with list_trials
#     - list_motifs_all, aligned with the above, the motif for each.
#     """
        
#     list_motifs, motifgroup = get_all_motifs_found_for_this_motifgroup(DatGroups, which_group, motifgroup)[:2]
#     # for each motif, extract trials

#     if list_motifs is None:
#         # Then this motifgroup doesnt exist?
#         return None, None, None

#     inds_idxs_all = []
#     inds_strokenums_all =[]
#     list_motifs_all = []
#     for motif in list_motifs:
#         inds_idxs = get_inds_this_motif(motifs_all_dict, motif, 
#             random_subset=ntrials_per_motif)
#         # print(inds_idxs)
#         # print(ntrials_per_motif)
#         # assert False
#         inds_idxs_all.extend(inds_idxs)
#         for _ in inds_idxs:
#             list_motifs_all.append(motif)

#     assert len(inds_idxs_all)==len(list_motifs_all)
#     return inds_idxs_all, list_motifs_all, motifgroup




# ###################### GETTING SETS OF TRIALS, FOR GENERATING EXPT


# def expt_extract_motifgroups_to_regenerate(which_group, DatGroups, motifs_all_dict):
#     """ Determine what are trials to generate for this motifgroup, collected from best to worst, 
#     and using criteria which loosen as you go thru and run out of groups to try.
#     PARAMS:
#     - which_group, string key into DatGroups, will get data only for this group.
#     """
#     import random
#     n_trials_get = 50 # how many unqiue trials to pull out. will stop at or slightly higher than this
#     ntrials_per_motif = 2 # for each motif, how many unique trials to get that have this motif
#     ntrials_max_per_motifgroup = 6 # useful, since diff_prims can have too many motifs per motifgroup...
#     # ntrials_per_motif = 2 
#     # ntrials_max_per_motifgroup = 5 # useful, since diff_prims can have too many motifs per motifgroup...

#     def list_intersect(list1, list2):
#         """ Returns True if any items intersect.
#         otehrwies False
#         """
#         for x in list1:
#             if x in list2:
#                 return True
#         return False

#     ALREADY_USED_MOTIFGROUPS = []
#     ALREADY_USED_SHAPES = []
#     ALREADY_USED_MOTIFGROUPS_INDS = []
#     def get_best_motifgroup(which_group, list_criteria_inorder_THIS):
#         for motifgroup_ind in range(len(DatGroups[which_group])):
#             inds_idxs_all, list_motifs_all, motifgroup = get_inds_this_motifgroup(DatGroups, which_group, 
#                                                                         motifs_all_dict, motifgroup_ind, 
#                                                                       ntrials_per_motif=ntrials_per_motif)
            
#             ### Take subset here before doing tests below.
#             if len(inds_idxs_all)>ntrials_max_per_motifgroup:
#                 # Take a random subset
#                 # NOTE: this must be taken in order, to make sure gets distribution of motifs
#                 assert ntrials_max_per_motifgroup > 2*ntrials_per_motif, "otherwise wil not necessaril;y get multiple motifs"
#                 if False:
#                     n = len(inds_idxs_all)
#                     inds_sub = random.sample(range(n), ntrials_max_per_motifgroup)
#                     inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
#                     list_motifs_all = [list_motifs_all[i] for i in inds_sub]
#                 else:
#                     inds_idxs_all = inds_idxs_all[:ntrials_max_per_motifgroup]
#                     list_motifs_all = list_motifs_all[:ntrials_max_per_motifgroup]
#             # print(list_motifs_all)
#             # assert False


#             #### Check constraints
#             # 1) already gotten this group?
#             if motifgroup in ALREADY_USED_MOTIFGROUPS:
#                 continue
                
#             # 1b) Don't take this if only one trial found. i.e., need at least two to make a set
#             if len(list(set(list_motifs_all)))<2:
#                 continue
                
#             # 1c) Dont take this if the motifs are same for all trials. ingore this if your
#             # goal is _actually_ to get same across trials
#             if which_group!='same':
#                 if len(list(set(list_motifs_all)))==1:
#                     for m in list_motifs_all:
#                         print(m)
#                     print("Confirm that indeed these are identical motifs")
#                     assert False

#             # 2) Other criteria
#             # -- some Criteria that prune the motifs/indices
#             # -- others that fail outright
#             FAIL = False
#             for crit in list_criteria_inorder_THIS:
#                 if crit=="dont_reuse_shape_complete":
#                     # then throw out =motifs if it resuses ANY shape
#                     inds_sub = []
#                     for i, motif in enumerate(list_motifs_all):
#                         shapes_in_motif = extract_shapes_in_motif(motif)
#                         if not list_intersect(shapes_in_motif, ALREADY_USED_SHAPES):
#                             # then keep
#                             inds_sub.append(i)
#                     # print(motifgroup_ind, crit, len(list_motifs_all), inds_sub)
#                     inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
#                     list_motifs_all = [list_motifs_all[i] for i in inds_sub]
#                     # print(len(inds_idxs_all))

#                 elif crit=="dont_reuse_shape":
#                     # Then throw out motif if it uses only shapes that are all arleady used (weaker constraint that above).
#                     inds_sub = []
#                     for i, motif in enumerate(list_motifs_all):
#                         shapes_in_motif = extract_shapes_in_motif(motif)
#                         if all([sh in ALREADY_USED_SHAPES for sh in shapes_in_motif]):
#                             # Then continue, since both shapes are already used.
#                             pass
#                         else:
#                             inds_sub.append(i)
#                     # print(motifgroup_ind, crit, len(list_motifs_all), inds_sub)
#                     inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
#                     list_motifs_all = [list_motifs_all[i] for i in inds_sub]
#                     # print(len(inds_idxs_all))
#                 elif crit=="within_motif_diff_shapes":
#                     # Prune motifs that are (shape1, shape1), eg.., circle to circle.
#                     inds_sub = []
#                     for i, motif in enumerate(list_motifs_all):
#                         shapes_in_motif = extract_shapes_in_motif(motif)
#                         if len(list(set(shapes_in_motif)))==1:
#                             continue
#                         else:
#                             # then keep
#                             inds_sub.append(i)
#                     # print(motifgroup_ind, crit, len(list_motifs_all), inds_sub)

#                     inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
#                     list_motifs_all = [list_motifs_all[i] for i in inds_sub]
#                 elif crit=="mult_trials_exist_for_at_least_two_motifs":
#                     # Then check that at leat 2 of the motif is associated with at least 2 trials
#                     n_good = 0
#                     for motif in set(list_motifs_all): # [motif1,, ..., motif2, ...]
#                         n = len([m for m in list_motifs_all if m==motif]) # num instances (trials) for this motif
#                         if n>1:
#                             n_good+=1
#                     if n_good<2:
#                         FAIL = True
#                 else:
#                     print(crit)
#                     assert False
#             if FAIL:
#                 continue
            
#             # 1b) Don't take this if only one trial found. i.e., need at least two to make a set
#             if len(inds_idxs_all)<2:
#                 continue
                
#             # 1c) Dont take this if the motifs are same for all trials. ingore this if your
#             # goal is _actually_ to get same across trials
#             if which_group!='same':
#                 if len(list(set(list_motifs_all)))==1:
#                     for m in list_motifs_all:
#                         print(m)
#                     print("Confirm that indeed these are identical motifs")
#                     assert False

#             ### Success! Keep this.
#             print("Success! taking motifgroup with index: ", motifgroup_ind)
#             if len(inds_idxs_all)>ntrials_max_per_motifgroup:
#                 # Take a random subset
#                 n = len(inds_idxs_all)
#                 inds_sub = random.sample(range(n), ntrials_max_per_motifgroup)
#                 inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
#                 list_motifs_all = [list_motifs_all[i] for i in inds_sub]
            
#             return inds_idxs_all, list_motifs_all, motifgroup, motifgroup_ind
        
#         # If got here, then failed to find anything
#         return None, None, None, None


#     ################### 
#     # - put most important to the right.
#     # list_criteria_inorder = ["mult_trials_exist_per_motif", "dont_reuse_shape", "within_motif_diff_shapes"] # order: will prune from left to right
#     list_criteria_inorder = ["dont_reuse_shape_complete", "mult_trials_exist_for_at_least_two_motifs", "dont_reuse_shape", "within_motif_diff_shapes"] # order: will prune from left to right
#     # list_criteria_inorder = [ "dont_reuse_shape", "within_motif_diff_shapes"] # order: will prune from left to right

#     DatTrials = {}
#     DatGroupsUsedSimple = {which_group:[]} # keep track of which motifgroups are used

#     for i in range(len(list_criteria_inorder)+1): # +1 so runs last time with no constraints
#         list_criteria_inorder_THIS = list_criteria_inorder[i:]
#         print("Criteria: ", list_criteria_inorder_THIS)
        
#         inds_idxs_all, list_motifs_all, motifgroup, motifgroup_ind = get_best_motifgroup(which_group, 
#             list_criteria_inorder_THIS)
#         while inds_idxs_all is not None:
            
#             ### For each trial, save this iformation
#             inds_trials_all = [x[0] for x in inds_idxs_all]
#             inds_strokenums_all = [x[1] for x in inds_idxs_all]
#             for trial, strokenum, motif in zip(inds_trials_all, inds_strokenums_all, list_motifs_all):
#                 item = (which_group, motifgroup, motif, strokenum)
#                 if trial in DatTrials.keys():
#                     DatTrials[trial].append(item)
#                 else:
#                     DatTrials[trial] = [item]

#             # Track which groups are used
#             ntrials = len(inds_idxs_all)
#             DatGroupsUsedSimple[which_group].append({"motifgroup":motifgroup, "ntrials":ntrials, "inds_idxs_all":inds_idxs_all, "list_motifs_all":list_motifs_all})
            
#             # Track what already used
#             ALREADY_USED_MOTIFGROUPS.append(motifgroup)
#             ALREADY_USED_MOTIFGROUPS_INDS.append(motifgroup_ind)

#             for motif in list_motifs_all:
#                 shapesthis = extract_shapes_in_motif(motif)
#                 ALREADY_USED_SHAPES.extend(shapesthis)

#             ### Try to get another motifgroup
#             inds_idxs_all, list_motifs_all, motifgroup, motifgroup_ind = get_best_motifgroup(which_group, 
#                 list_criteria_inorder_THIS)
#         print("Got this many motifgorups so far: ", len(ALREADY_USED_MOTIFGROUPS))
    
#     return DatTrials, DatGroupsUsedSimple, ALREADY_USED_MOTIFGROUPS_INDS
        
            
# def prune_dataset_unique_shapes_only(D):
#     """ Only keep trials where no shape is repeated within the trial
#     """
#     assert False,'in progress. copied from ntoebook...'
#     indtrial_keep = []
#     print(len(D.Dat))
#     for indtrial in range(len(D.Dat)):
#         shapes = D.taskclass_shapes_extract(indtrial)
#         nshape_uniq = len(set(shapes))
#         n = len(shapes)
#         if n - nshape_uniq > 1:
#             print(indtrial, '-', shapes)
#         else:
#             indtrial_keep.append(indtrial)

#     print(len(indtrial_keep))
# D.subsetDataframe(indtrial_keep)    