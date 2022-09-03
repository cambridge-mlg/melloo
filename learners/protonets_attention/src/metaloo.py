import utils
import numpy as np
import torch
            
def select_multinomial(number, weights, class_labels, spread_constraint):
    if spread_constraint == "by_class":
        classes = torch.unique(class_labels)
        candidate_indices = torch.zeros((len(classes), number), dtype=torch.long)
        for c in classes:
            c_indices = utils.extract_class_indices(class_labels, c)
            sub_weights = weights[c_indices]
            sub_weights_sm = torch.nn.functional.softmax(sub_weights, dim=0)
            class_candidate_indices = c_indices[torch.multinomial(sub_weights_sm, number, replacement=False)]
            candidate_indices[c] = class_candidate_indices
        return candidate_indices.flatten()
    elif spread_constraint == "none":
        weights_sm = torch.nn.functional.softmax(weights, dim=0)
        return torch.multinomial(weights_sm, number, replacement=False)
        
def select_divine(number, weights, embeddings, class_labels, spread_constraint, gamma):
    if spread_constraint == "by_class":
        classes = torch.unique(class_labels)
        candidate_indices = torch.zeros((len(classes), number), dtype=torch.long)
        for c in classes:
            c_indices = utils.extract_class_indices(class_labels, c)
            indices, div_terms, weight_terms = select_top_sum_redun_k(number, weights[c_indices], embeddings[c_indices], spread_constraint)
            class_candidate_indices = c_indices[indices]
            candidate_indices[c] = class_candidate_indices
        return candidate_indices.flatten()
    elif spread_constraint == "none":
        indices, div_terms, weight_terms =  select_top_sum_redun_k(number, weights, embeddings, spread_constraint)
        return indices
    
def select_top_sum_redun_k(k, weights, embeddings, gamma=25, plus_div=False):
    """
    Returns indices of k diverse points with highest importance; based on facility location
    """
    top_ind = torch.argsort(weights, descending=True)[0]
    selected_influence = weights[top_ind]
    selected_data = [embeddings[top_ind]]
    selected_indices = [top_ind.item()]

    kappa = np.sum(rbf_kernel(embeddings.cpu()).sum(axis=1))
    n = np.shape(weights)[0]
    weights = np.array(weights)
    diversity_terms, weight_terms = [], []
    while len(selected_indices) < k:
        candidates = np.setdiff1d(range(n), selected_indices)
        selected_data_cpu = torch.stack(selected_data).cpu()
        curr = np.sum(rbf_kernel(selected_data_cpu).sum(axis=1))
        diversity_term = rbf_kernel(embeddings.cpu(), selected_data_cpu).sum(axis=1)[candidates]
        diversity_term = (kappa - (curr+diversity_term))
        weights_term = selected_influence + weights[candidates]
        diversity_terms.append((diversity_term.mean(), diversity_term.max(), diversity_term.min()))
        weight_terms.append((weights_term.mean(), weights_term.max(), weights_term.min()))

        if plus_div:
            dist_term = pairwise_distances(embeddings.cpu(),selected_data_cpu).sum(axis=1)[candidates]
            objective_term = weights_term + gamma*diversity_term + dist_term
        else:
            objective_term = weights_term + gamma*diversity_term
        
        cand = candidates[np.argmax(objective_term)]
        selected_influence += weights[cand]
        selected_data += [embeddings[cand]]
        selected_indices += [cand]
    return selected_indices, weight_terms, diversity_terms

def select_by_dropping(weights, drop_rate=None, number=None):
    if (drop_rate != None and number != None):
        print("When selecting by dropping, only number to drop or drop rate can be specified, not both")
        return None
    if (drop_rate == None and number == None):
        print("When selecting by dropping, either number to drop or drop rate must be specified")
        return None
    if number != None:
        num_to_keep = len(weights) - number
    else:
        num_to_keep = int(len(weights) * (1 - drop_rate))
    ranking = torch.argsort(weights, descending=True)
    return ranking[0:num_to_keep], ranking[num_to_keep:]

def select_top_k(number, weights, spread_constraint, class_labels=None):
    if spread_constraint == "by_class":
        classes = torch.unique(class_labels)
        candidate_indices = torch.zeros((len(classes), number), dtype=torch.long)
        for c in classes:
            c_indices = utils.extract_class_indices(class_labels, c)
            sub_weights = weights[c_indices]
            sub_ranking = torch.argsort(sub_weights, descending=True)
            class_candidate_indices = c_indices[sub_ranking[0:number]]
            candidate_indices[c] = class_candidate_indices
        return candidate_indices.flatten()
    elif spread_constraint == "none":
        ranking = torch.argsort(weights, descending=True)
        return ranking[0:number]
        
        
def calculate_representer_values(model, loss, context_images, context_labels, context_features, target_labels, target_features,     
        params):
    l2_regularize_classifier = params["l2_regularize_classifier"]
    l2_lambda = params["l2_lambda"]
    kernel_agg = params["kernel_agg"]
    
    # Do the forward pass with the context set for the representer points:
    context_logits = model(context_images, context_labels, context_images, context_labels, MetaLearningState.META_TEST)
    task_loss = loss(context_logits, context_labels)
    regularization_term = (model.feature_adaptation_network.regularization_term())
    regularizer_scaling = 0.001
    task_loss += regularizer_scaling * regularization_term

    if l2_regularize_classifier:
        classifier_regularization_term = model.classifer_regularization_term()
        task_loss += l2_lambda * classifier_regularization_term
        
    # Representer values calculation    
    dl_dphi = torch.autograd.grad(task_loss, context_logits, retain_graph=True)[0]
    del context_logits
    gc.collect()
    alphas = dl_dphi/(-2.0 * l2_lambda * float(len(context_labels)))
    alphas_t = alphas.transpose(1,0)
    feature_prod = torch.matmul(context_features, target_features.transpose(1,0)).to(context_images.device)
    
    kernels_agg = []
    for k in range(alphas.shape[1]):
        alphas_k = alphas_t[k]
        tmp_mat = alphas_k.unsqueeze(1).expand(len(context_labels), len(target_labels))
        kernels_k = tmp_mat * feature_prod
        kernels_agg.append(kernels_k.unsqueeze(0))
    representers = torch.cat(kernels_agg, dim=0)
    

    if kernel_agg == 'sum':
        representers_agg = representers.sum(dim=0).transpose(1,0).cpu()
    
    elif kernel_agg == 'sum_abs':
        representers_agg = representers.abs().sum(dim=0).transpose(1,0).cpu()
    
    elif kernel_agg == 'class':
        representer_tmp = []
        for c in torch.unique(context_labels):
            c_indices = utils.extract_class_indices(context_labels, c)
            for i in c_indices:    
                representer_tmp.append(representers[c][i])
        representers_agg = torch.stack(representer_tmp).transpose(1,0).cpu()
        
    else:
        print("Unsupported kernel aggregation method specified")
        return None
    return representers_agg

def alternative_loo(model, loss, context_images, context_labels, target_images, target_labels, way):

    target_classes = target_labels.unique()
    weights = torch.zeros(len(context_labels))

    full_logits = model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
    full_loss = loss(full_logits, target_labels)

    with torch.no_grad():
        for i in range(context_images.shape[0]):
            if i == 0:
                context_images_loo = context_images[i + 1:]
                context_labels_loo = context_labels[i + 1:]
            else:
                context_images_loo = torch.cat((context_images[0:i], context_images[i + 1:]), 0)
                context_labels_loo = torch.cat((context_labels[0:i], context_labels[i + 1:]), 0)

            if len(context_labels_loo.unique()) < way:
                weights[i] = 10*loss
                continue

            logits = model(context_images_loo, context_labels_loo, target_images, target_labels, MetaLearningState.META_TEST)
            loss = loss(logits, target_labels)
            weights[i] = loss
            
    return weights
    
def remove_unrepresented_points(candidate_labels, target_images, target_labels):
    classes = torch.unique(target_labels).cpu().numpy()
    unrepresented = (set(classes)).difference(set(candidate_labels.cpu().numpy()))
    num_represented_classes = len(classes) - len(unrepresented)
    reduced_indices = []
    for c in classes:
        if c in unrepresented:
            continue
        else:
            reduced_indices.append(utils.extract_class_indices(target_labels, c))
    reduced_indices = torch.stack(reduced_indices).flatten()
    reduced_target_images = target_images[reduced_indices]
    reduced_labels = (target_labels[reduced_indices]).clone()
    reduced_candidate_labels = candidate_labels.clone()
    unrepresented_classes = list(unrepresented)
    unrepresented_classes.sort(reverse=True)
    for unc in unrepresented_classes:
        reduced_labels[reduced_labels > unc] = reduced_labels[reduced_labels > unc] - 1
        reduced_candidate_labels[reduced_candidate_labels > unc] = reduced_candidate_labels[reduced_candidate_labels > unc] - 1
    return reduced_candidate_labels, reduced_target_images, reduced_labels  

def calculate_loo(model, loss,  context_images, context_labels, target_images, target_labels):
    target_classes = target_labels.unique()
    loss_per_qp = torch.zeros((len(target_labels), len(context_labels)))
    with torch.no_grad():
        for i in range(context_images.shape[0]):
            if i == 0:
                context_images_loo = context_images[i + 1:]
                context_labels_loo = context_labels[i + 1:]
            else:
                context_images_loo = torch.cat((context_images[0:i], context_images[i + 1:]), 0)
                context_labels_loo = torch.cat((context_labels[0:i], context_labels[i + 1:]), 0)
                
            # Handle the case where we're dropping the only instance of a class
            # These labels are all re-normalized
            reduced_context_labels, reduced_target_images, reduced_target_labels = remove_unrepresented_points(context_labels_loo, target_images, target_labels)
            represented_classes = []
            for cc in target_classes:
                if cc in context_labels_loo:
                    represented_classes.append(cc.item())

            # Calculate accuracy on task using only selected candidates as context points
            reduced_target_logits = model(context_images_loo, reduced_context_labels, reduced_target_images, reduced_target_labels, MetaLearningState.META_TEST)

            #logits = model(context_images_loo, context_labels_loo, target_images, target_labels, MetaLearningState.META_TEST)
            rt = 0
            for t in range(len(target_labels)):
                # If this target label wasn't removed 
                if target_labels[t].item() in represented_classes:
                    loo_loss =  loss(reduced_target_logits[rt].unsqueeze(0), reduced_target_labels[rt].unsqueeze(0))
                    loss_per_qp[t][i] = loo_loss
                    rt += 1
                else:
                    # Add the things that were incorrectly classified by default, because they weren't represented in the loo context set
                    equal_chance_logits = torch.tensor([1.0/float(len(target_classes))]).repeat(len(target_classes)).to(context_images.device)
                    loo_loss = loss(equal_chance_logits.unsqueeze(0), target_labels[t].unsqueeze(0))
                    loss_per_qp[t][i] = loo_loss
                    
            
    #Somehow turn the loss into a weight:
    #weight_per_qp = loss_per_qp/loss_per_qp.max() - 1
    return loss_per_qp 
    
    
    
def add_image_rankings(image_ranking_dict, image_key, image, ranking_key, ranking):
    if image_key not in image_ranking_dict.keys():
       image_ranking_dict[image_key] = {}
    
    if ranking_key not in image_ranking_dict[image_key].keys():
        image_ranking_dict[image_key][ranking_key] = np.array(ranking)
    else:
        image_ranking_dict[image_key][ranking_key] = np.append(image_ranking_dict[image_key][ranking_key], ranking)
    return image_ranking_dict
    
def weights_from_multirankings(image_ranking_dict, ranking_key):
    agg_ranking = torch.zeros(len(image_ranking_dict.keys()))
    image_ids = np.zeros(len(agg_ranking), dtype=int)
    # For each image
    for i, img_id in enumerate(image_ranking_dict.keys()):
        agg_ranking[i] = image_ranking_dict[img_id][ranking_key].sum().item()
        image_ids[i] = img_id
    # Return parallel arrays of the ranking and the image id
    return agg_ranking, image_ids