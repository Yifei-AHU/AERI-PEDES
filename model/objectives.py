import torch
import torch.nn as nn
import torch.nn.functional as F

def triplet_loss_from_similarity(S_t2v, S_v2t, margin=0.2):
    """
    Args:
        S_t2v: [B, B] 文本到图像相似度矩阵
        S_v2t: [B, B] 图像到文本相似度矩阵
        margin: 间隔超参数
    Returns:
        scalar: 双向 triplet loss
    """
    B = S_t2v.size(0)
    device = S_t2v.device

    # --- 文本到图像方向 ---
    pos_t2v = torch.diag(S_t2v).view(B, 1)  # [B,1]
    # 对每个文本 i，负样本为所有图像 j != i
    cost_t2v = torch.clamp(margin + S_t2v - pos_t2v, min=0)
    mask = torch.eye(B, dtype=torch.bool, device=device)
    cost_t2v = cost_t2v.masked_fill(mask, 0)
    loss_t2v = cost_t2v.sum() / (B * (B - 1))

    # --- 图像到文本方向 ---
    pos_v2t = torch.diag(S_v2t).view(1, B)  # [1,B]
    cost_v2t = torch.clamp(margin + S_v2t - pos_v2t, min=0)
    cost_v2t = cost_v2t.masked_fill(mask, 0)
    loss_v2t = cost_v2t.sum() / (B * (B - 1))

    # 双向平均
    loss = (loss_t2v + loss_v2t) / 2
    return loss

def triplet_loss_hard_from_similarity(S_t2v, S_v2t, margin=0.2):
    B = S_t2v.size(0)
    device = S_t2v.device
    mask = ~torch.eye(B, dtype=torch.bool, device=device)

    # 文本→图像
    pos_t2v = torch.diag(S_t2v)
    neg_t2v = S_t2v[mask].view(B, B - 1).max(dim=1)[0]
    loss_t2v = F.relu(margin + neg_t2v - pos_t2v).mean()

    # 图像→文本
    pos_v2t = torch.diag(S_v2t)
    neg_v2t = S_v2t[mask].view(B, B - 1).max(dim=1)[0]
    loss_v2t = F.relu(margin + neg_v2t - pos_v2t).mean()

    return (loss_t2v + loss_v2t) / 2

def find_mutual_nearest_neighbors(image_features, text_features, k):
    # 归一化特征向量
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)

    # 计算余弦相似度
    image_text_similarities = torch.matmul(image_features, text_features.t())
    text_image_similarities = torch.matmul(text_features, image_features.t())

    # 找到图片特征的 k 个文本近邻
    _, image_to_text_nearest_neighbors = torch.topk(image_text_similarities, k, dim=1)
    _, text_to_image_nearest_neighbors = torch.topk(text_image_similarities, k, dim=1)

    mutual_nearest_neighbors = torch.zeros([image_features.size(0),text_features.size(0)])
    for i in range(image_features.size(0)):
        image_k_nearest = image_to_text_nearest_neighbors[i]
        text_k_nearest = text_to_image_nearest_neighbors[image_k_nearest]
        
        # 检查是否当前图片是文本的 k 近邻之一
        has_mutual = torch.where(text_k_nearest == i, 1, 0).sum(dim=1)
        mutual_text_index = has_mutual.nonzero()
        if len(mutual_text_index) !=0:
            for idx in mutual_text_index:
                mutual_nearest_neighbors[i,image_k_nearest[idx]] = 1


    return mutual_nearest_neighbors

def compute_part(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-6):
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).to(torch.int)

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    # ['trousers','genders','belongings','shoes','top','hairstyle']
    # text_proj_text =  F.softmax(logit_scale * text_norm @ text_norm.t(),dim=1)
    # labels = torch.where(text_proj_text>0.4,1,0) | labels
    k = 5 if text_fetures.size(0) > 5 else text_fetures.size(0)
    t2i_labels = find_mutual_nearest_neighbors(text_fetures,image_fetures,k=k).to(text_fetures.device).to(torch.int)
    i2t_labels = find_mutual_nearest_neighbors(image_fetures,text_fetures,k=k).to(text_fetures.device).to(torch.int)
    
    labels = (i2t_labels * t2i_labels) | labels
    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_patch(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size,length,num_dim = image_fetures.size() #32,192,512
    patch_feats = image_fetures.reshape(batch_size*length,num_dim)
    patch_feats = patch_feats / patch_feats.norm(dim=1, keepdim=True)
    text_fetures = text_fetures/ text_fetures.norm(dim=1, keepdim=True)
    sim_i2t = patch_feats @ text_fetures.t()
    sim_t2i = text_fetures @ patch_feats.t()

    label_i2t = sim_i2t.argmax(dim=-1)
    label_t2i = sim_t2i.argmax(dim=-1)
    loss = F.cross_entropy(logit_scale * sim_i2t, label_i2t) + F.cross_entropy(logit_scale * sim_t2i, label_t2i)
    
    return loss

def compute_sdm_per_sample(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching (per-sample loss)
    
    Returns:
        loss_per_sample: Tensor of shape [B], each element is the loss for that sample
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # [B,1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()  # [B,B]

    if image_id is not None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask

    # 归一化
    image_norm = F.normalize(image_fetures, dim=1)
    text_norm = F.normalize(text_fetures, dim=1)

    # cosine 相似度
    t2i_cosine_theta = text_norm @ image_norm.t()  # [B,B]
    i2t_cosine_theta = t2i_cosine_theta.t()      # [B,B]

    # 放大 logits
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # 归一化标签分布
    labels_distribute = labels / (labels.sum(dim=1, keepdim=True) + epsilon)  # [B,B]

    # i2t 损失
    i2t_pred = F.softmax(image_proj_text, dim=1)  # [B,B]
    i2t_loss_matrix = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    i2t_loss_per_sample = i2t_loss_matrix.sum(dim=1)  # [B]

    # t2i 损失
    t2i_pred = F.softmax(text_proj_image, dim=1)  # [B,B]
    t2i_loss_matrix = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_loss_per_sample = t2i_loss_matrix.sum(dim=1)  # [B]

    # 每个样本总损失
    loss_per_sample = i2t_loss_per_sample + t2i_loss_per_sample  # [B]

    return loss_per_sample

def compute_selective_align_loss(aerial_fetures, ground_features, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Selective alignment loss for text-aerial-bridge retrieval.

    Args:
        aerial_features: Tensor [B, D], 航拍特征
        ground_features: Tensor [B, D], 地面特征
        text_features:   Tensor [B, D], 文本特征
        pid:             Tensor [B], identity labels
        logit_scale:     float, 对齐相似度缩放
        threshold:       float, 文本-航空相似度高于阈值则直接对齐
    Returns:
        loss: scalar
    """

    aerial_fetures = F.normalize(aerial_fetures, dim=-1)
    ground_features = F.normalize(ground_features, dim=-1)
    text_fetures = F.normalize(text_fetures, dim=-1)

    sim_ta = torch.sum(text_fetures * aerial_fetures, dim=-1)  # [B]
    sim_tg = torch.sum(text_fetures * ground_features, dim=-1)  # [B]

    loss_direct_sdm = compute_sdm_per_sample(aerial_fetures, text_fetures, pid, logit_scale)
    loss_bridge_sdm = compute_sdm_per_sample(ground_features, text_fetures, pid, logit_scale) + compute_sdm_per_sample(aerial_fetures, ground_features.detach(), pid, logit_scale)

    k = 1
    delta = sim_ta - sim_tg
    alpha = torch.sigmoid(k*delta)

    loss = torch.mean(alpha * loss_direct_sdm + (1 - alpha) * loss_bridge_sdm)

    return loss

def compute_fa_loss(S_t2v, S_v2t, pid, logit_scale,epsilon=1e-8):
    batch_size = S_t2v.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    text_proj_image = logit_scale * S_t2v 
    image_proj_text = logit_scale * S_v2t 

    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def entropy_loss(mu, eps=1e-8):
    """
    熵约束损失（Entropy Regularization Loss）
    mu: 隶属度张量, shape = [B, N] (每个样本的N个token隶属度)
    """
    mu = F.softmax(mu, dim=-1)  # 确保归一化
    entropy = -torch.sum(mu * torch.log(mu + eps), dim=-1)
    loss = -torch.mean(entropy)  # 最大化熵 => 取负号作为loss
    return loss

def compute_fa_infonce_loss(S_t2v, S_v2t, logit_scale=0.07):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = S_t2v.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(S_t2v.device)

    logit_scale = 1 / logit_scale

    # cosine similarity as logits
    logits_per_image = logit_scale * S_v2t
    logits_per_text = logit_scale * S_t2v

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

