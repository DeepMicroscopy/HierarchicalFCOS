from sklearn.neighbors import KDTree
import numpy as np


def calc_center(boxes):
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    return cx,cy

def get_number_of_fps(boxes,scores,annos, radius = 25, det_th = 0.5):
    keep = scores > det_th
    boxes = boxes[keep]

    # calc center of predicted boxes
    cx_pred, cy_pred = calc_center(boxes)
    # calc center of target boxes
    cx_gt,cy_gt = calc_center(annos)

    isDet = np.zeros(boxes.shape[0] + annos.shape[0])
    isDet[0:boxes.shape[0]] = 1 

    if annos.shape[0] > 0 and boxes.shape[0] > 0:
        cx = np.hstack((cx_pred, cx_gt))
        cy = np.hstack((cy_pred, cy_gt))
        # set up kdtree 
        X = np.dstack((cx, cy))[0]
        # build a KDTree with annotations
        tree = KDTree(X[isDet == 0])

        # query tree with predictions, ind contains the index of the annotation which matches the prediction
        ind,dist = tree.query_radius(X[isDet == 1], r=radius, return_distance = True, sort_results = True)

        # delete doubeling neigbours
        ind = np.array([i[0] if len(i) > 0 else -1 for i in ind])
        return np.sum(ind==-1)
    
    else:
        return len(boxes)
    

def match_scores_targets(boxes,scores,reg_scores,annos,reg_targets, radius = 25, det_th = 0.5):
    keep = scores > det_th
    boxes = boxes[keep]
    
    # calc center of predicted boxes
    cx_pred, cy_pred = calc_center(boxes)
    # calc center of target boxes
    cx_gt,cy_gt = calc_center(annos)

    isDet = np.zeros(boxes.shape[0] + annos.shape[0])
    isDet[0:boxes.shape[0]] = 1 

    if annos.shape[0] > 0 and boxes.shape[0] > 0:
        cx = np.hstack((cx_pred, cx_gt))
        cy = np.hstack((cy_pred, cy_gt))
        # set up kdtree 
        X = np.dstack((cx, cy))[0]
        # build a KDTree with annotations
        tree = KDTree(X[isDet == 0])

        # query tree with predictions, ind contains the index of the annotation which matches the prediction
        ind,dist = tree.query_radius(X[isDet == 1], r=radius, return_distance = True, sort_results = True)

        # delete doubeling neigbours
        ind = np.array([i[0] if len(i) > 0 else -1 for i in ind])
        dist = np.array([i[0] if len(i) > 0 else -1 for i in dist])

        # Aufbauen eines Dict bei dem die Feldnamen die indice der benachbarten Punkte im kd-tree sind
        # im Feld an Stelle 0 steht der index der Prediction, welche am nächsten an diesem Punkt liegt
        tmp = {}
        for idx,i in enumerate(ind):
            if i != -1:
                if i not in tmp:
                    tmp[i] = [idx,dist[idx]]
                else:
                    if  tmp[i][-1] > dist[idx]:
                        tmp[i] = [idx,dist[idx]]

        gt_scores = []
        pred_scores = []
        for gt_index in tmp.keys():
            pred_index = tmp[gt_index][0]
            pred_scores.append(reg_scores[pred_index])
            gt_scores.append(reg_targets[gt_index])

        return gt_scores, pred_scores, X[isDet == 0], X[isDet == 1]
    
    else:
        return [],[],[],[]