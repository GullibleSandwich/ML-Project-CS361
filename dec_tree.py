import numpy as np
import pandas as pd

def RSS_reduction(child_L, child_R, parent):
    rss_parent = sum((parent - np.mean(parent))**2)
    rss_child_L = sum((child_L - np.mean(child_L))**2) 
    rss_child_R = sum((child_R - np.mean(child_R))**2)
    return rss_parent - (rss_child_L + rss_child_R)

def sort_x_by_y(x, y):
    unique_xs = np.unique(x)
    y_mean_by_x = np.array([y[x == unique_x].mean() for unique_x in unique_xs])
    ordered_xs = unique_xs[np.argsort(y_mean_by_x)]
    return ordered_xs

def all_rows_equal(X):
    return (X == X[0]).all()

class Node:
    
    def __init__(self, Xsub, ysub, ID, depth = 0, parent_ID = None, leaf = True):
        self.ID = ID
        self.Xsub = Xsub
        self.ysub = ysub
        self.size = len(ysub)
        self.depth = depth
        self.parent_ID = parent_ID
        self.leaf = leaf
        
class Splitter:
    
    def __init__(self):
        self.rss_reduction = 0
        self.no_split = True
        
    def _replace_split(self, rss_reduction, d, dtype = 'quant', t = None, L_values = None):
        self.rss_reduction = rss_reduction
        self.d = d
        self.dtype = dtype
        self.t = t        
        self.L_values = L_values     
        self.no_split = False

class DecisionTreeRegressor:
    
    def __init__(self, max_depth=100, min_size=2, C=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.C = C
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        dtypes = [np.array(list(self.X[:, d])).dtype for d in range(self.D)]
        self.dtypes = ['quant' if (dtype == float or dtype == int) else 'cat' for dtype in dtypes]
        self.nodes_dict = {}
        self.current_ID = 0
        initial_node = Node(Xsub=X, ysub=y, ID=self.current_ID, parent_ID=None)
        self.nodes_dict[self.current_ID] = initial_node
        self.current_ID += 1
        self._build()
    
    # Buld the decision tree
    def _build(self):

        eligible_buds = self.nodes_dict
        current_depth = 0
        while current_depth < self.max_depth:
            current_depth += 1
            eligible_buds = {ID: node for (ID, node) in self.nodes_dict.items() if 
                                (node.leaf == True) &
                                (node.size >= self.min_size) & 
                                (~all_rows_equal(node.Xsub)) &
                                (len(np.unique(node.ysub)) > 1)}
            if len(eligible_buds) == 0:
                break
            for ID, bud in eligible_buds.items():
                self._find_split(bud)
                if not self.splitter.no_split:
                    self._make_split()

    
    # Find the best split for a node
    def _find_split(self, bud):
        splitter = Splitter()
        splitter.bud_ID = bud.ID
        eligible_predictors = np.random.permutation(np.arange(self.D))[:self.C] if self.C is not None else np.arange(self.D)
        
        for predictor in eligible_predictors:
            X_sub = bud.Xsub[:, predictor]
            dtype = self.dtypes[predictor]
            
            if len(np.unique(X_sub)) == 1:
                continue
            
            if dtype == 'quant':
                thresholds = np.linspace(np.min(X_sub), np.max(X_sub), num=self.C + 1)[1:-1]
                best_rss_reduction = -np.inf
                best_threshold = None
                
                for threshold in thresholds:
                    y_sub_L = bud.ysub[X_sub <= threshold]
                    y_sub_R = bud.ysub[X_sub > threshold]
                    rss_reduction = RSS_reduction(y_sub_L, y_sub_R, bud.ysub)
                    
                    if rss_reduction > best_rss_reduction:
                        best_rss_reduction = rss_reduction
                        best_threshold = threshold
                        
                if best_rss_reduction > splitter.rss_reduction:
                    splitter._replace_split(best_rss_reduction, predictor, dtype='quant', t=best_threshold)
            
            else:
                ordered_values = sort_x_by_y(X_sub, bud.ysub)
                num_splits = min(self.C, len(ordered_values) - 1)
                split_indices = np.random.choice(np.arange(1, len(ordered_values)), size=num_splits, replace=False)
                
                for index in split_indices:
                    L_values = ordered_values[:index]
                    y_sub_L = bud.ysub[np.isin(X_sub, L_values)]
                    y_sub_R = bud.ysub[~np.isin(X_sub, L_values)]
                    rss_reduction = RSS_reduction(y_sub_L, y_sub_R, bud.ysub)
                    
                    if rss_reduction > splitter.rss_reduction:
                        splitter._replace_split(rss_reduction, predictor, dtype='cat', L_values=L_values)
        
        self.splitter = splitter

    # Make split
    def _make_split(self):
        """
        Make a split based on the best split found.
        """
        parent_node = self.nodes_dict[self.splitter.bud_ID]
        parent_node.leaf = False
        parent_node.child_L = self.current_ID
        parent_node.child_R = self.current_ID + 1
        parent_node.d = self.splitter.d
        parent_node.dtype = self.splitter.dtype
        parent_node.t = self.splitter.t        
        parent_node.L_values = self.splitter.L_values
        
        X_sub = parent_node.Xsub[:, parent_node.d]
        
        if parent_node.dtype == 'quant':
            L_condition = X_sub <= parent_node.t
        else:
            L_condition = np.isin(X_sub, parent_node.L_values)
        
        Xchild_L = parent_node.Xsub[L_condition]
        ychild_L = parent_node.ysub[L_condition]
        Xchild_R = parent_node.Xsub[~L_condition]
        ychild_R = parent_node.ysub[~L_condition]
        
        child_node_L = Node(Xchild_L, ychild_L, depth=parent_node.depth + 1,
                            ID=self.current_ID, parent_ID=parent_node.ID)
        child_node_R = Node(Xchild_R, ychild_R, depth=parent_node.depth + 1,
                            ID=self.current_ID + 1, parent_ID=parent_node.ID)
        
        self.nodes_dict[self.current_ID] = child_node_L
        self.nodes_dict[self.current_ID + 1] = child_node_R
        self.current_ID += 2


        # Get leaf node means
        def _get_leaf_means(self):
            self.leaf_means = {}
            for node_ID, node in self.nodes_dict.items():
                if node.leaf:
                    self.leaf_means[node_ID] = node.ysub.mean()

        # Predict using the trained decision tree
        def predict(self, X_test):
            self._get_leaf_means()
            yhat = []
            for x in X_test:
                node = self.nodes_dict[0] 
                while not node.leaf:
                    if node.dtype == 'quant':
                        if x[node.d] <= node.t:
                            node = self.nodes_dict[node.child_L]
                        else:
                            node = self.nodes_dict[node.child_R]
                    else:
                        if x[node.d] in node.L_values:
                            node = self.nodes_dict[node.child_L]
                        else:
                            node = self.nodes_dict[node.child_R]
                yhat.append(self.leaf_means[node.ID])
            return np.array(yhat)