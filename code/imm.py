from sklearn.cluster import KMeans
import numpy as np
import uuid


class Node:
    
    def __init__(self):

        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.cluster = None
        self.parent = None
        self.layer = None

class imm(object):
    
    def __init__(self):
        self.tree = None
    
    def fit(self, x, y, u): # fit the tree to the data
        root = Node()
        id = uuid.uuid1() # generate uuid for root node
        root.id = id
        root.layer = 0
        self.tree = self.build_tree(x, y, u, root)
            
    def build_tree(self, x, y, u, node):
        #check if array is homogenous
        y = np.array(y)
        if y.size == 0:
            # No more points
            return node
        first = y[0]
        count = 0
        for label in y:
            if label == first:
                count += 1
        
        if count == len(y):
            # Leaf node
            node.cluster = first
            return node
        
        else:
            
            #populate arrays of r and l
            l = np.zeros(len(x[0]))
            r = np.zeros(len(x[0]))
            
            for i in range(len(x[0])):
                
                arr = np.zeros(len(x))
                for j in range(len(x)):
                    arr[j] = u[y[j]][i]
                
                l[i] = np.amin(arr)
                r[i] = np.amax(arr)
            
            mistakes = []
            cutoffList = np.vstack([l[:], r[:]]).mean(axis = 0)
            
            #iterate through features
            for i in range(len(cutoffList)):
                sum = 0
                for j in range(len(x)):
                    sum += self.mistake(x[j],u[y[j]], i, cutoffList[i])
                mistakes.append(sum)
            
            i = np.argmin(mistakes)
            theta = cutoffList[i]
            
            M = []
            L = []
            R = []
            
            for j in range(len(x)):
                if self.mistake(x[j],u[y[j]], i, theta) == 1:
                    M.append(j)
                elif x[j][i] <= theta:
                    L.append(j)
                elif x[j][i] > theta:
                    R.append(j)
            
            leftx = []
            lefty = []
            rightx = []
            righty = []
            
            for e in range(len(x)):
                if e in L:
                    leftx.append(x[e])
                    lefty.append(y[e])
                elif e in R:
                    rightx.append(x[e])
                    righty.append(y[e])
            
            
            
            node.feature = i
            node.threshold = theta

            left_node = Node()
            left_node.parent = node
            id = uuid.uuid1()
            left_node.id = id
            left_node.layer = node.layer + 1

            right_node = Node()
            right_node.parent = node
            id = uuid.uuid1()
            right_node.id = id
            right_node.layer = node.layer + 1

            node.left = self.build_tree(leftx, lefty, u, left_node)
            node.right = self.build_tree(rightx, righty, u, right_node)
            
            return node
             
    def mistake(self, x , u, i, val): # calculate mistake of a split
            
        if (x[i] <= val) != (u[i] <= val):
            return 1
        else:
            return 0
    
    def predict(self, x): # predict class of instance x using tree 
        
        if self.tree is None:
            raise TypeError('Tree is untrained.')
        else:
            return self.evaluate_tree(self.tree, x) 
        
    def evaluate_tree(self, node, x):
        if node.cluster is not None: # leaf found
            return node.cluster
        
        if x[node.feature] <= node.threshold: # threshold evaluation
            return self.evaluate_tree(node.left, x)
        else:
            return self.evaluate_tree(node.right, x)
        
    def get_path(self, x): # return path to x
        if self.tree is None:
            raise TypeError('Tree is untrained.')
        else:
            return self.traverse_path(self.tree, x)
        
    def traverse_path(self, node, x):
        feature = node.feature
        threshold = node.threshold
        
        if node.cluster is not None:
            return np.array([node])
        
        if x[feature] <= threshold:
            return np.append(np.array([node]), self.traverse_path(node.left, x))
        else:
            return np.append(np.array([node]), self.traverse_path(node.right, x))

    def write_tree(self): # converts tree to a string
        if self.tree is None:
            raise TypeError('Tree is untrained.')
        else:
            return self.write(self.tree)
        
    def write(self, node, prefix="", is_left=True):
        if node is None:
            return ""
        
        result = prefix
        result += "├── " if is_left else "└── "
        result += f"L{node.layer}: f{node.feature} < {node.threshold}" if node.cluster is None else f"L{node.layer}, C: {node.cluster}"
        result += "\n"
        
        if node.left is not None or node.right is not None:
            if node.left is not None:
                result += self.write(node.left, prefix + ("│   " if is_left else "    "), True)
            if node.right is not None:
                result += self.write(node.right, prefix + ("│   " if is_left else "    "), False)
        
        return result