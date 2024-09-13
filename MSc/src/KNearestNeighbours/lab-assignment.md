# First Lab Assignment  
  
For the first lab assignment, you are asked to implement the "decide" method in the code of the KNN script introduced during the course lab session.  
  
The "decide" method of the class determines the final action for a test point \(x\) that maximizes the expected utility \(U\).  
  
In mathematical terms:  
  
$$ \mathop{\mathrm{argmax}}_a E[U| a, x] =  \mathop{\mathrm{argmax}}_a \sum_y U(a,y)P(y|x) $$  
  
Where **U** is the utility matrix.
  
The code that you need to download and complete can be found at the following link: [KNearestNeighbours/NearestNeighbourClassifier.py](https://github.com/olethrosdc/machine-learning-MSc/blob/main/src/KNearestNeighbours/NearestNeighbourClassifier.py)  
  
Additionally, we have included hints within the code to help you with the assignment.
