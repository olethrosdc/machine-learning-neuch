
import numpy as np

class Generator:
    def __init__(n_students, n_schools):
        self.n_students = n_students
        self.n_schools = n_schools
    def generate_data():
        # it should return gender, school choice and admission for all students
        pass

class FairModel(Generator):        
    def gender():
        return np.random.choice(2)
    def grade():
        return np.random.uniform()
    def school():
        return np.random.choice(self.n_schools)
    def acceptance(grade):
        if (grade > 0.5):
            return 1
        else:
            return 0
    def generate_data():
        gender = np.zeros(self.n_students)
        school = np.zeros(self.n_students)
        grade = np.zeros(self.n_students)
        admission = np.zeros(self.n_students)
        for t in range(self.n_students):
            gender[t] = self.gender()
            grade[t] = self.grade()
            school[t] = self.school()
            admission[t] = self.acceptance(grade[t])
        return gender, school, grade, admission
    
class BiasedChoiceModel:
    def __init__(n_students, n_schools):
        self.n_students = n_students
        self.n_schools = n_schools
        self.choice_probabilities = np.zeros(size=[2, self.n_schools])
        self.choice_probabilities[0] = np.linspace(1, n_schools, n_schools)
        self.choice_probabilities[1] = np.linspace(n_schools, 1, n_schools)
        for i in range(2):
            self.choice_probabilities[i] /= np.sum(self.choice_probabilities[i])

    def gender():
        return np.random.choice(2)
    def grade():
        return np.random.uniform()
    def school(gender):
        return np.random.choice(self.n_schools, p = self.choice_probabilities[gender])
    def acceptance(school, grade):
        # you get accepted harder in higher-ranked schools (with greater number)
        if (grade > 1 - 1 / (1+school)):
            return 1
        else:
            return 0
    def generate_data():
        gender = np.zeros(self.n_students)
        school = np.zeros(self.n_students)
        grade = np.zeros(self.n_students)
        admission = np.zeros(self.n_students)
        for t in range(self.n_students):
            gender[t] = self.gender()
            grade[t] = self.grade()
            school[t] = self.school(gender[t])
            admission[t] = self.acceptance(school[t], grade[t])
        return gender, school, grade, admission
    

        
     


    
