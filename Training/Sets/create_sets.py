from os import listdir, path
from random import shuffle

SEPARATOR = '_'

class Attrib:
    def __init__(self, number, id, path):
        self.number, self.id, self.path = number, id, path

    def __str__(self):
        return 'Number : ' + str(self.number) + ', Id : ' + str(self.id) + ', Path : ' + self.path

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.path == other.path


def listdir_nohidden(_path):
    output = []
    if path.isdir(_path):
        for f in listdir(_path):
            if not f.startswith('.'):
                output.append(f)
    return output


def analyze_image(filepath):
    filename = filepath[1+filepath.rfind('/'):]
    words = filename.split(SEPARATOR)
    words[-1] = words[-1][0:words[-1].find('.')]#Remove extension
    return Attrib(words[0], words[1], filepath)


def load_image(dirs):
    output = []
    for d in dirs:
        i = 0
        for f in listdir_nohidden(d):
            for ff in listdir_nohidden(d+f):
                output.append(analyze_image(d + f + '/' + ff))
                i += 1
        print(d, ' contains ', i, ' items')
    shuffle(output)
    return output


def query(dirs, n, different_than=None):
    output = []
    i = 0

    for f in load_image(dirs):
        if i >= n:
            break
        if different_than is None or (different_than is not None and f not in different_than):
            output.append(f)
            i += 1
    return output


def save_file(path, list):
    with open(path, 'w+') as f:
        for l in list:
            f.write(l.path + '\n')

# The data is from the Machine Learning book
filepath = '/Users/Diego/Github/Digit-Dataset/'
d = [filepath]

nb_training = 270*0.8
nb_validation = 270*0.2

training = query(d, nb_training, different_than=None)
validation = query(d, nb_validation, different_than=training)

print "\nTraining ", len(training), " items\n"
for t in training:
    print(t)
print "\nValidation ", len(validation), " items\n"
for v in validation:
    print(v)

save_file('/Users/Diego/Desktop/training.list', training)
save_file('/Users/Diego/Desktop/validation.list', validation)