# Package nécésaire

import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

# Fonction qui créer la liste des coordonnée des points (landmarks)
def land2coords(landmarks, dtype="int"):
    
    coords = np.zeros((68, 2), dtype=dtype)
    
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    return coords

def moyenne(tableau):
    return sum(tableau, 0.0) / len(tableau)

def variance(tableau):
    m=moyenne(tableau)
    return moyenne([(x-m)**2 for x in tableau])

def ecartype(tableau):
    return variance(tableau)**0.5

def distEucli(p1,p2):
    a = (p2[1]-p1[1])*(p2[1]-p1[1])
    b = (p2[0]-p1[0])*(p2[0]-p1[0])
    
    return np.sqrt(a+b)


if __name__=="__main__":
    
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    vid = cv2.VideoCapture(0)
    
     #Variables
     
    disttemp = 0
    compteurf = 0
    compteurd = 0
    disteye = []
    ancpoint28 = (0,0)
    ancpoint34 = (0,0)
    tetemouvement = []
    nbframe = 0
    i = 1
    newtab = []
    droite = []
    
    while True:
        ret,frame = vid.read()       
        
        if ret == False:
            break
        nbframe = nbframe + 1
    
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        face_boundaries = face_detector(frame_gray,0)

        for (enum,face) in enumerate(face_boundaries):
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            cv2.rectangle(frame, (x,y), (x+w, y+h), (120,160,230),2)

            landmarks = landmark_predictor(frame_gray, face)
            landmarks = land2coords(landmarks)
            
            for (a,b) in landmarks:
                cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)
            
            #Tracer la distance entre les deux paupières.
            
            #Oeil droit
            point38 = landmarks[37]
            point42 = landmarks[41]
            
            point39 = landmarks[38]
            point41 = landmarks[40]
            
            point37 = landmarks[36]
            point40 = landmarks[39]
            
            #Oeil gauche
            point44 = landmarks[43]
            point48 = landmarks[47]
            
            point45 = landmarks[44]
            point47 = landmarks[46]
            
            point43 = landmarks[42]
            point46 = landmarks[45]
            
            point28 = landmarks[27]
            point34 = landmarks[33]
            
            cv2.line(frame, (point38[0], point38[1]), (point42[0], point42[1]), (0, 0, 255))
            cv2.line(frame, (point39[0], point39[1]), (point41[0], point41[1]), (0, 0, 255))
            
            cv2.line(frame, (point44[0], point44[1]), (point48[0], point48[1]), (0, 0, 255))
            cv2.line(frame, (point45[0], point45[1]), (point47[0], point47[1]), (0, 0, 255))
            
            cv2.line(frame, (point28[0], point28[1]), (point34[0], point34[1]), (0, 255, 0))
            
            if(compteurf == 4):
                compteurf = 0
                disteye.append(disttemp)
                disttemp = 0
            else:
                compteurf = compteurf + 1
                disttemp = disttemp +  ((distEucli(point38,point42) + distEucli(point39,point41) + distEucli(point44,point48) + distEucli(point45,point47))/(4*h))
            
            #Calcul mouvement de la tête
            
            
            if( (np.abs(ancpoint28[1])+np.abs(ancpoint28[0])) != 0):
                
                point28 = landmarks[27]
                point34 = landmarks[33]
                
                tetemou = (distEucli(point28,ancpoint28) +distEucli(point34,ancpoint34))/(2*h)
                tetemouvement.append(tetemou)
                
                ancpoint28 = landmarks[27]
                ancpoint34 = landmarks[33]
            else:
                ancpoint28 = landmarks[27]
                ancpoint34 = landmarks[33]
            
            if(nbframe == 20*i):
                tab1=[]
                for a in range(int((nbframe/5)-4),int((nbframe/5)-1)):
                    tab1.append(disteye[a])
                droite.append((max(tab1)+ min(tab1))/2)
                i = i+1
                for p in range(0,4):
                    newtab.append((max(tab1)+ min(tab1))/2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break;
            
        
    vid.release()
    cv2.destroyAllWindows()
    
    fps = vid.get(5)
    if(fps == 0):
        fps = 30
    
    temps = nbframe /fps
    
    milieu = (max(disteye)+ min(disteye))/2
    mi = []
    
    for i in range (len(disteye)):
        mi.append(milieu)
    
    plt.title("Graphe du nombre de clignement")
    plt.plot(disteye)
    plt.plot(newtab)
    plt.plot(mi)
    plt.ylabel('Distance entre les paupières')
    plt.show()
    
    Nbcligno = 0
    
    
   
    comp = 0
    
    for j in droite :
        if(disteye[comp] < j and disteye[comp] < milieu):
            Nbcligno =  Nbcligno + 1
        comp = comp +1
        if(disteye[comp] < j and disteye[comp] < milieu):
            Nbcligno =  Nbcligno + 1
        comp = comp +1
        if(disteye[comp] < j and disteye[comp] < milieu):
            Nbcligno =  Nbcligno + 1
        comp = comp +1
        if(disteye[comp] < j and disteye[comp] < milieu):
            Nbcligno =  Nbcligno + 1
        comp = comp +1
                         
    fcligno = Nbcligno/temps
    mouvementtete = (sum(tetemouvement,0.0)/nbframe)
    ouvertureeye = (sum(disteye, 0.0)/(len(disteye)))
    
    print("Frame vidéo : " , nbframe)
    print("Temps : " , temps , " sec")
    print("Nombres de clignement : ", Nbcligno)
    print("Fréquence de clignement : ", fcligno)
    print("Mouvement de la tête : ", mouvementtete)
    print("Ouverture moyenne : ", ouvertureeye)
    plt.title("Graphe du mouvement de la tête")
    plt.plot(tetemouvement)
    plt.ylabel('Taux de mouvement')
    plt.show()
    
    
    # SVM
    
    
    X = np.array([[ 0.2638867924528302 , 0.0062888365613 , 0.186326871078 ],
             [ 0.3979419087136929 , 0.004011889102 , 0.185774596927 ],
             [ 0.36421875 , 0.00574081288725 , 0.187276115632 ],
             [ 0.7280161943319837 , 0.00932398719322 , 0.194269200023 ],
             [ 0.8071590909090907 , 0.0100628191429 , 0.205719891534 ],
             [ 0.79328414977396825 , 0.013176248959 , 0.197358978748 ]])

    Y = [0,0,0,1,1,1]
    
    clf = svm.SVC(kernel='rbf', C = 1.0)
    
    clf.fit(X,Y)
    
    result=-1
    result = clf.predict([[fcligno,mouvementtete,ouvertureeye]])
    
    if(result == 0):
        print("Non Stressé")
    else:
        if(result == 1):
            print("Stressé")
        else:
            print("Erreur")